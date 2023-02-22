/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Anders Johansson (Harvard)
------------------------------------------------------------------------- */

#include <pair_kliffnequip.h>
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "potential_file_reader.h"
#include "tokenizer.h"

#include <cmath>
#include <cstring>
#include <numeric>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
//#include <c10/cuda/CUDACachingAllocator.h>


using namespace LAMMPS_NS;

PairKLIFFNEQUIP::PairKLIFFNEQUIP(LAMMPS *lmp) : Pair(lmp) {
  restartinfo = 0;
  manybody_flag = 1;

  if(torch::cuda::is_available()){
    device = torch::kCUDA;
  }
  else {
    device = torch::kCPU;
  }
  std::cout << "NEQUIP is using device " << device << "\n";

  if(const char* env_p = std::getenv("NEQUIP_DEBUG")){
    std::cout << "PairKLIFFNEQUIP is in DEBUG mode, since NEQUIP_DEBUG is in env\n";
    debug_mode = 1;
  }
}

PairKLIFFNEQUIP::~PairKLIFFNEQUIP(){
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(type_mapper);
  }
}

void PairKLIFFNEQUIP::init_style(){
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style NEQUIP requires atom IDs");

  // need a full neighbor list
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;

  neighbor->requests[irequest]->ghost = 1;

  if (force->newton_pair == 1)
    error->all(FLERR,"Pair style NEQUIP requires newton pair off");
}

double PairKLIFFNEQUIP::init_one(int i, int j)
{
  return cutoff;
}

void PairKLIFFNEQUIP::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(type_mapper, n+1, "pair:type_mapper");

}

void PairKLIFFNEQUIP::settings(int narg, char ** /*arg*/) {
  // "flare" should be the only word after "pair_style" in the input file.
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style command");
}

void PairKLIFFNEQUIP::coeff(int narg, char **arg) {

  if (!allocated)
    allocate();

  int ntypes = atom->ntypes;

  // Should be exactly 3 arguments following "pair_coeff" in the input file
  if (narg != (3+ntypes))
    error->all(FLERR, "Incorrect args for pair coefficients");

  // Ensure I,J args are "* *".
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
      setflag[i][j] = 0;

  // Parse the definition of each atom type
  char **elements = new char*[ntypes+1];
  for (int i = 1; i <= ntypes; i++){
      elements[i] = new char [strlen(arg[i+2])+1];
      strcpy(elements[i], arg[i+2]);
      if (screen) fprintf(screen, "NequIP Coeff: type %d is element %s\n", i, elements[i]);
  }

  // Initiate type mapper
  for (int i = 1; i<= ntypes; i++){
      type_mapper[i] = -1;
  }

  std::cout << "Loading model from " << arg[2] << "\n";

  std::unordered_map<std::string, std::string> metadata = {
    {"config", ""},
    {"nequip_version", ""},
    {"r_max", ""},
    {"n_species", ""},
    {"type_names", ""},
    {"_jit_bailout_depth", ""},
    {"_jit_fusion_strategy", ""},
    {"allow_tf32", ""}
  };
  model = torch::jit::load(std::string(arg[2]), device, metadata);
  model.eval();

  if (model.hasattr("training")) {
    std::cout << "Freezing TorchScript model...\n";
    #ifdef DO_TORCH_FREEZE_HACK
      // Do the hack
      // Copied from the implementation of torch::jit::freeze,
      // except without the broken check
      // See https://github.com/pytorch/pytorch/blob/dfbd030854359207cb3040b864614affeace11ce/torch/csrc/jit/api/module.cpp
      bool optimize_numerics = true;  // the default
      // the {} is preserved_attrs
      auto out_mod = freeze_module(
        model, {}
      );
      // See 1.11 bugfix in https://github.com/pytorch/pytorch/pull/71436
      auto graph = out_mod.get_method("forward").graph();
      OptimizeFrozenGraph(graph, optimize_numerics);
      model = out_mod;
    #else
      // Do it normally
      model = torch::jit::freeze(model);
    #endif
  }

  #if (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR <= 10)
    // Set JIT bailout to avoid long recompilations for many steps
    size_t jit_bailout_depth;
    if (metadata["_jit_bailout_depth"].empty()) {
      // This is the default used in the Python code
      jit_bailout_depth = 2;
    } else {
      jit_bailout_depth = std::stoi(metadata["_jit_bailout_depth"]);
    }
    torch::jit::getBailoutDepth() = jit_bailout_depth;
  #else
    // In PyTorch >=1.11, this is now set_fusion_strategy
    torch::jit::FusionStrategy strategy;
    if (metadata["_jit_fusion_strategy"].empty()) {
      // This is the default used in the Python code
      strategy = {{torch::jit::FusionBehavior::DYNAMIC, 3}};
    } else {
      std::stringstream strat_stream(metadata["_jit_fusion_strategy"]);
      std::string fusion_type, fusion_depth;
      while(std::getline(strat_stream, fusion_type, ',')) {
        std::getline(strat_stream, fusion_depth, ';');
        strategy.push_back({fusion_type == "STATIC" ? torch::jit::FusionBehavior::STATIC : torch::jit::FusionBehavior::DYNAMIC, std::stoi(fusion_depth)});
      }
    }
    torch::jit::setFusionStrategy(strategy);
  #endif

  // Set whether to allow TF32:
  bool allow_tf32;
  if (metadata["allow_tf32"].empty()) {
    // Better safe than sorry
    allow_tf32 = false;
  } else {
    // It gets saved as an int 0/1
    allow_tf32 = std::stoi(metadata["allow_tf32"]);
  }
  // See https://pytorch.org/docs/stable/notes/cuda.html
  at::globalContext().setAllowTF32CuBLAS(allow_tf32);
  at::globalContext().setAllowTF32CuDNN(allow_tf32);

  // cutoff = std::stod(metadata["r_max"]);
    cutoff = 4.0 * 3;
  // match the type names in the pair_coeff to the metadata
  // to construct a type mapper from LAMMPS type to NequIP atom_types
  int n_species = 1; //std::stod(metadata["n_species"]);
  std::stringstream ss;
  ss << "Si";//metadata["type_names"];
  for (int i = 0; i < n_species; i++){
      char ele[100];
      ss >> ele;
      for (int itype = 1; itype <= ntypes; itype++)
          if (strcmp(elements[itype], ele) == 0)
              type_mapper[itype] = i;
  }

  // set setflag i,j for type pairs where both are mapped to elements
  for (int i = 1; i <= ntypes; i++)
    for (int j = i; j <= ntypes; j++)
        if ((type_mapper[i] >= 0) && (type_mapper[j] >= 0))
            setflag[i][j] = 1;

  if (elements){
      for (int i=1; i<ntypes; i++)
          if (elements[i]) delete [] elements[i];
      delete [] elements;
  }

}

// Force and energy computation
void PairKLIFFNEQUIP::compute(int eflag, int vflag){
  ev_init(eflag, vflag);

  // Get info from lammps:
  // Atom positions, including ghost atoms

  double **x = atom->x;
  // Atom forces
  double **f = atom->f;
  // Atom IDs, unique, reproducible, the "real" indices
  // Probably 1-based
  tagint *tag = atom->tag;
  // Atom types, 1-based
  int *type = atom->type;
  // Number of local/real atoms
  int nlocal = atom->nlocal;
  // Whether Newton is on (i.e. reverse "communication" of forces on ghost atoms).
  int newton_pair = force->newton_pair;
  // Should probably be off.
  if (newton_pair==1)
    error->all(FLERR,"Pair style NEQUIP requires 'newton off'");

  // Number of local/real atoms
  int inum = list->inum;
  assert(inum==nlocal); // This should be true, if my understanding is correct
  // Number of ghost atoms
  int nghost = list->gnum;
  // Total number of atoms
  int ntotal = inum + nghost;
  // Mapping from neigh list ordering to x/f ordering
  int *ilist = list->ilist;
  // Number of neighbors per atom
  int *numneigh = list->numneigh;
  // Neighbor list per atom
  int **firstneigh = list->firstneigh;

  // Total number of bonds (sum of number of neighbors)
  int nedges = std::accumulate(numneigh, numneigh+ntotal, 0);
  torch::Tensor pos_tensor = torch::from_blob(x[0], {ntotal, 3}, torch::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
  pos_tensor.retain_grad();
    std::vector<torch::IValue> input_vector(6);
  
    int n_layers = 3; // TODO: make this an input
    std::tuple<int, int> bond_pair, rev_bond_pair;
    std::vector<std::set<std::tuple<long, long> > > unrolled_graph(n_layers);
    std::vector<int> next_list, prev_list;

    double cutoff_sq = 4.0 * 4.0;//cutoff * cutoff; // TODO: make this an input
    for (int atom_i = 0; atom_i < nlocal; atom_i++) {
        prev_list.push_back(atom_i);
        for (int i = 0; i < n_layers; i++) {
            if (!prev_list.empty()) {
                do {
                    int curr_atom = prev_list.back();
                    prev_list.pop_back();
                    int numberOfNeighbors = numneigh[curr_atom];
                    for (int j = 0; j < numberOfNeighbors; j++) {
                        if ((std::pow(x[curr_atom][0] - x[firstneigh[curr_atom][j]][0], 2) +
                             std::pow(x[curr_atom][1] - x[firstneigh[curr_atom][j]][1], 2) +
                             std::pow(x[curr_atom][2] - x[firstneigh[curr_atom][j]][2], 2))
                            <= cutoff_sq) {
                            bond_pair = std::make_tuple(curr_atom, firstneigh[curr_atom][j]);
                            rev_bond_pair = std::make_tuple(firstneigh[curr_atom][j], curr_atom);
                            unrolled_graph[i].insert(bond_pair);
                            unrolled_graph[i].insert(rev_bond_pair);
                            next_list.push_back((firstneigh[curr_atom][j]));
                        }
                    }
                } while (!prev_list.empty());
                prev_list.swap(next_list);
            }
        }
        prev_list.clear();
    }

    long **graph_edge_indices = new long *[n_layers];
    int iii = 0;
    for (auto const &edge_index_set: unrolled_graph) {
        int jjj = 0;
        int graph_size = static_cast<int>(edge_index_set.size());
        // Sanitize previous graph
        graph_edge_indices[iii] = new long[graph_size * 2];
        for (auto bond_pair: edge_index_set) {
            graph_edge_indices[iii][jjj] = std::get<0>(bond_pair);
            graph_edge_indices[iii][jjj + graph_size] = std::get<1>(bond_pair);
            jjj++;
        }
        iii++;
    }


    
    int64_t * contraction_array = new int64_t[ntotal];
    for (int i = 0; i < ntotal; i++) {
        contraction_array[i] = (i < nlocal) ? 0 : 1;
    }

    int64_t * species_atomic_number = new int64_t[ntotal];
    for (int i = 0; i < ntotal; i++) {
            species_atomic_number[i] = 0;
    }
    
    torch::Tensor contraction_tensor = torch::from_blob(contraction_array, {ntotal}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor species_tensor = torch::from_blob(species_atomic_number, {ntotal}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor edge_index0 = torch::from_blob(graph_edge_indices[0], {2, unrolled_graph[0].size()}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor edge_index1 = torch::from_blob(graph_edge_indices[1], {2, unrolled_graph[1].size()}, torch::TensorOptions().dtype(torch::kInt64));
    torch::Tensor edge_index2 = torch::from_blob(graph_edge_indices[2], {2, unrolled_graph[2].size()}, torch::TensorOptions().dtype(torch::kInt64));

    input_vector[0] = species_tensor.to(device);
    input_vector[1] = pos_tensor.to(device);
    input_vector[2] = edge_index0.to(device);
    input_vector[3] = edge_index1.to(device);
    input_vector[4] = edge_index2.to(device);
    input_vector[5] = contraction_tensor.to(device);
        
  auto output = model.forward(input_vector).toTuple()->elements();

  torch::Tensor forces_tensor = output[1].toTensor().cpu();
  auto forces = forces_tensor.accessor<double, 2>();

  torch::Tensor energy_tensor = output[0].toTensor();
  torch::Tensor total_energy_tensor = energy_tensor.sum().cpu();

  // store the total energy where LAMMPS wants it
  eng_vdwl = total_energy_tensor.data_ptr<double>()[0];

  torch::Tensor atomic_energy_tensor = energy_tensor.cpu();
  auto atomic_energies = atomic_energy_tensor.accessor<double, 2>();
  double atomic_energy_sum = atomic_energy_tensor.sum().data_ptr<double>()[0];

  if(debug_mode){
    std::cout << "NequIP model output:\n";
    std::cout << "forces: " << forces_tensor << "\n";
    std::cout << "total_energy: " << total_energy_tensor << "\n";
    std::cout << "atomic_energy: " << atomic_energy_tensor << "\n";
  }

  // Write forces and per-atom energies (0-based tags here)
  // for(int itag = 0; itag < inum; itag++){
  for(int itag = 0; itag < ntotal; itag++){
    f[itag][0] = -forces[itag][0];
    f[itag][1] = -forces[itag][1];
    f[itag][2] = -forces[itag][2];
    if (eflag_atom) eatom[itag] = atomic_energies[itag][0];
  }

  for (int i = 0; i < n_layers; i++) {
    delete[] graph_edge_indices[i];
  }
  delete[] contraction_array;
  delete[] species_atomic_number;
}
