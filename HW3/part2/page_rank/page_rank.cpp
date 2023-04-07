#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"
#include <vector>
#include <stdint.h>



// inline bool atomic_compare_and_swap(double& val, double old_val, double new_val){
//   uint64_t* ptr = reinterpret_cast<uint64_t*>(&val);
//   const uint64_t* old_val_ptr = reinterpret_cast<const uint64_t*>(&old_val);
//   const uint64_t* new_val_ptr = reinterpret_cast<const uint64_t*>(&new_val);
//   return __sync_bool_compare_and_swap(ptr, *old_val_ptr, *new_val_ptr);
// }



// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  double* score_old = (double*) malloc(numNodes * sizeof(double));
  // calculate number of edges leaving vj
  double* num_leaving_vj = (double*) malloc(numNodes * sizeof(double));
  // collect all vertex that with no outgoing edges
  bool* no_outgoing_edges = (bool*) malloc(numNodes * sizeof(bool));
  double next_no_outgoing_sum = 0.0;

#pragma omp parallel for reduction(+:next_no_outgoing_sum)
  for (int i = 0; i < numNodes; ++i){
    score_old[i] = equal_prob;
    num_leaving_vj[i] = (double)(outgoing_size(g, i));
    if(outgoing_size(g, i) == 0){
      no_outgoing_edges[i] = true;
      next_no_outgoing_sum += (damping * score_old[i] / numNodes);
    }
    else{
      no_outgoing_edges[i] = false;
    }
  }
  

  double current_no_outgoing_sum;
  double global_diff = 1e9;
  double tmp;
  const Vertex* start;
  const Vertex* end;
  const Vertex* v;
  
  
  while(global_diff >= convergence){

    global_diff = 0.0;
    
    current_no_outgoing_sum = next_no_outgoing_sum;
    next_no_outgoing_sum = 0.0;

    #pragma omp parallel for private(start, end, tmp, v) reduction(+:global_diff, next_no_outgoing_sum)
    for(int i=0; i<numNodes; i++){

      start = incoming_begin(g, i);
      end = incoming_end(g, i);
      tmp = 0.0;
      int thread_id = omp_get_thread_num();

      for (v=start; v!=end; v++){
        tmp += score_old[*v]/num_leaving_vj[*v];
      }
      solution[i] = tmp;
      solution[i] = (damping * solution[i]) + (1.0 - damping) / numNodes;
      solution[i] += current_no_outgoing_sum;
    
      global_diff += abs(score_old[i] - solution[i]);
      if(no_outgoing_edges[i]){
        next_no_outgoing_sum += (damping * solution[i] / numNodes);
      }
    }

    #pragma omp parallel for
    for(int i=0; i<numNodes; i++){
      score_old[i] = solution[i];
    }

  }


  free(score_old);
  free(num_leaving_vj);
  free(no_outgoing_edges);


  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
