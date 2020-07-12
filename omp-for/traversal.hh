#pragma once

#include <omp.h>

#include <kernels.hh>
#include <node.hh>
#include <get-deps.hh>

template <class T>
void upwards_pass(FMM<T>* fmm, node_t<T>* node)
{
  for (size_t i = 0; i < node->num_children; ++i) {
#pragma omp task
    upwards_pass(fmm, &fmm->nodes[node->child[i]]);
  }
#pragma omp taskwait

  if (node->is_leaf())
    p2m(fmm, node);
  else
    m2m(fmm, node);
}

template <class T>
void downwards_pass(FMM<T>* fmm, node_t<T>* node)
{
  if (node->is_leaf())
    l2p(fmm, node);
  else {
    l2l(fmm, node);
    for (size_t i = 0; i < node->num_children; ++i) {
#pragma omp task
      downwards_pass(fmm, &fmm->nodes[node->child[i]]);
    }
  }
#pragma omp taskwait
}

template <class T>
void perform_traversals(FMM<T>* fmm)
{
#pragma omp parallel
#pragma omp single
  printf("Running on %d threads\n", omp_get_num_threads());

  Timer timer;
  Timer tot_timer;

  timer.start();
  tot_timer.start();
#pragma omp parallel
#pragma omp single
  upwards_pass(fmm, &fmm->nodes[fmm->root]);
  timer.stop();
  printf("\n");
  printf("%-20s %12.8f\n", "Upwards Time (s) ", timer.elapsed());

  Timer deps_timer;
  deps_timer.start();

  std::vector<std::vector<size_t>> p2p_deps(fmm->num_nodes);
  std::vector<std::vector<size_t>> m2l_deps(fmm->num_nodes);

  get_deps_omp(fmm, &p2p_deps, &m2l_deps);

  deps_timer.stop();
  printf("    %-16s %12.8f\n", "Deps. Time (s) ", deps_timer.elapsed());

  Timer compute_timer;
  compute_timer.start();
#pragma omp parallel for schedule(guided)
  for (size_t i = 0; i < fmm->num_nodes; ++i) {
    node_t<T>* target = &fmm->nodes[i];
    for (size_t j = 0; j < p2p_deps[i].size(); ++j) {
      node_t<T>* source = fmm->nodes + p2p_deps[i][j];
      if (target == source) {
        p2p_tiled(fmm, target);
      }
      else {
        p2p_tiled(fmm, target, source);
      }
    }
  }
  compute_timer.stop();
  printf("    %-16s %12.8f\n", "P2P Time (s) ", compute_timer.elapsed());

  compute_timer.start();
#pragma omp parallel for schedule(guided)
  for (size_t i = 0; i < fmm->num_nodes; ++i) {
    node_t<T>* target = &fmm->nodes[i];
    for (size_t j = 0; j < m2l_deps[i].size(); ++j) {
      node_t<T>* source = fmm->nodes + m2l_deps[i][j];
      m2l(fmm, target, source);
    }
  }
  compute_timer.stop();
  timer.stop();
  printf("    %-16s %12.8f\n", "M2L Time (s) ", compute_timer.elapsed());
  printf("%-20s %12.8f\n", "DTT Time (s) ", timer.elapsed());

  timer.start();
#pragma omp parallel
#pragma omp single
  downwards_pass(fmm, &fmm->nodes[fmm->root]);
  timer.stop();
  printf("%-20s %12.8f\n", "Downwards Time (s) ", timer.elapsed());

  tot_timer.stop();
  printf("--------------------\n");
  printf("%-20s %12.8f\n", "Total Time (s) ", tot_timer.elapsed());
  printf("--------------------\n\n");
}
