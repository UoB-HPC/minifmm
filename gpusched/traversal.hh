#pragma once

#include <omp.h>

#include <kernels.hh>
#include <node.hh>

#include <gpusched.h>

#include <gpu-kernels.hh>

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
__device__ void upwards_pass_gpu_task(worker_t* worker, task_t* task)
{
  FMM<T>* fmm = (FMM<T>*)get_private(task, 0);
  node_t<T>* node = (node_t<T>*)get_private(task, 1);

  for (size_t i = 0; i < node->num_children; ++i) {
    void* args[2] = {fmm, fmm->nodes + node->child[i]};
    generate_task(worker, upwards_pass_gpu_task<T>, 2, args);
  }
  taskwait(worker);

  if (node->is_leaf())
    p2m_gpu<32, 1>(fmm, node);
  else
    m2m_gpu<32, 1>(fmm, node);
}

template <class T>
void upwards_pass_gpu(team_t* h_team, team_t* d_team, FMM<T>*h_fmm, FMM<T>* d_fmm)
{
  const int nargs = 2;
  void* args[nargs] = {d_fmm, h_fmm->nodes + h_fmm->root};

  fork_team<upwards_pass_gpu_task<T>>(h_team, d_team, nargs, args);
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
__device__ void dtt_task(worker_t* worker, task_t* task)
{
  FMM<T>* fmm = (FMM<T>*)get_private(task, 0);
  node_t<T>* target = (node_t<T>*)get_private(task, 1);
  node_t<T>* source = (node_t<T>*)get_private(task, 2);

  T dx = source->cx - target->cx;
  T dy = source->cy - target->cy;
  T dz = source->cz - target->cz;
  T r2 = dx * dx + dy * dy + dz * dz;
  T d1 = source->rad * static_cast<T>(2.0);
  T d2 = target->rad * static_cast<T>(2.0);

  if ((d1 + d2) * (d1 + d2) < fmm->theta2 * r2) {
    m2l_gpu<NTHREADS, NWORKERS>(fmm, target, source);
  }
  else if (source->is_leaf() && target->is_leaf()) {
    if (target == source)
      p2p_gpu<32, 16, 4, NTHREADS, NWORKERS>(fmm, target);
    else
      p2p_gpu<32, 16, 4, NTHREADS, NWORKERS>(fmm, target, source);
  }
  else {
    T target_sz = target->rad;
    T source_sz = source->rad;
    if (source->is_leaf() || ((target_sz >= source_sz) && !target->is_leaf())) {
      for (size_t i = 0; i < target->num_children; ++i) {
        node_t<T>* child = fmm->nodes + target->child[i];
        if (target->num_points > TASK_CUTOFF) {
          void* args[3] = {fmm, child, source};
          generate_task(worker, dtt_task<T>, 3, args);
        }
        else {
          ((void**)task->storage)[1] = child;
          dtt_task<T>(worker, task);
          ((void**)task->storage)[1] = target;
        }
      }
    }
    else {
      for (size_t i = 0; i < source->num_children; ++i) {
        node_t<T>* child = fmm->nodes + source->child[i];
        // void* args[3] = {fmm, target, child};
        // generate_task_cond(worker, dtt_task<T>, 3, args,
        //                   source->num_points > TASK_CUTOFF);
        if (source->num_points > TASK_CUTOFF) {
          void* args[3] = {fmm, target, child};
          generate_task(worker, dtt_task<T>, 3, args);
        }
        else {
          ((void**)task->storage)[2] = child;
          dtt_task<T>(worker, task);
          ((void**)task->storage)[2] = source;
        }
      }
    }
  }
}

template <class T>
void perform_traversals(FMM<T>* fmm)
{
#pragma omp parallel
#pragma omp single
  printf("Running on %d threads\n", omp_get_num_threads());

  Timer timer;
  Timer tot_timer;

//  timer.start();
//  tot_timer.start();
//#pragma omp parallel
//#pragma omp single
//  upwards_pass(fmm, &fmm->nodes[fmm->root]);
//  timer.stop();
//  printf("\n");
//  printf("%-20s %12.8f\n", "Upwards Time (s) ", timer.elapsed());

  //#pragma omp parallel
  //#pragma omp single
  // dual_tree(fmm, &fmm->nodes[fmm->root], &fmm->nodes[fmm->root]);

  FMM<T>* h_fmm;
  FMM<T>* d_fmm;
  init_device_fmm(fmm, &h_fmm, &d_fmm);

  const char* num_blocks_str = getenv("GPUSCHED_NUM_BLOCKS");
  const int num_blocks =
      (num_blocks_str == NULL) ? 56 * 5 : atoi(num_blocks_str);

  team_t* h_team;
  team_t* d_team;
  create_team(num_blocks, 1024 * 512, &h_team, &d_team);

  timer.start();
  tot_timer.start();
  upwards_pass_gpu(h_team, d_team, h_fmm, d_fmm);
  timer.stop();
  printf("\n");
  printf("%-20s %12.8f\n", "Upwards Time (s) ", timer.elapsed());

  const int nargs = 3;
  node_t<T>* d_root_node = h_fmm->nodes + h_fmm->root;
  void* args[nargs] = {d_fmm, d_root_node, d_root_node};

  timer.start();
  fork_team<dtt_task<T>>(h_team, d_team, nargs, args);
  timer.stop();
  printf("%-20s %12.8f\n", "DTT Time (s) ", timer.elapsed());

  fini_device_fmm(fmm, h_fmm, d_fmm);

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
