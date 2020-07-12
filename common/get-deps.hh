#pragma once

#include <cstdlib>
#include <vector>

#include <node.hh>
#include <fmm.hh>

template <class T>
void get_deps_omp_task(FMM<T>* fmm, node_t<T>* target, node_t<T>* source,
                       std::vector<std::vector<size_t>>* p2p_deps,
                       std::vector<std::vector<size_t>>* m2l_deps)
{
  T dx = source->cx - target->cx;
  T dy = source->cy - target->cy;
  T dz = source->cz - target->cz;
  T r2 = dx * dx + dy * dy + dz * dz;
  T d1 = source->rad * static_cast<T>(2.0);
  T d2 = target->rad * static_cast<T>(2.0);

  if ((d1 + d2) * (d1 + d2) < fmm->theta2 * r2) {
    omp_set_lock(&target->m2l_lock);
    (*m2l_deps)[target->node_idx].push_back(source->node_idx);
    omp_unset_lock(&target->m2l_lock);
  }
  else if (source->is_leaf() && target->is_leaf()) {
    omp_set_lock(&target->p2p_lock);
    (*p2p_deps)[target->node_idx].push_back(source->node_idx);
    omp_unset_lock(&target->p2p_lock);
  }
  else {
    T target_sz = target->rad;
    T source_sz = source->rad;
    if (source->is_leaf() || ((target_sz >= source_sz) && !target->is_leaf())) {
      for (size_t i = 0; i < target->num_children; ++i) {
        node_t<T>* child = &fmm->nodes[target->child[i]];
#pragma omp task if (target->num_points > TASK_CUTOFF)
        get_deps_omp_task(fmm, child, source, p2p_deps, m2l_deps);
      }
    }
    else {
      for (size_t i = 0; i < source->num_children; ++i) {
        //#pragma omp task if(source->num_points > TASK_CUTOFF &&
        // SOURCE_TASK_SPAWN)
        node_t<T>* child = &fmm->nodes[source->child[i]];
        get_deps_omp_task(fmm, target, child, p2p_deps, m2l_deps);
      }
    }
  }
}

template <class T>
void get_deps_omp(FMM<T>* fmm, std::vector<std::vector<size_t>>* p2p_deps,
                  std::vector<std::vector<size_t>>* m2l_deps)
{
  node_t<T>* root_node = fmm->nodes + fmm->root;
#pragma omp parallel
#pragma omp single
  get_deps_omp_task(fmm, root_node, root_node, p2p_deps, m2l_deps);
}

void pack_deps(std::vector<std::vector<size_t>>& deps, size_t** ret_nodes,
               size_t** ret_deps, size_t** ret_offsets, size_t** ret_sizes,
               size_t* ret_count, size_t* ret_num_nodes)
{
  size_t* prefixes1d = (size_t*)malloc(sizeof(size_t) * deps.size());
  size_t* prefixes2d = (size_t*)malloc(sizeof(size_t) * deps.size());

  prefixes1d[0] = 0;
  prefixes2d[0] = 0;

  size_t count = 0;
  size_t num_nodes = 0;

  count += deps[0].size();
  if (deps[0].size() > 0) num_nodes++;

  for (size_t i = 1; i < deps.size(); ++i) {
    size_t flag = (deps[i - 1].size() > 0);
    prefixes1d[i] = flag + prefixes1d[i - 1];
    prefixes2d[i] = deps[i - 1].size() + prefixes2d[i - 1];
    count += deps[i].size();
    if (deps[i].size() > 0) num_nodes++;
  }

  size_t* deps_array = (size_t*)malloc(sizeof(size_t) * count);
  size_t* nodes = (size_t*)malloc(sizeof(size_t) * num_nodes);
  size_t* sizes = (size_t*)malloc(sizeof(size_t) * num_nodes);
  size_t* offsets = (size_t*)malloc(sizeof(size_t) * num_nodes);

#pragma omp parallel for
  for (size_t i = 0; i < deps.size(); ++i) {
    if (deps[i].size() > 0) {
      size_t beg = prefixes2d[i];
      size_t end = beg + deps[i].size();
      size_t c = 0;
      for (size_t j = beg; j < end; ++j) {
        deps_array[j] = deps[i][c++];
      }
      const size_t idx = prefixes1d[i];
      sizes[idx] = deps[i].size();
      offsets[idx] = beg;
      nodes[idx] = i;
    }
  }

  free(prefixes1d);
  free(prefixes2d);

  *ret_nodes = nodes;
  *ret_deps = deps_array;
  *ret_offsets = offsets;
  *ret_sizes = sizes;
  *ret_count = count;
  *ret_num_nodes = num_nodes;
}
