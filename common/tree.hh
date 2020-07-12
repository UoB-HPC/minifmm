#pragma once

#include <algorithm>
#include <array>
#include <fmm.hh>
#include <iostream>
#include <node.hh>
#include <timer.hh>

template <class T>
void get_bound_box(FMM<T>* fmm, size_t start, size_t end,
                   std::array<std::pair<T*, T*>, 3>& lims)
{
  lims[0] = std::minmax_element(&fmm->x[start], &fmm->x[end]);
  lims[1] = std::minmax_element(&fmm->y[start], &fmm->y[end]);
  lims[2] = std::minmax_element(&fmm->z[start], &fmm->z[end]);
}

// stable in-place mergesort
template <class T>
void reorder(T* x, T* y, T* z, T* w, std::vector<size_t> indices, size_t start,
             size_t end)
{
  // for (size_t i = start; i < end; ++i) {
  //  const T tx = x[i];
  //  const T ty = y[i];
  //  const T tz = z[i];
  //  size_t j = i;
  //  while (true) {
  //    size_t k = indices[j];
  //    indices[j] = j;
  //    if (k == i) break;
  //    if (k >= end) {
  //      printf("problem k = %zu\n", k);
  //      exit(1);
  //    }
  //    x[j] = x[k];
  //    y[j] = y[k];
  //    z[j] = z[k];
  //    j = k;
  //  }
  //  x[j] = tx;
  //  y[j] = ty;
  //  z[j] = tz;
  //}
  std::vector<T> temp(end - start);
  for (size_t i = start; i < end; ++i) temp[i - start] = x[indices[i - start]];
  for (size_t i = start; i < end; ++i) x[i] = temp[i - start];

  for (size_t i = start; i < end; ++i) temp[i - start] = y[indices[i - start]];
  for (size_t i = start; i < end; ++i) y[i] = temp[i - start];

  for (size_t i = start; i < end; ++i) temp[i - start] = z[indices[i - start]];
  for (size_t i = start; i < end; ++i) z[i] = temp[i - start];

  for (size_t i = start; i < end; ++i) temp[i - start] = w[indices[i - start]];
  for (size_t i = start; i < end; ++i) w[i] = temp[i - start];
}

template <class T>
size_t construct_tree(FMM<T>* fmm, std::vector<node_t<T>>& nodes, size_t start,
                      size_t end, int depth, T cx, T cy, T cz, T rad)
{
  const size_t node_idx = nodes.size();
  nodes.push_back(
      node_t<T>(cx, cy, cz, rad, end - start, start, 0, node_idx, depth));

  // const size_t num_points = end - start + 1;
  if (end - start <= fmm->ncrit) {
  }
  else {
    std::vector<size_t> indices(end - start);
    std::vector<size_t> octants(end - start);

    size_t num_oct[8] = {0};
    size_t oct_beg[8] = {0};

    for (size_t i = start; i < end; ++i) {
      const size_t oct =
          ((fmm->x[i] > cx) << 2) | ((fmm->y[i] > cy) << 1) | (fmm->z[i] > cz);
      octants[i - start] = oct;
      num_oct[oct]++;
    }

    std::partial_sum(num_oct, num_oct + 7, oct_beg + 1);

    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&octants](const size_t i, const size_t j) {
                return (octants[i] < octants[j]);
              });
    std::for_each(indices.begin(), indices.end(),
                  [start](size_t& i) { i += start; });

    reorder(fmm->x, fmm->y, fmm->z, fmm->w, indices, start, end);

    size_t child[8] = {0};
    size_t num_children = 0;

    for (size_t i = 0; i < 8; ++i) {
      T nrad = rad / 2.0;
      T ncx = ((i >> 2) & 1) ? (cx + nrad) : (cx - nrad);
      T ncy = ((i >> 1) & 1) ? (cy + nrad) : (cy - nrad);
      T ncz = ((i >> 0) & 1) ? (cz + nrad) : (cz - nrad);

      if (num_oct[i]) {
        // offset oct ptrs by start of the current points array
        child[num_children] = construct_tree(fmm, nodes, start + oct_beg[i],
                                             start + oct_beg[i] + num_oct[i],
                                             depth + 1, ncx, ncy, ncz, nrad);
        num_children++;
      }
    }
    nodes[node_idx].num_children = num_children;
    for (size_t i = 0; i < num_children; ++i) {
      nodes[node_idx].child[i] = child[i];
    }
  }
  return node_idx;
}

template <class T>
void build_tree(FMM<T>* fmm)
{
  Timer timer;
  timer.start();

  std::array<std::pair<T*, T*>, 3> lims;
  get_bound_box(fmm, 0, fmm->num_points, lims);

  T cx = (*lims[0].second + *lims[0].first) / static_cast<T>(2.0);
  T cy = (*lims[1].second + *lims[1].first) / static_cast<T>(2.0);
  T cz = (*lims[2].second + *lims[2].first) / static_cast<T>(2.0);

  std::array<T, 3> radii;
  for (int i = 0; i < 3; ++i) {
    radii[i] = (*lims[i].second - *lims[i].first) / static_cast<T>(2.0);
  }

  T rad = *std::max_element(radii.begin(), radii.end());
  // make sure no points lie on the edge of the node
  rad += std::numeric_limits<T>::epsilon();

  std::vector<node_t<T>> nodes;
  fmm->root =
      construct_tree(fmm, nodes, 0, fmm->num_points, 0, cx, cy, cz, rad);

  fmm->num_nodes = nodes.size();
  fmm->nodes = (node_t<T>*)malloc(sizeof(node_t<T>) * fmm->num_nodes);
  for (size_t n = 0; n < fmm->num_nodes; ++n) {
    fmm->nodes[n] = nodes[n];
  }
  timer.stop();
  // printf("built tree in %fs\n", timer.elapsed());

  printf("num_nodes = %zu\n", fmm->num_nodes);

  printf("root %zu has %zu children\n", fmm->root,
         fmm->nodes[fmm->root].num_children);

  // Now we know the number of nodes we can allocate the multipole storage and
  // assign to each node
  fmm->m = (complex_t<T>*)calloc(fmm->num_multipoles * fmm->num_nodes,
                                 sizeof(complex_t<T>));
  fmm->l = (complex_t<T>*)calloc(fmm->num_multipoles * fmm->num_nodes,
                                 sizeof(complex_t<T>));

  size_t max_depth = 0;
  for (size_t n = 0; n < fmm->num_nodes; ++n) {
    fmm->nodes[n].mult_idx = n * fmm->num_multipoles;
    max_depth = std::max(max_depth, fmm->nodes[n].level);
  }
  printf("max tree depth = %zu\n", max_depth);
}
