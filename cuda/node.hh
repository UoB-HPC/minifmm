#pragma once

#include <omp.h>

template <class T>
struct node_t {
  node_t() = default;
  ~node_t() = default;
  node_t(T arg_cx, T arg_cy, T arg_cz, T arg_rad, size_t arg_num_points,
         size_t arg_point_idx, size_t arg_mult_idx, size_t arg_node_idx,
         size_t arg_level)
      : cx{arg_cx},
        cy{arg_cy},
        cz{arg_cz},
        rad{arg_rad},
        num_points{arg_num_points},
        point_idx{arg_point_idx},
        mult_idx{arg_mult_idx},
        node_idx{arg_node_idx},
        level{arg_level}
  {
    omp_init_lock(&p2p_lock);
    omp_init_lock(&m2l_lock);
  }
  T cx;
  T cy;
  T cz;
  T rad;
  size_t num_children = 0;
  size_t child[8] = {0};
  size_t num_points;
  size_t point_idx;
  size_t mult_idx;
  size_t node_idx;
  size_t level;
  omp_lock_t p2p_lock;
  omp_lock_t m2l_lock;

  bool is_leaf() const { return (num_children == 0); }
};
