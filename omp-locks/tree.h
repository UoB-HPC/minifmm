#pragma once

#include <stdlib.h>
#include <omp.h>

#include "type.h"
#include "params.h"

typedef struct t_node
{
    TYPE center[3];
    size_t num_children;
    struct t_node* child[8];
    size_t num_points;
    TYPE rad;
    TYPE* x;
    TYPE* y;
    TYPE* z;
    TYPE* w;
    TYPE* ax;
    TYPE* ay;
    TYPE* az;
    TYPE* p;
    TYPE_COMPLEX* M;
    TYPE_COMPLEX* L;
    size_t arr_idx;
    omp_lock_t p2p_lock;
    omp_lock_t m2l_lock;
} t_node;

static inline
int is_leaf(t_node* node) { return node->num_children == 0; }

void build_tree(t_fmm_params* params);

void free_tree(t_fmm_params* params);

void tree_to_result(t_fmm_params* params);
