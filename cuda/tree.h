#pragma once

#include <stdlib.h>

#include "type.h"
#include "params.h"

typedef struct t_node
{
    TYPE center[3];
    size_t num_children;
    size_t child[8];
    size_t num_points;
    TYPE rad;
    //TYPE* x;
    //TYPE* y;
    //TYPE* z;
    //TYPE* w;
    //TYPE* ax;
    //TYPE* ay;
    //TYPE* az;
    //TYPE* p;
    ////TYPE_COMPLEX* M;
    ////TYPE_COMPLEX* L;
    //TYPE* M_real;
    //TYPE* M_imag;
    //TYPE* L_real;
    //TYPE* L_imag;
    size_t arr_idx;
    int level;
    int num_desc_nodes;

    size_t point_idx;
    size_t mult_idx;
    size_t offset;
} t_node;

static inline __host__ __device__
int is_leaf(t_node* node) { return node->num_children == 0; }

void build_tree(t_fmm_params* params);

void free_tree(t_fmm_params* params);

void tree_to_result(t_fmm_params* params);

static inline
t_node* get_node(t_fmm_params* params, size_t node) 
{ 
    return &params->node_array[node]; 
}
