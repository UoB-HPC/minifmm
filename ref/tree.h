#pragma once

#include <stdlib.h>

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
} t_node;

static inline
int is_leaf(t_node* node) { return node->num_children == 0; }

void build_tree(t_fmm_options* options);

void free_tree(t_fmm_options* options);
