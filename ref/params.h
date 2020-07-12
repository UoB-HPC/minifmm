#pragma once

#include "type.h"

struct t_node;

typedef struct 
{
    struct t_node* root;
    TYPE* points;
    TYPE* weights;
    TYPE* points_ordered;
    TYPE* weights_ordered;
    size_t num_points;
    size_t ncrit;
    int num_terms;
    TYPE theta;
    TYPE theta2;
    size_t num_samples;

    size_t num_multipoles;
    size_t num_nodes;
    size_t num_spharm_terms;

    TYPE* acc;
    TYPE* pot;

    TYPE_COMPLEX* inner_factors;
    TYPE_COMPLEX* outer_factors;
} t_fmm_params;

