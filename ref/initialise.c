#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include "params.h"
#include "tree.h"
#include "parse_args.h"
#include "initialise.h"
#include "rng.h"
#include "timer.h"

    static inline
TYPE_COMPLEX ipow(int n)
{
    TYPE_COMPLEX i;
    i = (n & 0x1) ? I : TYPE_ONE;
    i *= (n & 0x2) ? -TYPE_ONE : TYPE_ONE;
    return i;
}

static inline
TYPE neg_pow_n(int n)
{
    return (TYPE)(1 + ((n & 0x01)*-2));
}

#define S_IDX(n,m) ((n)*(n)+(n)+(m))

void precompute(t_fmm_params* params)
{
    int num_terms = params->num_terms;

    params->inner_factors = (TYPE_COMPLEX*)malloc(sizeof(TYPE_COMPLEX)*num_terms*num_terms);
    params->outer_factors = (TYPE_COMPLEX*)malloc(sizeof(TYPE_COMPLEX)*num_terms*num_terms);
    
    int max = 2*num_terms-1;
    TYPE factorial[max];
    factorial[0] = TYPE_ONE;
    for (int i = 1; i < max; ++i) factorial[i] = i*factorial[i-1];

    for (int n = 0; n < num_terms; ++n)
    {
        for (int m = -n; m <= n; ++m)
        {
            params->inner_factors[S_IDX(n,m)] = neg_pow_n(n)*ipow(abs(m))/factorial[n+abs(m)];
            params->outer_factors[S_IDX(n,m)] = ipow(-abs(m))*factorial[n-abs(m)];
        }
    }
}

void init_data(t_fmm_params* params)
{
    params->points = (TYPE*)malloc(sizeof(TYPE)*params->num_points*3);
    params->points_ordered = (TYPE*)malloc(sizeof(TYPE)*params->num_points*3);
    
    params->weights = (TYPE*)malloc(sizeof(TYPE)*params->num_points);
    params->weights_ordered = (TYPE*)malloc(sizeof(TYPE)*params->num_points);

    params->acc = (TYPE*)malloc(sizeof(TYPE)*params->num_points*3);
    params->pot = (TYPE*)malloc(sizeof(TYPE)*params->num_points);
    
    // seed_rng(time(NULL));
    seed_rng(42);
    for (size_t i = 0; i < params->num_points; ++i) params->weights[i] = 1.0;
    for (size_t i = 0; i < params->num_points*3; ++i) params->points[i] = rand_range(-1.0, 1.0);
    for (size_t i = 0; i < params->num_points*3; ++i) params->acc[i] = 0.0;
    for (size_t i = 0; i < params->num_points; ++i) params->pot[i] = 0.0;

    for (size_t i = 0; i < params->num_points; ++i) params->weights_ordered[i] = params->weights[i];
    for (size_t i = 0; i < params->num_points*3; ++i) params->points_ordered[i] = params->points[i];
}

void initialise(int argc, char** argv, t_fmm_params* params)
{
    parse_fmm_args(argc, argv, params);
    print_args(params);
    check_args(params);

    // TODO change to correct value
    params->num_multipoles = params->num_terms*params->num_terms;
    params->num_spharm_terms = params->num_terms*params->num_terms;

    init_data(params);
    
    precompute(params);

    t_timer t;
    start(&t);
    build_tree(params);
    stop(&t);
    printf("Built tree structure in %fs\n", t.elapsed); 

    params->theta2 = params->theta*params->theta;
}

