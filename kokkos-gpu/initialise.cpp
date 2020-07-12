#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include "params.h"
#include "tree.h"
#include "parse_args.h"
#include "initialise.h"
#include "rng.h"
#include "timer.h"
#include "util.h"

#define S_IDX(n,m) ((n)*(n)+(n)+(m))

void precompute(t_fmm_params* params)
{
    int num_terms = params->num_terms;

    params->inner_factors_real = (TYPE*)malloc(sizeof(TYPE)*num_terms*num_terms);
    params->inner_factors_imag = (TYPE*)malloc(sizeof(TYPE)*num_terms*num_terms);
    params->outer_factors_real = (TYPE*)malloc(sizeof(TYPE)*num_terms*num_terms);
    params->outer_factors_imag = (TYPE*)malloc(sizeof(TYPE)*num_terms*num_terms);
    
    int max = 2*num_terms-1;
    TYPE factorial[max];
    factorial[0] = TYPE_ONE;
    for (int i = 1; i < max; ++i) factorial[i] = i*factorial[i-1];

    for (int n = 0; n < num_terms; ++n)
    {
        for (int m = -n; m <= n; ++m)
        {
            TYPE inner_real; TYPE inner_imag;
            TYPE outer_real; TYPE outer_imag;
            ipow(abs(m), &inner_real, &inner_imag);
            ipow(-abs(m), &outer_real, &outer_imag);
            params->inner_factors_real[S_IDX(n,m)] = neg_pow_n(n)*inner_real/factorial[n+abs(m)];
            params->inner_factors_imag[S_IDX(n,m)] = neg_pow_n(n)*inner_imag/factorial[n+abs(m)];
            params->outer_factors_real[S_IDX(n,m)] = outer_real*factorial[n-abs(m)];
            params->outer_factors_imag[S_IDX(n,m)] = outer_imag*factorial[n-abs(m)];
        }
    }
}

void init_data(t_fmm_params* params)
{
    params->num_nodes = 0;
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

