#include <stdlib.h>

#include "params.h"
#include "tree.h"
#include "parse_args.h"
#include "initialise.h"
#include "rng.h"

void precompute(t_fmm_options* options)
{
    int num_terms = options->num_terms;

    options->C = (TYPE_COMPLEX*)malloc(sizeof(TYPE_COMPLEX)*num_terms*num_terms*num_terms*num_terms);
    options->A = (TYPE*)malloc(sizeof(TYPE)*2*num_terms*num_terms);
    options->spharm_factor = (TYPE*)malloc(sizeof(TYPE)*num_terms*num_terms);

    for (int n = 0; n < num_terms; ++n)
    {
        for (int m = -n; m <= n; ++m)
        {
            TYPE sign = (n&1) ? -1.0 : 1.0;
            TYPE fact1 = 1.0, fact2 = 1.0;
            for (int i = 1; i <= n-m; ++i) fact1 *= i;
            for (int i = 1; i <= n+m; ++i) fact2 *= i;
            options->A[n*n+n+m] = sign/TYPE_SQRT(fact1*fact2);

            fact1 = 1.0; fact2 = 1.0;
            for (int i = 1; i <= n-abs(m); ++i) fact1 *= i;
            for (int i = 1; i <= n+abs(m); ++i) fact2 *= i;
            options->spharm_factor[n*n+n+m] = TYPE_SQRT(fact1/fact2);
        }
    }


    for (int j = 0, jk = 0, jknm = 0; j < num_terms; ++j)
    {
        for (int k = -j; k <= j; ++k, ++jk)
        {
            for (int n = 0, nm = 0; n < num_terms; ++n)
            {
                for (int m=-n; m <= n; ++m, ++jknm, ++nm)
                {
                    TYPE sign = (j & 1) ? -1.0 : 1.0;
                    int jnkm = (j+n)*(j+n)+(j+n)+(m-k);
                    options->C[jknm] = TYPE_CPOW(I, abs(k-m)-abs(k)-abs(m)) * 
                        (sign*options->A[nm]*options->A[jk]/options->A[jnkm]);
                        // (sign*compute_a(n, m) *compute_a(j,k)/compute_a(j+n,m-k));
                    
                }
            }
        }
    }
}

void init_data(t_fmm_options* options)
{
    options->points = (TYPE*)malloc(sizeof(TYPE)*options->num_points*3);
    options->points_ordered = (TYPE*)malloc(sizeof(TYPE)*options->num_points*3);
    
    options->weights = (TYPE*)malloc(sizeof(TYPE)*options->num_points);
    options->weights_ordered = (TYPE*)malloc(sizeof(TYPE)*options->num_points);

    options->acc = (TYPE*)malloc(sizeof(TYPE)*options->num_points*3);
    options->pot = (TYPE*)malloc(sizeof(TYPE)*options->num_points);
    
    // seed_rng(time(NULL));
    seed_rng(42);
    for (size_t i = 0; i < options->num_points; ++i) options->weights[i] = 1.0;
    for (size_t i = 0; i < options->num_points*3; ++i) options->points[i] = rand_range(-1.0, 1.0);
    for (size_t i = 0; i < options->num_points*3; ++i) options->acc[i] = 0.0;
    for (size_t i = 0; i < options->num_points; ++i) options->pot[i] = 0.0;
}

void initialise(int argc, char** argv, t_fmm_options* options)
{
    parse_fmm_args(argc, argv, options);
    print_args(options);
    check_args(options);

    // change to correct value
    options->num_multipoles = options->num_terms*options->num_terms;
    options->num_spharm_terms = options->num_terms*options->num_terms;

    init_data(options);
    
    precompute(options);

    build_tree(options); 
}
