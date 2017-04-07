#include <stdlib.h>

#include "finalise.h"
#include "tree.h"
#include "params.h"

void free_precompute(t_fmm_options* options)
{
    free(options->C);
    free(options->A);
    free(options->spharm_factor);
}

void free_data(t_fmm_options* options)
{
    free(options->points);
    free(options->points_ordered);
    free(options->weights);
    free(options->weights_ordered);
    free(options->acc);
    free(options->pot);
}

void finalise(t_fmm_options* options)
{
    free_tree(options);
    free_precompute(options);
    free_data(options);
}