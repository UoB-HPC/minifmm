#include <stdlib.h>

#include "finalise.h"
#include "tree.h"
#include "params.h"

void free_precompute(t_fmm_params* params)
{
    free(params->inner_factors_real);
    free(params->outer_factors_real);
    free(params->inner_factors_imag);
    free(params->outer_factors_imag);
}

void free_data(t_fmm_params* params)
{
    free(params->points);
    free(params->points_ordered);
    free(params->weights);
    free(params->weights_ordered);
    free(params->acc);
    free(params->pot);
}

void finalise(t_fmm_params* params)
{
    //free_tree(params);
    free_precompute(params);
    free_data(params);
}
