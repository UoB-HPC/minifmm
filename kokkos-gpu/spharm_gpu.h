#pragma once

#include "type.h"
#include "params.h"

void init_spharm_gpu(t_fmm_params* params);

__device__
void compute_spharm_gpu(int num_terms, TYPE base_r, TYPE mult_r, TYPE theta, TYPE phi, TYPE* factors_real, TYPE* factors_imag, TYPE* Y_rn_real, TYPE* Y_rn_imag);

__device__
void compute_outer_gpu(int num_terms, TYPE r, TYPE theta, TYPE phi, TYPE* outer_real, TYPE* outer_imag);
