#pragma once

#include "params.h"
#include "type.h"

void compute_inner(t_fmm_params* params, TYPE r, TYPE theta, TYPE phi, TYPE* inner_real, TYPE* inner_imag);
void compute_outer(t_fmm_params* params, TYPE r, TYPE theta, TYPE phi, TYPE* outer_real, TYPE* outer_imag);
void compute_inner_deriv(t_fmm_params* params, TYPE r, TYPE theta, TYPE phi, TYPE* inner_real, TYPE* inner_imag, TYPE* inner_deriv_real, TYPE* outer_deriv_imag);
void compute_outer_deriv(t_fmm_params* params, TYPE r, TYPE theta, TYPE phi, TYPE* outer_real, TYPE* outer_imag, TYPE* outer_deriv_real, TYPE* outer_deriv_imag);

