#pragma once

#include "params.h"
#include "type.h"

void compute_inner(t_fmm_params* params, TYPE r, TYPE theta, TYPE phi, TYPE_COMPLEX* inner);
void compute_outer(t_fmm_params* params, TYPE r, TYPE theta, TYPE phi, TYPE_COMPLEX* outer);
void compute_inner_deriv(t_fmm_params* params, TYPE r, TYPE theta, TYPE phi, TYPE_COMPLEX* inner, TYPE_COMPLEX* inner_deriv);
void compute_outer_deriv(t_fmm_params* params, TYPE r, TYPE theta, TYPE phi, TYPE_COMPLEX* outer, TYPE_COMPLEX* outer_deriv);

