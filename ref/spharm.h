#pragma once

#include "params.h"
#include "type.h"

void spharm_r_n(t_fmm_options* options, TYPE base_r, TYPE mult_r, TYPE theta, TYPE phi, TYPE_COMPLEX* Y_rn);

void spharm_r_n_d_theta(t_fmm_options* options, TYPE base_r, TYPE mult_r, TYPE theta, TYPE phi, TYPE_COMPLEX* Y_rn, TYPE_COMPLEX* Y_rn_div_theta);