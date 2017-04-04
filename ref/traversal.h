#pragma once

#include "params.h"

void perform_fmm(t_fmm_options* options);

void dual_tree_traversal(t_fmm_options* options);

void calc_local_expansions(t_fmm_options* options);

void calc_multipoles_at_nodes(t_fmm_options* options);
