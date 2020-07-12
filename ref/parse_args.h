#pragma once

#include <stdlib.h>

#include "params.h"

void parse_fmm_args(int argc, char** argv, t_fmm_params* params);

void check_args(t_fmm_params* params);

void print_args(t_fmm_params* params);

