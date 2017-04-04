#pragma once

#include <stdlib.h>

#include "params.h"

#if defined(__unix__) || defined(__linux__) || defined(__APPLE__)
#define __FMM_POSIX
#include <unistd.h>
#include <getopt.h>

void parse_fmm_args(int argc, char** argv, t_fmm_options* options);

void check_args(t_fmm_options* options);

void print_args(t_fmm_options* options);

#else 
    #error [parse_args.h] non-posix platforms are not supported
#endif