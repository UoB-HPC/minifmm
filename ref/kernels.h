#pragma once

#include "params.h"
#include "tree.h"
#include "traversal.h"

void p2p(t_fmm_options* options, t_node* target, t_node* source);

void p2p_intrinsics(t_fmm_options* options, t_node* target, t_node* source);

void p2p_one_node(t_fmm_options* options, t_node* node);

void p2m(t_fmm_options* options, t_node* node);

void m2m(t_fmm_options* options, t_node* node);

void m2l(t_fmm_options* options, t_node* target, t_node* source);

void l2l(t_fmm_options* options, t_node* target, t_node* source);

void l2p(t_fmm_options* options, t_node* node);

void m2p(t_fmm_options* options, t_node* target, t_node* source);

void precompute(t_fmm_options* options);
