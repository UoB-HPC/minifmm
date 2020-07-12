#include <stdlib.h>

#include "type.h"
#include "tree.h"
#include "kernels.h"
#include "traversal.h"
#include "dtt.h"
#include "timer.h"

void calc_multipoles_at_nodes_core(t_fmm_params* params, t_node* node)
{
    if (node == NULL) return;

    for (size_t i = 0; i < node->num_children; ++i)
        calc_multipoles_at_nodes_core(params, node->child[i]);

    const TYPE_COMPLEX* M_node = node->M;
    const size_t nm = params->num_multipoles;
    if (is_leaf(node))
    {
        #pragma omp task depend(out: M_node[:nm])
        p2m(params, node);
    }
    else
    {
        const TYPE_COMPLEX* M_child[8];
        for (int i = 0; i < 8; ++i) M_child[i] = node->child[i]->M;
        #pragma omp task depend(out: M_node[:nm]) depend(in: M_child[0][:nm], M_child[1][:nm],  \
            M_child[2][:nm], M_child[3][:nm], M_child[4][:nm], M_child[5][:nm], M_child[6][:nm] \
            , M_child[7][:nm])
        m2m(params, node);
    }
}

void calc_multipoles_at_nodes(t_fmm_params* params)
{
    calc_multipoles_at_nodes_core(params, params->root);
}

void calc_local_expansions_core(t_fmm_params* params, t_node* child, t_node* parent)
{
    const TYPE_COMPLEX* L_parent = parent->L;
    const TYPE_COMPLEX* L_child = child->L;
    const size_t nm = params->num_multipoles;
    #pragma omp task depend(in: L_parent[:nm]) depend(out: L_child[:nm])
    l2l(params, child, parent);
    if (is_leaf(child)) 
    {
        const TYPE* ax = child->ax;
        const TYPE* ay = child->ay;
        const TYPE* az = child->az;
        const TYPE* p = child->p;
        const size_t np = child->num_points;
        #pragma omp task depend(in: L_child[:nm]) depend(out: ax[:np], ay[:np], az[:np], p[:np])
        l2p(params, child);
    }

    for (size_t i = 0; i < child->num_children; ++i)
    {
        calc_local_expansions_core(params, child->child[i], child);
    }
}

void calc_local_expansions(t_fmm_params* params)
{
    if (is_leaf(params->root)) return l2p(params, params->root);
    {
        for (size_t i = 0; i < params->root->num_children; ++i)
        {
            calc_local_expansions_core(params, params->root->child[i], params->root);
        }
    }
}

void perform_fmm(t_fmm_params* params)
{
    t_timer timer; 
    start(&timer);

    #pragma omp parallel
    #pragma omp single
    {
        calc_multipoles_at_nodes(params);
        dual_tree_traversal(params);
        calc_local_expansions(params);
    }
    
    stop(&timer);
    printf("Performed all tree passes in %fs\n", timer.elapsed);
}

