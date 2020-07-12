#include <stdlib.h>

#include "type.h"
#include "tree.h"
#include "kernels.h"
#include "traversal.h"
#include "dtt.h"
#include "timer.h"
#include "util.h"

void calc_multipoles_at_nodes_core(t_fmm_params* params, t_node* node)
{
    if (node == NULL) return;

    for (size_t i = 0; i < node->num_children; ++i)
        calc_multipoles_at_nodes_core(params, get_node(params, node->child[i]));
    
    if (is_leaf(node))
    {
        p2m(params, node);
    }
    else
    {
        m2m(params, node);
    }
}

void calc_multipoles_at_nodes(t_fmm_params* params)
{
    calc_multipoles_at_nodes_core(params, get_node(params, params->root));
}

void calc_local_expansions_core(t_fmm_params* params, t_node* current, t_node* parent)
{
    l2l(params, current, parent);
    if (is_leaf(current)) 
    {
        l2p(params, current);
    }

    for (size_t i = 0; i < current->num_children; ++i)
    {
        calc_local_expansions_core(params, get_node(params, current->child[i]), current);
    }
    
}

void calc_local_expansions(t_fmm_params* params)
{
    t_node* root = get_node(params, params->root);
    if (is_leaf(root)) return l2p(params, root);
    {
        for (size_t i = 0; i < root->num_children; ++i)
        {
            calc_local_expansions_core(params, get_node(params, root->child[i]), root);
        }
    }
}

void perform_fmm(t_fmm_params* params)
{
    t_timer timer; 
    
    start(&timer);
    
    calc_multipoles_at_nodes(params);
    stop(&timer);
    printf("Performed upward tree pass in %fs\n", timer.elapsed);

    start(&timer);
    dual_tree_traversal(params);
    stop(&timer);
    printf("Performed dual tree traversal in %fs\n", timer.elapsed);

    start(&timer);
    
    calc_local_expansions(params);
    stop(&timer);
    
    printf("Performed downward tree pass in %fs\n", timer.elapsed);

    size_t np = params->num_points;
    for (size_t i = 0; i < np; ++i)
    {
        params->acc[0*np+i] = params->ax[i];
        params->acc[1*np+i] = params->ay[i];
        params->acc[2*np+i] = params->az[i];
        params->pot[i] = params->p[i];    
    }
}

