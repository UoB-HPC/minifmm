#include <stdlib.h>
#include <cilk/cilk.h>

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
        cilk_spawn(
            calc_multipoles_at_nodes_core(params, node->child[i])
        );

    cilk_sync;

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
    calc_multipoles_at_nodes_core(params, params->root);
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
        cilk_spawn(
            calc_local_expansions_core(params, current->child[i], current)
        );
    }
    cilk_sync;
}

void calc_local_expansions(t_fmm_params* params)
{
    if (is_leaf(params->root)) return l2p(params, params->root);
    {
        for (size_t i = 0; i < params->root->num_children; ++i)
        {
            cilk_spawn(
                calc_local_expansions_core(params, params->root->child[i], params->root)
            );
        }
        cilk_sync;
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
}
