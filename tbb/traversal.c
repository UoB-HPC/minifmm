#include <stdlib.h>

#include <tbb/task_group.h>
#include <tbb/task_scheduler_init.h>

#include "type.h"
#include "tree.h"
#include "kernels.h"
#include "traversal.h"
#include "dtt.h"
#include "timer.h"

using namespace tbb;

void calc_multipoles_at_nodes_core(t_fmm_params* params, t_node* node)
{
    if (node == NULL) return;
    
    task_group tg;
    for (size_t i = 0; i < node->num_children; ++i)
        tg.run([&params, i, &node]
        {
            calc_multipoles_at_nodes_core(params, node->child[i]);
        });
    tg.wait();

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

    task_group tg;
    for (size_t i = 0; i < current->num_children; ++i)
    {
        tg.run([&params, i, &current]
        {
            calc_local_expansions_core(params, current->child[i], current);
        });
    }
    tg.wait();
}

void calc_local_expansions(t_fmm_params* params)
{
    if (is_leaf(params->root)) return l2p(params, params->root);
    {
        task_group tg;
        for (size_t i = 0; i < params->root->num_children; ++i)
        {
            tg.run([&params, i]
            {
                calc_local_expansions_core(params, params->root->child[i], params->root);
            });
        }
        tg.wait();
    }
}

void perform_fmm(t_fmm_params* params)
{
    t_timer timer; 

    const char* nts = getenv("OMP_NUM_THREADS");
    int num_threads = (nts == NULL) ? 1 : atoi(nts);
    tbb::task_scheduler_init init(num_threads);
    
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
