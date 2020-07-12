#include <stdlib.h>
#include <omp.h>

#include "type.h"
#include "tree.h"
#include "kernels.h"
#include "timer.h"
#include "dtt.h"

void dual_tree_traversal_core(t_fmm_params* params, t_node* target, t_node* source)
{
    TYPE dx = source->center[0] - target->center[0];
    TYPE dy = source->center[1] - target->center[1];
    TYPE dz = source->center[2] - target->center[2];
    TYPE r2 = dx*dx + dy*dy + dz*dz;
    TYPE d1 = source->rad*2.0;
    TYPE d2 = target->rad*2.0;

    if ((d1+d2)*(d1+d2) < params->theta2*r2)
    {
        m2l(params, target, source);
    }
    else if (is_leaf(source) && is_leaf(target))
    {
        p2p(params, target, source);
    }
    else
    {
        TYPE target_sz = target->rad;
        TYPE source_sz = source->rad;

        if (is_leaf(source) || (target_sz >= source_sz && !is_leaf(target))) 
        {
            for (size_t i = 0; i < target->num_children; ++i)
                #pragma omp task
                dual_tree_traversal_core(params, target->child[i], source);
        }
        else
        {
             for (size_t i = 0; i < source->num_children; ++i)
                dual_tree_traversal_core(params, target, source->child[i]);
        }
    }
    #pragma omp taskwait
}

void dual_tree_traversal(t_fmm_params* params)
{   
    #pragma omp parallel
    {
        #pragma omp single
        {
            dual_tree_traversal_core(params, params->root, params->root);
        }
    }
}

