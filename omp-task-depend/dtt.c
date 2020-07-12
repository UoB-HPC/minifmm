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
        const TYPE_COMPLEX* L_ptr = target->L;
        const TYPE_COMPLEX* M_ptr = target->M;
        const size_t nm = params->num_multipoles;
        #pragma omp task depend(in: M_ptr[:nm]) depend(out: L_ptr[:nm])
        m2l(params, target, source);
    }
    else if (is_leaf(source) && is_leaf(target))
    {
        const TYPE* ax = target->ax; 
        const TYPE* ay = target->ay; 
        const TYPE* az = target->az; 
        const TYPE* p = target->p; 
        const size_t np = target->num_points;
        #pragma omp task depend(out: ax[:np], ay[:np], az[:np], p[:np])
        p2p(params, target, source);
    }
    else
    {
        TYPE target_sz = target->rad;
        TYPE source_sz = source->rad;

        if (is_leaf(source) || (target_sz >= source_sz && !is_leaf(target))) 
        {
            for (size_t i = 0; i < target->num_children; ++i)
                dual_tree_traversal_core(params, target->child[i], source);
        }
        else
        {
             for (size_t i = 0; i < source->num_children; ++i)
                dual_tree_traversal_core(params, target, source->child[i]);
        }
    }
}

void dual_tree_traversal(t_fmm_params* params)
{   
    dual_tree_traversal_core(params, params->root, params->root);
}

