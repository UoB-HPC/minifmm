#include <stdlib.h>
#include <omp.h>

#include "type.h"
#include "tree.h"
#include "kernels.h"
#include "traversal.h"
#include "timer.h"

void calc_multipoles_at_nodes_core(t_fmm_options* options, t_node* node)
{
    if (node == NULL) return;

    for (size_t i = 0; i < node->num_children; ++i)
        calc_multipoles_at_nodes_core(options, node->child[i]);
    
    if (is_leaf(node))
    {
#ifdef _OPENMP
    const TYPE_COMPLEX* M __attribute__((unused)) = node->M;
    const size_t num_multipoles __attribute__((unused)) = options->num_multipoles;
    #pragma omp task depend(out: M[0:num_multipoles])
#endif
    p2m(options, node);
    }
    else
    {
#ifdef _OPENMP
    TYPE_COMPLEX* M[8] = {0};
    const TYPE_COMPLEX* M_parent __attribute__((unused)) = node->M;
    for (size_t i = 0; i < node->num_children; ++i) M[i] = node->child[i]->M;
    #pragma omp task depend(in: M[0][0:options->num_multipoles], M[1][0:options->num_multipoles], M[2][0:options->num_multipoles], M[3][0:options->num_multipoles], \
    M[4][0:options->num_multipoles], M[5][0:options->num_multipoles], M[6][0:options->num_multipoles], M[7][0:options->num_multipoles]) \
    depend(out: M_parent[0:options->num_multipoles])
#endif
    m2m(options, node);
    }
}

void calc_multipoles_at_nodes(t_fmm_options* options)
{
    calc_multipoles_at_nodes_core(options, options->root);
}

void calc_local_expansions_core(t_fmm_options* options, t_node* current, t_node* parent)
{
#ifdef _OPENMP
    const TYPE_COMPLEX* L_parent __attribute__((unused)) = parent->L;
    const TYPE_COMPLEX* L_current __attribute__((unused)) = current->L;
    const size_t num_multipoles __attribute__((unused)) = options->num_multipoles;
    #pragma omp task depend(in: L_parent[0:num_multipoles]) depend(out: L_current[0:num_multipoles])
#endif
    l2l(options, current, parent);
    if (is_leaf(current)) 
    {
#ifdef _OPENMP
        const size_t end = current->num_points;
        const TYPE* ax __attribute__((unused)) = current->ax;
        const TYPE* ay __attribute__((unused)) = current->ay;
        const TYPE* az __attribute__((unused)) = current->az;
        const TYPE* p __attribute__((unused)) = current->p;
        #pragma omp task depend(in: L_current[0:num_multipoles]) depend(out: ax[0:end], ay[0:end], az[0:end], p[0:end])
#endif
        l2p(options, current);
    }

    for (size_t i = 0; i < current->num_children; ++i)
    {
        calc_local_expansions_core(options, current->child[i], current);
    }
}

void calc_local_expansions(t_fmm_options* options)
{
    if (is_leaf(options->root)) return l2p(options, options->root);
    for (size_t i = 0; i < options->root->num_children; ++i)
    {
        calc_local_expansions_core(options, options->root->child[i], options->root);
    }
}

extern size_t* thread_p2p_interactions;
extern double* thread_p2p_times;

void dual_tree_traversal_core(t_fmm_options* options, t_node* target, t_node* source)
{
    TYPE dx = source->center[0] - target->center[0];
    TYPE dy = source->center[1] - target->center[1];
    TYPE dz = source->center[2] - target->center[2];
    TYPE r = TYPE_SQRT(dx*dx + dy*dy + dz*dz);
    TYPE d1 = source->rad*2.0;
    TYPE d2 = target->rad*2.0;

    if ((d1+d2)/r < options->theta)
    {
#ifdef _OPENMP
        const TYPE_COMPLEX* M __attribute__((unused)) = source->M;
        const TYPE_COMPLEX* L __attribute__((unused)) = target->L;
        const size_t num_multipoles __attribute__((unused)) = options->num_multipoles;
        #pragma omp task depend(in: M[0:num_multipoles]) depend(out: L[0:num_multipoles])
#endif
        m2l(options, target, source);
    }
    else if (is_leaf(source) && is_leaf(target))
    {
#ifdef _OPENMP
        const TYPE* ax __attribute__((unused)) = target->ax;
        const TYPE* ay __attribute__((unused)) = target->ay;
        const TYPE* az __attribute__((unused)) = target->az;
        const TYPE* p __attribute__((unused)) = target->p;
        const size_t end = target->num_points;
#endif
        if (source == target)
        {
            #pragma omp task depend(out: ax[0:end], ay[0:end], az[0:end], p[0:end])
            {
                int thread_num = omp_get_thread_num();
                t_timer timer;
                start(&timer);
                p2p_one_node(options, target); 
                stop(&timer);
                thread_p2p_times[thread_num] += timer.elapsed;
                thread_p2p_interactions[thread_num] += target->num_points;
            }
        }
        else
        {
            #pragma omp task depend(out: ax[0:end], ay[0:end], az[0:end], p[0:end])
            {
                int thread_num = omp_get_thread_num();
                t_timer timer;
                start(&timer);
                p2p(options, target, source); 
                stop(&timer);
                thread_p2p_times[thread_num] += timer.elapsed;
                thread_p2p_interactions[thread_num] += target->num_points*source->num_points;
            }
        }
    }
    else
    {
        TYPE target_sz = target->rad;
        TYPE source_sz = source->rad;

        if (is_leaf(source))
        {
            for (size_t i = 0; i < target->num_children; ++i)
                dual_tree_traversal_core(options, target->child[i], source);
        }
        else if (is_leaf(target))
        {
             for (size_t i = 0; i < source->num_children; ++i)
                dual_tree_traversal_core(options, target, source->child[i]);
        }
        else if (target_sz >= source_sz)
        {
            for (size_t i = 0; i < target->num_children; ++i)
                dual_tree_traversal_core(options, target->child[i], source);
        }
        else
        {
            for (size_t i = 0; i < source->num_children; ++i)
                dual_tree_traversal_core(options, target, source->child[i]);
        }
    }
}

void dual_tree_traversal(t_fmm_options* options)
{   
    dual_tree_traversal_core(options, options->root, options->root);
}

void perform_fmm(t_fmm_options* options)
{
    calc_multipoles_at_nodes(options);
    dual_tree_traversal(options);
    calc_local_expansions(options);
}
