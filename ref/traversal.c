#include <stdlib.h>

#include "type.h"
#include "tree.h"
#include "kernels.h"
#include "traversal.h"

void calc_multipoles_at_nodes_core(t_fmm_options* options, t_node* node)
{
    if (node == NULL) return;

    for (size_t i = 0; i < node->num_children; ++i)
        calc_multipoles_at_nodes_core(options, node->child[i]);
    
    if (is_leaf(node))
    {
        p2m(options, node);
    }
    else
    {
        m2m(options, node);
    }
}

void calc_multipoles_at_nodes(t_fmm_options* options)
{
    calc_multipoles_at_nodes_core(options, options->root);
}

void calc_local_expansions_core(t_fmm_options* options, t_node* current, t_node* parent)
{
    l2l(options, current, parent);
    if (is_leaf(current)) 
    {
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

size_t num_particle_interactions = 0;
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
        m2l(options, target, source);
    }
    else if (is_leaf(source) && is_leaf(target))
    {
        if (target == source) 
        {
            p2p_one_node(options, target);
            num_particle_interactions += target->num_points;
        }
        else 
        {
            p2p(options, target, source);
            num_particle_interactions += target->num_points*source->num_points;
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
