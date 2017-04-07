#include <stdio.h>
#include <stdlib.h>

#include "type.h"
#include "util.h"
#include "tree.h"

void bound_box(TYPE* mins, TYPE* maxs, TYPE* points, size_t num_points)
{
    for (int i = 0; i < 3; ++i) mins[i] = TYPE_MAX;
    for (int i = 0; i < 3; ++i) maxs[i] = -TYPE_MAX;

    for (size_t i = 0; i < num_points; ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            mins[d] = MIN(mins[d], points[d*num_points+i]);
            maxs[d] = MAX(maxs[d], points[d*num_points+i]);
        }
    }
}

static size_t __num_node_allocations = 0;

t_node* allocate_node(t_fmm_options* options)
{
    t_node* node = (t_node*)malloc(sizeof(t_node));
    node->M = (TYPE_COMPLEX*)malloc(sizeof(TYPE_COMPLEX)*options->num_multipoles);
    node->L = (TYPE_COMPLEX*)malloc(sizeof(TYPE_COMPLEX)*options->num_multipoles);
    ++__num_node_allocations;
    return node;
}

void free_node(t_node* node)
{
    free(node->M);
    free(node->L);
    free(node);
    --__num_node_allocations;
}

size_t get_num_nodes() { return __num_node_allocations; }

void construct_tree(t_node* parent, size_t start, size_t end, size_t ncrit, TYPE* points, TYPE* weights,
    TYPE* points_ordered, TYPE* weights_ordered, TYPE* acc, TYPE* pot,
    int points_switched, size_t num_points, int level, t_fmm_options* options)
{
    parent->num_points = end - start;
    parent->num_children = 0;
    parent->x = (!points_switched) ? &points_ordered[0*num_points+start] : &points[0*num_points+start];
    parent->y = (!points_switched) ? &points_ordered[1*num_points+start] : &points[1*num_points+start];
    parent->z = (!points_switched) ? &points_ordered[2*num_points+start] : &points[2*num_points+start];
    parent->w = (!points_switched) ? &weights_ordered[start] : &weights[start];
    parent->ax = &acc[0*num_points+start];
    parent->ay = &acc[1*num_points+start];
    parent->az = &acc[2*num_points+start];
    parent->p = &pot[start];

    if (end - start <= ncrit)
    {
        if (!points_switched)
        {
            for (size_t i = start; i < end; ++i)
            {
                for (int d = 0; d < 3; ++d) points_ordered[d*num_points+i] = points[d*num_points+i];
                weights_ordered[i] = weights[i];
            }
        }
        return;
    }

    size_t num_points_per_oct[8] = {0};
    for (size_t i = start; i < end; ++i)
    {
        // courtesy of exaFMM
        int oct = (points[0*num_points+i] > parent->center[0]) + 
            ((points[1*num_points+i] > parent->center[1]) << 1) + 
            ((points[2*num_points+i] > parent->center[2]) << 2);
        num_points_per_oct[oct]++;           
    }

    size_t oct_pointers[8] = {start};
    size_t oct_pointers_copy[8] = {start};
    for (int i = 1; i < 8; ++i) oct_pointers[i] = oct_pointers[i-1] + num_points_per_oct[i-1];
    for (int i = 0; i < 8; ++i) oct_pointers_copy[i] = oct_pointers[i];

    for (size_t j = start; j < end; ++j)
    {
        int oct = (points[0*num_points+j] > parent->center[0]) + 
            ((points[1*num_points+j] > parent->center[1]) << 1) + 
            ((points[2*num_points+j] > parent->center[2]) << 2);

        size_t i = oct_pointers_copy[oct];
        for (int d = 0; d < 3; ++d) 
        {
            points_ordered[d*num_points+i] = points[d*num_points+j];
        }
        weights_ordered[i] = weights[j];
        oct_pointers_copy[oct]++;
    }

    TYPE new_r = parent->rad/(TYPE_TWO);
    for (int i = 0; i < 8; ++i)
    {
        if (num_points_per_oct[i])
        {
            t_node* child = allocate_node(options);
            for (int d = 0; d < 3; ++d) child->center[d] = ((i >> d) & 1) ? (parent->center[d] + new_r) : (parent->center[d] - new_r);
            child->rad = new_r;
            parent->child[parent->num_children++] = child;
            //dfs tree
            construct_tree(child, oct_pointers[i], oct_pointers[i]+num_points_per_oct[i], ncrit, points_ordered, weights_ordered, points, weights, acc, pot, !points_switched, num_points, level+1, options);
        }
    }
}

void build_tree(t_fmm_options* options)
{
    TYPE mins[3], maxs[3];
    bound_box(mins, maxs, options->points, options->num_points);

    printf("bound box --- \n");
    for (int d = 0; d < 3; ++d) printf("%f %f\n", mins[d], maxs[d]);
    
    t_node* root = allocate_node(options);
    TYPE max_rad = TYPE_ZERO;
    for (int d = 0; d < 3; ++d) 
    {
        root->center[d] = (maxs[d] + mins[d])/(TYPE_TWO);
        TYPE rad = (maxs[d] - mins[d])/(TYPE_TWO);
        max_rad = MAX(rad, max_rad);
    }
    // need to add EPS for points that lie on border of node
    root->rad = max_rad + TYPE_EPS;

    construct_tree(root, 0, options->num_points, options->ncrit, options->points, options->weights, 
        options->points_ordered, options->weights_ordered, options->acc, options->pot, 0, options->num_points, 0, options);

    options->num_nodes = get_num_nodes();
    printf("Tree has %zu nodes\n", options->num_nodes);

    options->root = root;
}

void free_tree_core(t_node* node)
{
    for (size_t i = 0; i < node->num_children; ++i) free_node(node->child[i]);
    free_node(node);
}

void free_tree(t_fmm_options* options)
{
    free_tree_core(options->root);
}