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
t_node* __node_head;
t_node* __node_curr;
t_node* __node_tail;

t_node* allocate_node(t_fmm_options* options)
{
    // t_node* node = (t_node*)malloc(sizeof(t_node));

    if (__node_curr == __node_tail) 
    {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    t_node* node = __node_curr++;

    node->M = (TYPE_COMPLEX*)malloc(sizeof(TYPE_COMPLEX)*options->num_multipoles);
    node->L = (TYPE_COMPLEX*)malloc(sizeof(TYPE_COMPLEX)*options->num_multipoles);
    for (size_t i = 0; i < options->num_multipoles; ++i)
    {
        node->M[i] = node->L[i] = 0;
    }
    ++__num_node_allocations;
    return node;
}

void init_node_storage(t_fmm_options* options)
{
    size_t approx_num_leaves = ((options->num_points/options->ncrit)*4);
    size_t max_num_nodes = approx_num_leaves * (size_t)log2((double)approx_num_leaves);
    __node_head = (t_node*)malloc(sizeof(t_node)*max_num_nodes);
    __node_tail = __node_head + max_num_nodes - 1;
    __node_curr = __node_head;
}

#include "immintrin.h"

void free_node(t_node* node)
{
    free(node->M);
    free(node->L);
    if (is_leaf(node))
    {
        _mm_free(node->x);
    }
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

        size_t point_sz = 64/sizeof(TYPE);
        size_t pts_in_leaf = end - start;
        size_t padding = point_sz - (pts_in_leaf % (point_sz));
        TYPE* node_mem = (TYPE*)_mm_malloc(sizeof(TYPE)*8*(pts_in_leaf+padding), 64);
        parent->x  = &node_mem[0*(pts_in_leaf+padding)];
        parent->y  = &node_mem[1*(pts_in_leaf+padding)];
        parent->z  = &node_mem[2*(pts_in_leaf+padding)];
        parent->w  = &node_mem[3*(pts_in_leaf+padding)];
        parent->ax = &node_mem[4*(pts_in_leaf+padding)];
        parent->ay = &node_mem[5*(pts_in_leaf+padding)];
        parent->az = &node_mem[6*(pts_in_leaf+padding)];
        parent->p  = &node_mem[7*(pts_in_leaf+padding)];

        TYPE* points_ptr = (!points_switched) ? points_ordered : points;
        TYPE* weights_ptr = (!points_switched) ? weights_ordered : weights;

        for (size_t i = 0; i < pts_in_leaf; ++i)
        {
            parent->x[i] = points_ptr[0*num_points+i+start];
            parent->y[i] = points_ptr[1*num_points+i+start];
            parent->z[i] = points_ptr[2*num_points+i+start];
            parent->w[i] = weights_ptr[i+start];
            parent->ax[i] = parent->ay[i] = parent->az[i] = parent->p[i] = TYPE_ZERO;
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
    init_node_storage(options);

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
    for (size_t i = 0; i < node->num_children; ++i) free_tree_core(node->child[i]);
    free_node(node);
}

void free_tree(t_fmm_options* options)
{
    free_tree_core(options->root);
    free(__node_head);
}

void tree_to_result(t_fmm_options* options)
{
    size_t counter = 0;
    for (size_t i = 0; i < options->num_nodes; ++i)
    {
        t_node* node = &options->root[i];
        if (is_leaf(node))
        {
            for (size_t j = 0; j < node->num_points; ++j)
            {
                options->acc[0*options->num_points+counter] = node->ax[j];
                options->acc[1*options->num_points+counter] = node->ay[j];
                options->acc[2*options->num_points+counter] = node->az[j];
                options->pot[counter] = node->p[j];
                ++counter;
            }
        }
    }
}