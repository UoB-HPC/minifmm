#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <immintrin.h>
#include <omp.h>

#include "type.h"
#include "util.h"
#include "tree.h"

#ifndef CACHELINE_SZ
    #define CACHLINE_SZ 64
#endif

#define CACHELINE_PADDING_SZ(x) ((CACHLINE_SZ - (x % CACHLINE_SZ)) % CACHLINE_SZ)

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
static size_t __current_max_nodes = 0;
#define NODE_RECORD_GROWTH_FACTOR 1.5

t_node** __node_array;
t_node** __node_arr_curr;

size_t record_node(t_node* node)
{
    if (__num_node_allocations == __current_max_nodes)
    {
        // increase size of node records
        __current_max_nodes = (double)__current_max_nodes * NODE_RECORD_GROWTH_FACTOR;
        //printf("expanding max nodes to %zu\n", __current_max_nodes);
        __node_array = (t_node**)realloc(__node_array, sizeof(t_node*)*__current_max_nodes);
        __node_arr_curr = __node_array + __num_node_allocations - 1;

        if (__node_array == NULL)
        {
            fprintf(stderr, "error: could not allocate enough tree nodes\n"); 
            exit(1);
        }
    }
    *__node_arr_curr = node;
    size_t ret = __node_arr_curr - __node_array;
    __node_arr_curr++;
    return ret;
}

void init_node_array(t_fmm_params* params)
{
    __current_max_nodes = (params->num_points/params->ncrit);
    __node_array = (t_node**)malloc(sizeof(t_node*)*__current_max_nodes);
    __node_arr_curr = __node_array;
    //printf("init allocating %zu node records\n", __current_max_nodes);
}

t_node* allocate_node(t_fmm_params* params, size_t pts_in_leaf, TYPE* points, TYPE* weights)
{
    size_t node_padding = CACHELINE_PADDING_SZ(sizeof(t_node));
    size_t node_alloc_sz = sizeof(t_node) + node_padding;

    size_t point_padding = CACHELINE_PADDING_SZ(sizeof(TYPE)*pts_in_leaf);
    // 8 points for x, y, z, w, ax, ay, az, p
    size_t point_alloc_sz = 8*(sizeof(TYPE)*pts_in_leaf + point_padding);
    
    size_t cmplx_sz = sizeof(TYPE_COMPLEX);
    size_t multipole_padding = CACHELINE_PADDING_SZ(cmplx_sz * params->num_multipoles);
    // *2 because of local and multipole expansions
    size_t multipole_alloc_sz = 2*(cmplx_sz*params->num_multipoles + multipole_padding);

    size_t alloc_sz = node_alloc_sz + multipole_alloc_sz + point_alloc_sz;

    //TODO change to own alloc function
    char* mem = (char*)_mm_malloc(alloc_sz, 64);
    memset(mem, 0, alloc_sz);

    t_node* node = (t_node*)mem;

    TYPE_COMPLEX* mult_mem = (TYPE_COMPLEX*)&mem[node_alloc_sz];

    node->M = mult_mem;
    node->L = (TYPE_COMPLEX*)((char*)mult_mem + params->num_multipoles*cmplx_sz + multipole_padding);

    if (pts_in_leaf > 0)
    {
        TYPE* point_mem = (TYPE*)&mem[node_alloc_sz + multipole_alloc_sz];
        node->x  = point_mem;
        node->y  = (TYPE*)((char*)point_mem + 1*(pts_in_leaf*sizeof(TYPE) + point_padding));
        node->z  = (TYPE*)((char*)point_mem + 2*(pts_in_leaf*sizeof(TYPE) + point_padding));
        node->w  = (TYPE*)((char*)point_mem + 3*(pts_in_leaf*sizeof(TYPE) + point_padding));
        node->ax = (TYPE*)((char*)point_mem + 4*(pts_in_leaf*sizeof(TYPE) + point_padding));
        node->ay = (TYPE*)((char*)point_mem + 5*(pts_in_leaf*sizeof(TYPE) + point_padding));
        node->az = (TYPE*)((char*)point_mem + 6*(pts_in_leaf*sizeof(TYPE) + point_padding));
        node->p  = (TYPE*)((char*)point_mem + 7*(pts_in_leaf*sizeof(TYPE) + point_padding));

        for (size_t i = 0; i < pts_in_leaf; ++i)
        {
            node->x[i] = points[0*params->num_points+i];
            node->y[i] = points[1*params->num_points+i];
            node->z[i] = points[2*params->num_points+i];
            node->w[i] = weights[i];
        }
    }

    omp_init_lock(&node->p2p_lock);
    omp_init_lock(&node->m2l_lock);

    ++__num_node_allocations;
    node->arr_idx = record_node(node);
    return node;
}

void free_node(t_node* node)
{
    //TODO change to own alloc function
    _mm_free(node);
    --__num_node_allocations;
}

size_t get_num_nodes() { return __num_node_allocations; }

t_node* construct_tree(TYPE cx, TYPE cy, TYPE cz, TYPE r, size_t start, size_t end, size_t ncrit, TYPE* points, TYPE* weights,
    TYPE* points_ordered, TYPE* weights_ordered, TYPE* acc, TYPE* pot,
    int points_switched, size_t num_points, int level, t_fmm_params* params)
{
    TYPE* curr_points = (!points_switched) ? points_ordered : points;
    TYPE* curr_weights = (!points_switched) ? weights_ordered : weights;

    t_node* parent = (end - start <= ncrit) ? 
        allocate_node(params, end - start, curr_points+start, curr_weights+start) :
        allocate_node(params, 0, NULL, NULL);
    parent->center[0] = cx;
    parent->center[1] = cy;
    parent->center[2] = cz;
    parent->rad = r;

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

    // if leaf node
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
    }
    else
    {
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
                TYPE ncx, ncy, ncz;
                ncx = ((i >> 0) & 1) ? (parent->center[0] + new_r) : (parent->center[0] - new_r);
                ncy = ((i >> 1) & 1) ? (parent->center[1] + new_r) : (parent->center[1] - new_r);
                ncz = ((i >> 2) & 1) ? (parent->center[2] + new_r) : (parent->center[2] - new_r);

                t_node* child = construct_tree(ncx, ncy, ncz, new_r, oct_pointers[i], oct_pointers[i]+num_points_per_oct[i], ncrit, points_ordered, weights_ordered, points, weights, acc, pot, !points_switched, num_points, level+1, params);
                parent->child[parent->num_children++] = child;
                //dfs tree
            }
        }
    }
    return parent;
}

void build_tree(t_fmm_params* params)
{
    init_node_array(params);

    TYPE mins[3], maxs[3];
    bound_box(mins, maxs, params->points, params->num_points);

    //printf("bound box --- \n");
    //for (int d = 0; d < 3; ++d) printf("%f %f\n", mins[d], maxs[d]);
    
    TYPE max_rad = TYPE_ZERO;
    TYPE center[3];
    for (int d = 0; d < 3; ++d) 
    {
        center[d] = (maxs[d] + mins[d])/(TYPE_TWO);
        TYPE rad = (maxs[d] - mins[d])/(TYPE_TWO);
        max_rad = MAX(rad, max_rad);
    }
    // need to add EPS for points that lie on border of node
    max_rad += TYPE_EPS;

    params->root = construct_tree(center[0], center[1], center[2], max_rad, 0, params->num_points, params->ncrit, params->points, params->weights, 
        params->points_ordered, params->weights_ordered, params->acc, params->pot, 0, params->num_points, 0, params);

    params->num_nodes = get_num_nodes();
    //printf("Tree has %zu nodes\n", params->num_nodes);
}

void free_tree_core(t_node* node)
{
    for (size_t i = 0; i < node->num_children; ++i) free_tree_core(node->child[i]);
    free_node(node);
}

void free_tree(t_fmm_params* params)
{
    free_tree_core(params->root);
    free(__node_array);
}

void tree_to_result(t_fmm_params* params)
{
    //printf("total %zu records\n", __node_arr_curr - __node_array);

    size_t counter = 0;
    for (size_t i = 0; i < params->num_nodes; ++i)
    {
        t_node* node = __node_array[i];
        if (is_leaf(node))
        {
            for (size_t j = 0; j < node->num_points; ++j)
            {
                params->acc[0*params->num_points+counter] = node->ax[j];
                params->acc[1*params->num_points+counter] = node->ay[j];
                params->acc[2*params->num_points+counter] = node->az[j];
                params->pot[counter] = node->p[j];
                ++counter;
            }
        }
    }
}
