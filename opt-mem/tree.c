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

static t_node* __node_head;
static t_node* __node_curr;
static t_node* __node_tail;
static size_t __node_allocations = 0;
static TYPE_COMPLEX* __M_storage;
static TYPE_COMPLEX* __L_storage;

t_node* allocate_node()
{
    if (__node_curr == __node_tail)
    {
        fprintf(stderr, "out of memory for nodes, larger allocation needed - num allocs = %zu\n", __node_allocations);
        exit(1);
    }
    ++__node_allocations;
    return __node_curr++;
    // return (t_node*)malloc(sizeof(t_node));
}

void init_node_storage(size_t num_points, size_t nodes_per_point)
{
    size_t alloc_sz = num_points*nodes_per_point;
    __node_head = (t_node*)malloc(sizeof(t_node)*alloc_sz);
    if (__node_head == NULL)
    {
        fprintf(stderr, "could not allocate enough space for %zu nodes", alloc_sz);
        exit(1);
    }
    __node_curr = __node_head;
    __node_tail = __node_head + alloc_sz - 1;
}

size_t get_num_nodes() { return __node_allocations; }

void allocate_multipole_storage(t_fmm_options* options)
{
    __M_storage = (TYPE_COMPLEX*)malloc(sizeof(TYPE_COMPLEX)*options->num_nodes*options->num_multipoles);
    __L_storage = (TYPE_COMPLEX*)malloc(sizeof(TYPE_COMPLEX)*options->num_nodes*options->num_multipoles);
    for (size_t i = 0; i < options->num_nodes; ++i)
    {
        __node_head[i].M = &__M_storage[i*options->num_multipoles];
        __node_head[i].L = &__L_storage[i*options->num_multipoles];
    }
}

static inline
void swap(TYPE* a, TYPE* b)
{
    TYPE t = *a;
    *a = *b;
    *b = t;
}


static size_t __base_index = 0;
void register_alloc(t_node* node, size_t num_points)
{
    // if (__base_index )
}

static int counter = 0;
void construct_tree(t_node* parent, size_t start, size_t end, size_t ncrit, TYPE* points, TYPE* weights,
    TYPE* points_ordered, TYPE* weights_ordered, TYPE* acc, TYPE* pot,
    int points_switched, size_t num_points, int level)
{
    if (start > end) printf("errorw %zu %zu\n", start, end);
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
            t_node* child = allocate_node();
            for (int d = 0; d < 3; ++d) child->center[d] = ((i >> d) & 1) ? (parent->center[d] + new_r) : (parent->center[d] - new_r);
            child->rad = new_r;
            parent->child[parent->num_children++] = child;
            //dfs tree
            construct_tree(child, oct_pointers[i], oct_pointers[i]+num_points_per_oct[i], ncrit, points_ordered, weights_ordered, points, weights, acc, pot, !points_switched, num_points, level+1);
        }
    }
}

void build_tree(t_fmm_options* options)
{
    init_node_storage(options->num_points, 8);
    TYPE mins[3], maxs[3];
    bound_box(mins, maxs, options->points, options->num_points);

    printf("bound box --- \n");
    for (int d = 0; d < 3; ++d) printf("%f %f\n", mins[d], maxs[d]);
    
    t_node* root = allocate_node();
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
        options->points_ordered, options->weights_ordered, options->acc, options->pot, 0, options->num_points, 0);

    options->num_nodes = get_num_nodes();
    printf("Tree has %zu nodes\n", options->num_nodes);
    allocate_multipole_storage(options);

    options->root = root;
}

// void init_data(double** ppoints, double** pmass, size_t num_points)
// {
//     double* points = (double*)malloc(sizeof(double)*num_points*3);
//     double* mass = (double*)malloc(sizeof(double)*num_points);

//     for (size_t i = 0; i < num_points; ++i)
//     {
//         mass[i] = 1.0;
//     }

//     // seed_rng(time(NULL));
//     seed_rng(42);
//     for (size_t i = 0; i < num_points*3; ++i)
//     {
//         points[i] = rand_range(-1.0, 1.0);
//     }
//         // points[i] = change_range(points[i], 0.0, 1.0, -1.0, 1.0);
        
//     *ppoints = points;
//     *pmass = mass;
// }

// static size_t point_count = 0;
// void check_tree(t_node* root)
// {
//     if (root == NULL) return;

//     for (int b = 0; b < root->num_children; ++b) check_tree(root->child[b]);

//     if (root->num_children == 0) 
//     {
//         point_count += root->num_points;
//         for (int i = 0; i < root->num_points; ++i)
//         {
//             double r = root->rad;
//             int oob = 0;
//             if (root->x[i] > root->center[0]+r || root->x[i] < root->center[0]-r) oob = 1;
//             if (root->y[i] > root->center[1]+r || root->y[i] < root->center[1]-r) oob = 1;
//             if (root->z[i] > root->center[2]+r || root->z[i] < root->center[2]-r) oob = 1;
//             if (oob) 
//             {
//                 printf("point %d of node %d is oob\n", i, root - __node_head);
//                 for (int d = 0; d < 3; ++d) printf("%f ", root->center[d]);
//                 printf("\nr = %f\n", r);
//                 printf("point = %f %f %f\n\n\n", root->x[i], root->y[i], root->z[i]);
//             }
//         }
//     }

//     if (root->num_points == 0) printf("error node has no points\n");
    
// }
// #include "timer.h"

// int main(int argc, char** argv)
// {
//     double* points;
//     double* weights;

//     size_t num_points = (argc >= 2) ? atoi(argv[1]) : 100;
//     size_t ncrit = (argc >= 3) ? atoi(argv[2]) : 1;

//     init_node_storage(num_points, 8);
//     init_data(&points, &weights, num_points);

//     double* points_ordered = malloc(sizeof(double)*num_points*3);
//     double* weights_ordered = malloc(sizeof(double)*num_points);

//     t_node* root;
//     t_timer timer;
//     start(&timer);
//     root = build_tree(points, weights, points_ordered, weights_ordered, num_points, ncrit);
//     stop(&timer);
//     timer_print(&timer, "build tree");

//     // for (size_t i = 0; i < num_points; ++i)
//     // {
//     //     for (int d = 0; d < 3; ++d)
//     //     printf("%f ", points_ordered[d*num_points+i]);
//     //     printf("\n");
//     // }

//     check_tree(root);
//     printf("point count = %zu\n", point_count);
// }
