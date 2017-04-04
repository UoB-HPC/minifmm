#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#include "params.h"
#include "traversal.h"
#include "type.h"
#include "kernels.h"
#include "timer.h"
#include "verify.h"
#include "initialise.h"

// void print_tree(t_node* root, t_node* rroot)
// {
//     printf("node %zu has %zu points\n", root-rroot, root->num_points);
//     for (size_t i = 0; i < root->num_children; ++i) print_tree(root->child[i], rroot);
// }

// void direct_tree(t_fmm_options* options)
// {
//     printf("num nodes = %zu\n", options->num_nodes);
//     for (size_t i = 0; i < options->num_nodes; ++i)
//     {
//         for (size_t j = 0; j < options->num_nodes; ++j)
//         {
//             if (is_leaf(&options->root[i]) && is_leaf(&options->root[j])) p2p(options, &options->root[i], &options->root[j]);
//         }
//     }
// }



extern int m2l_calls;
int main(int argc, char** argv)
{
    t_fmm_options options;
    initialise(argc, argv, &options);

    t_timer timer;
    start(&timer);
    perform_fmm(&options);
    // direct_tree(&options);
    stop(&timer);
    timer_print(&timer, "fmm");

    // check(&options);
    TYPE a_err, p_err;
    verify(&options, &a_err, &p_err);
    printf("%f %f\n", a_err, p_err);
    printf("m2l calls = %d\n", m2l_calls);
    // printf("num wrong = %d\n", num_wrong);
    // print_tree(options.root, options.root);


    // for (size_t i = 0; i < options.num_points; ++i)
    // {
    //     for (int d = 0; d < 3; ++d) 
    //         printf("%f ", options.points_ordered[d*options.num_points+i]);
    //     printf("\n");
    // }

}