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
#include "util.h"

double* thread_p2p_times;
size_t* thread_p2p_interactions;
int num_threads = 0;

extern int m2l_calls;
int main(int argc, char** argv)
{
    t_fmm_options options;
    initialise(argc, argv, &options);

    t_timer timer;
    start(&timer);

    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            thread_p2p_times = (double*)malloc(sizeof(double)*num_threads);
            for (int i = 0; i  < num_threads; ++i) thread_p2p_times[i] = 0.0;
            thread_p2p_interactions = (size_t*)malloc(sizeof(size_t)*num_threads);
            printf("running computation on %d threads\n", omp_get_num_threads());
            perform_fmm(&options);
        }
    }
    stop(&timer);
    timer_print(&timer, "fmm");

    double gpi_tot = 0.0, gpi_max = 0.0;

    printf("\n\nthread results--------\n");

    for (int i = 0; i < num_threads; ++i)
    {
        printf("time = %f, interactions = %zu\n", thread_p2p_times[i], thread_p2p_interactions[i]);
        double gpi = (thread_p2p_times[i] > 0.0) ? (double)thread_p2p_interactions[i]/1000000000.0/thread_p2p_times[i] :
            0.0;
        gpi_tot += gpi;
        gpi_max = MAX(gpi_max, gpi);
    }

    printf("----------\ntot. GPI/s = %f, avg. GPI/s = %f, max GPI/s = %f\n", gpi_tot, gpi_tot/num_threads, gpi_max);

    TYPE a_err, p_err;
    verify(&options, &a_err, &p_err);
    printf("%.15f %.15f\n", a_err, p_err);
    printf("m2l calls = %d\n", m2l_calls);
}
