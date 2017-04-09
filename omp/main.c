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
            printf("running computation on %d threads\n", omp_get_num_threads());
            perform_fmm(&options);
        }
    }
    stop(&timer);
    timer_print(&timer, "fmm");

    TYPE a_err, p_err;
    verify(&options, &a_err, &p_err);
    printf("%.15f %.15f\n", a_err, p_err);
    printf("m2l calls = %d\n", m2l_calls);
}
