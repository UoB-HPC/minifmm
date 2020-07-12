#include <stdio.h>
#include <stdlib.h>

#include <Kokkos_Core.hpp>

#include "params.h"
#include "traversal.h"
#include "type.h"
#include "kernels.h"
#include "timer.h"
#include "verify.h"
#include "initialise.h"
#include "finalise.h"
#include "util.h"
extern int m2l_calls;
int main(int argc, char** argv)
{
    Kokkos::initialize();

    t_fmm_params params;
    initialise(argc, argv, &params);
    
    printf("starting computation\n");
    printf(SEPERATOR);
    t_timer timer;
    start(&timer);
    perform_fmm(&params);
    stop(&timer);
    printf(SEPERATOR);
    printf("Total elapsed FMM time = %f\n", timer.elapsed);

    //tree_to_result(&params);
    printf("m2l calls = %d\n", m2l_calls);

    TYPE a_err, p_err;
    verify(&params, &a_err, &p_err);
    printf(SEPERATOR);
    printf("force err.     = %e\n", a_err);
    printf("potential err. = %e\n", p_err);

    finalise(&params);
    Kokkos::finalize();
    return 0;
}
