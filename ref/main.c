#include <stdio.h>
#include <stdlib.h>

#include "params.h"
#include "traversal.h"
#include "type.h"
#include "kernels.h"
#include "timer.h"
#include "verify.h"
#include "initialise.h"
#include "finalise.h"
#include "util.h"

int main(int argc, char** argv)
{
    t_fmm_params params;
    initialise(argc, argv, &params);

    printf(SEPERATOR);
    t_timer timer;
    start(&timer);
    perform_fmm(&params);
    stop(&timer);
    printf(SEPERATOR);
    printf("Total elapsed FMM time = %f\n", timer.elapsed);

    TYPE a_err, p_err;
    verify(&params, &a_err, &p_err);
    printf(SEPERATOR);
    printf("force err.     = %e\n", a_err);
    printf("potential err. = %e\n", p_err);

    finalise(&params);
    return 0;
}
