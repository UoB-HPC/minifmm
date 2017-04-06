#include <stdlib.h>

#include "type.h"
#include "params.h"
#include "verify.h"

TYPE get_l2_error(TYPE* estimate, TYPE* actual, size_t num_points, size_t lda_exact, size_t lda_approx, int dim)
{
    TYPE diff = TYPE_ZERO, norm = TYPE_ZERO;
    for (size_t i = 0; i < num_points; ++i)
    {       
        TYPE tmp_diff = TYPE_ZERO;
        for (int d = 0; d < dim; ++d) tmp_diff += (estimate[d*lda_approx+i] - actual[d*lda_exact+i]) *
            (estimate[d*lda_approx+i] - actual[d*lda_exact+i]);
        diff += TYPE_ABS(tmp_diff);

        TYPE tmp_norm = TYPE_ZERO;
        for (int d = 0; d < dim; ++d) tmp_norm += actual[d*lda_exact+i] * actual[d*lda_exact+i];
        norm += TYPE_ABS(tmp_norm); 
    }
    return TYPE_SQRT(diff/norm);
}

void direct_method(TYPE* points, TYPE* m, size_t num_samples, size_t num_points, TYPE* a, TYPE* pot)
{
    TYPE* x = &points[0*num_points];
    TYPE* y = &points[1*num_points];
    TYPE* z = &points[2*num_points];
    TYPE* ax = &a[0*num_samples];
    TYPE* ay = &a[1*num_samples];
    TYPE* az = &a[2*num_samples];

    for (size_t i = 0; i < num_samples; ++i)
    {
        TYPE axt = 0.0, ayt = 0.0, azt = 0.0, pott = 0.0;
        TYPE xi = x[i], yi = y[i], zi = z[i];
        for (size_t j = 0; j < num_points; ++j)
        {
            TYPE dx = x[j] - xi;
            TYPE dy = y[j] - yi;
            TYPE dz = z[j] - zi;
            TYPE r = sqrt(dx*dx + dy*dy + dz*dz);
            TYPE inv_r = (r == 0.0) ? 0.0 : 1.0/r;
            TYPE inv_r_cubed = inv_r*inv_r*inv_r;
            axt += m[j]*dx*inv_r_cubed;
            ayt += m[j]*dy*inv_r_cubed;
            azt += m[j]*dz*inv_r_cubed;
            pott += m[j]*inv_r;
        }
        ax[i] += axt;
        ay[i] += ayt;
        az[i] += azt;
        pot[i] += pott;
    }
}

void verify(t_fmm_options* options, TYPE* a_err, TYPE* p_err)
{
    TYPE* acc_exact = (TYPE*)malloc(sizeof(TYPE)*options->num_samples*3);
    TYPE* pot_exact = (TYPE*)malloc(sizeof(TYPE)*options->num_samples);

    for (size_t i = 0; i < options->num_samples*3; ++i) acc_exact[i] = 0.0;
    for (size_t i = 0; i < options->num_samples; ++i) pot_exact[i] = 0.0;

    direct_method(options->points_ordered, options->weights_ordered, options->num_samples, options->num_points, acc_exact, pot_exact);

    *a_err = get_l2_error(options->acc, acc_exact, options->num_samples, options->num_samples, options->num_points, 3);
    *p_err = get_l2_error(options->pot, pot_exact, options->num_samples, 0, 0, 1);
}
