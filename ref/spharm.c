#include <stdlib.h>
#include <stdio.h>

#include "params.h"
#include "type.h"
#include "spharm.h"


#define S_IDX(n,m) ((n)*(n)+(n)+(m))

// #include <gsl/gsl_sf_legendre.h>

// void spharm_r_n(t_fmm_options* options, TYPE base_r, TYPE mult_r, TYPE theta, TYPE phi, TYPE_COMPLEX* Y_rn)
// {
//     int num_terms = options->num_terms;
//     size_t array_size = gsl_sf_legendre_array_n(num_terms-1);
//     TYPE p_temp[array_size];
//     gsl_sf_legendre_array_e(GSL_SF_LEGENDRE_NONE, num_terms-1, cos(theta), -1.0, p_temp);

//     TYPE power_r = base_r;
//     for (int n = 0; n < num_terms; ++n)
//     {
//         int m = 0;
//         Y_rn[n*n+n+m] = options->spharm_factor[n*n+n+m]*p_temp[n*(n+1)/2+m]*TYPE_CEXP(I*m*phi)*power_r;
//         for (m = 1; m <= n; ++m)
//         {
//             Y_rn[n*n+n+m] = options->spharm_factor[n*n+n+m]*p_temp[n*(n+1)/2+m]*TYPE_CEXP(I*m*phi)*power_r;
//             Y_rn[n*n+n-m] = conj(Y_rn[n*n+n+m]);
//         }
//         power_r *= mult_r;
//     }
// }

// void spharm_r_n_d_theta(t_fmm_options* options, TYPE base_r, TYPE mult_r, TYPE theta, TYPE phi, TYPE_COMPLEX* Y_rn, TYPE_COMPLEX* Y_rn_div_theta)
// {
//     int num_terms = options->num_terms;
//     size_t array_size = gsl_sf_legendre_array_n(num_terms-1);
//     TYPE p_temp[array_size];
//     TYPE p_temp_div[array_size];

//     gsl_sf_legendre_deriv_alt_array_e(GSL_SF_LEGENDRE_NONE, num_terms-1, cos(theta), -1.0, p_temp, p_temp_div);

//     TYPE power_r = base_r;
//     for (int n = 0; n < num_terms; ++n)
//     {
//         int m = 0;
//         Y_rn[n*n+n+m] = options->spharm_factor[n*n+n+m]*p_temp[n*(n+1)/2+m]*TYPE_CEXP(I*m*phi)*power_r;
//         Y_rn_div_theta[n*n+n+m] = options->spharm_factor[n*n+n+m]*p_temp_div[n*(n+1)/2+m]*TYPE_CEXP(I*m*phi)*power_r;

//         for (m = 1; m <= n; ++m)
//         {
//             Y_rn[n*n+n+m] = options->spharm_factor[n*n+n+m]*p_temp[n*(n+1)/2+m]*TYPE_CEXP(I*m*phi)*power_r;
//             Y_rn[n*n+n-m] = conj(Y_rn[n*n+n+m]);

//             Y_rn_div_theta[n*n+n+m] = options->spharm_factor[n*n+n+m]*p_temp_div[n*(n+1)/2+m]*TYPE_CEXP(I*m*phi)*power_r;
//         }
//         power_r *= mult_r;
//     }
// }

void spharm_r_n(t_fmm_options* options, TYPE base_r, TYPE mult_r, TYPE theta, TYPE phi, TYPE_COMPLEX* Y_rn)
{
    const size_t nmax = options->num_terms-1;
    const TYPE x = TYPE_COS(theta);
    const TYPE neg_u = -TYPE_SIN(theta);
    size_t n, m;
    TYPE pnm, pmm;
    TYPE twomm1;
    TYPE pm1 = x, pm2 = TYPE_ONE;    
    TYPE power_r = base_r;

    Y_rn[S_IDX(0,0)] = pm2*power_r;
    power_r *= mult_r;
    Y_rn[S_IDX(1,0)] = pm1*power_r;
    power_r *= mult_r;
    for (n = 2; n <= nmax; ++n)
    {
        pnm = ((2*n - 1) * x * pm1 - (n - 1) * pm2) / (TYPE) n;
        Y_rn[S_IDX(n,0)] = options->spharm_factor[n*n+n]*pnm*power_r;
        pm2 = pm1;
        pm1 = pnm;
        power_r *= mult_r;
    }

    pmm = TYPE_ONE;
    twomm1 = -TYPE_ONE;
    power_r = base_r*mult_r;
    TYPE_COMPLEX e_ip = TYPE_CEXP(I*phi);
    TYPE_COMPLEX e_imp = e_ip;
    for (m = 1; m <= nmax - 1; ++m)
    {
        twomm1 += TYPE_TWO;
        pmm *= neg_u * twomm1;
        pm2 = pmm;

        Y_rn[S_IDX(m,m)] = options->spharm_factor[S_IDX(m,m)]*pmm*e_imp*power_r;
        Y_rn[S_IDX(m,-m)] = TYPE_CONJ(Y_rn[S_IDX(m,m)]);
        power_r *= mult_r;

        pm1 = x * pmm * (2*m + 1);
        Y_rn[S_IDX(m+1,m)] = options->spharm_factor[S_IDX(m+1,m)]*pm1*e_imp*power_r;
        Y_rn[S_IDX(m+1,-m)] = TYPE_CONJ(Y_rn[S_IDX(m+1,m)]);

        TYPE power_r2 = power_r*mult_r;
        for (n = m + 2; n <= nmax; ++n)
        {
            pnm = ((2*n - 1) * x * pm1 - (n + m - 1) * pm2) / (TYPE) (n - m);
            pm2 = pm1;
            pm1 = pnm;
            Y_rn[S_IDX(n,m)] = options->spharm_factor[S_IDX(n,m)]*pnm*e_imp*power_r2;
            Y_rn[S_IDX(n,-m)] = TYPE_CONJ(Y_rn[S_IDX(n,m)]);
            power_r2 *= mult_r;
        }
        e_imp *= e_ip;
    }
    twomm1 += TYPE_TWO;
    pmm *= neg_u * twomm1;
    Y_rn[S_IDX(m,m)] = options->spharm_factor[S_IDX(m,m)]*pmm*TYPE_CEXP(I*m*phi)*power_r;
    Y_rn[S_IDX(m,-m)] = TYPE_CONJ(Y_rn[S_IDX(m,m)]);
}

void spharm_r_n_d_theta(t_fmm_options* options, TYPE base_r, TYPE mult_r, TYPE theta, TYPE phi, TYPE_COMPLEX* Y_rn, TYPE_COMPLEX* Y_rn_deriv)
{
    const size_t nmax = options->num_terms-1;
    const TYPE x = TYPE_COS(theta);
    const TYPE neg_u = -TYPE_SIN(theta);
    const TYPE neg_u_inv = 1.0/neg_u;
    const TYPE xbyu = x * -neg_u_inv;
    size_t n, m;
    TYPE pnm, pmm;
    TYPE twomm1;
    TYPE pm1 = x, pm2 = TYPE_ONE;    
    TYPE power_r = base_r;

    Y_rn[S_IDX(0,0)] = pm2*power_r;
    Y_rn_deriv[S_IDX(0,0)] = 0.0;
    power_r *= mult_r;

    Y_rn[S_IDX(1,0)] = pm1*power_r;
    Y_rn_deriv[S_IDX(1,0)] = neg_u*power_r;
    power_r *= mult_r;

    for (n = 2; n <= nmax; ++n)
    {
        pnm = ((2*n - 1) * x * pm1 - (n - 1) * pm2) / (TYPE) n;
        TYPE_COMPLEX coeff = options->spharm_factor[S_IDX(n,0)]*power_r;
        Y_rn[S_IDX(n,0)] = coeff*pnm;
        TYPE dpnm = (TYPE)n * (pm1 - x * pnm) * neg_u_inv;
        Y_rn_deriv[S_IDX(n,0)] = coeff*dpnm;
        pm2 = pm1;
        pm1 = pnm;
        power_r *= mult_r;
    }

    pmm = TYPE_ONE;
    twomm1 = -TYPE_ONE;
    power_r = base_r*mult_r;
    TYPE_COMPLEX e_ip = TYPE_CEXP(I*phi);
    TYPE_COMPLEX e_imp = e_ip;
    for (m = 1; m <= nmax - 1; ++m)
    {
        twomm1 += TYPE_TWO;
        pmm *= neg_u * twomm1;
        pm2 = pmm;

        TYPE_COMPLEX coeff = options->spharm_factor[S_IDX(m,m)]*e_imp*power_r;
        Y_rn[S_IDX(m,m)] = pmm*coeff;
        Y_rn[S_IDX(m,-m)] = TYPE_CONJ(Y_rn[S_IDX(m,m)]);
        TYPE dpnm = m * xbyu * pmm;
        Y_rn_deriv[S_IDX(m,m)] = coeff*dpnm;
        power_r *= mult_r;

        pm1 = x * pmm * (2*m + 1);
        coeff = options->spharm_factor[S_IDX(m+1,m)]*e_imp*power_r;
        Y_rn[S_IDX(m+1,m)] = coeff*pm1;
        Y_rn[S_IDX(m+1,-m)] = TYPE_CONJ(Y_rn[S_IDX(m+1,m)]);
        dpnm = neg_u_inv * ((2*m + 1) * pmm - (m+1) * x * pm1);
        Y_rn_deriv[S_IDX(m+1,m)] = coeff*dpnm;
        
        TYPE power_r2 = power_r*mult_r;
        for (n = m + 2; n <= nmax; ++n)
        {
            pnm = ((2*n - 1) * x * pm1 - (n + m - 1) * pm2) / (TYPE) (n - m);
            coeff = options->spharm_factor[S_IDX(n,m)]*e_imp*power_r2;
            Y_rn[S_IDX(n,m)] = coeff*pnm;
            Y_rn[S_IDX(n,-m)] = TYPE_CONJ(Y_rn[S_IDX(n,m)]);
            dpnm = neg_u_inv * ((n + m) * pm1 - n * x * pnm);
            Y_rn_deriv[S_IDX(n,m)] = coeff*dpnm;
            pm2 = pm1;
            pm1 = pnm;
            power_r2 *= mult_r;
        }
        e_imp *= e_ip;
    }
    twomm1 += TYPE_TWO;
    pmm *= neg_u * twomm1;
    TYPE_COMPLEX coeff = options->spharm_factor[S_IDX(m,m)]*TYPE_CEXP(I*m*phi)*power_r;
    Y_rn[S_IDX(m,m)] = coeff*pmm;
    Y_rn[S_IDX(m,-m)] = TYPE_CONJ(Y_rn[S_IDX(m,m)]);
    TYPE dpnm = nmax * x * pmm * -neg_u_inv;
    Y_rn_deriv[S_IDX(m,m)] = coeff*dpnm;
}