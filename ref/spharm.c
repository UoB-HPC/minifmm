#include <stdlib.h>
#include <stdio.h>

#include "params.h"
#include "type.h"
#include "spharm.h"

#define S_IDX(n,m) ((n)*(n)+(n)+(m))

static inline
TYPE neg_pow_n(int n)
{
    return (TYPE)(1 + ((n & 0x01)*-2));
}

void compute_spharm(t_fmm_params* params, TYPE base_r, TYPE mult_r, TYPE theta, TYPE phi, TYPE_COMPLEX* factors, TYPE_COMPLEX* Y_rn)
{
    const size_t nmax = params->num_terms-1;
    const TYPE x = TYPE_COS(theta);
    //used to change the Condon-Shortley phase factor - csphase=-1.0 includes phase factor
    const TYPE csphase = 1.0;
    
    const TYPE u = csphase*TYPE_SIN(theta);
    size_t n, m;
    TYPE pnm, pmm;
    TYPE twomm1;
    TYPE pm1 = x, pm2 = TYPE_ONE;    
    TYPE power_r = base_r;

    Y_rn[S_IDX(0,0)] = factors[S_IDX(0,0)]*pm2*power_r;
    power_r *= mult_r;
    Y_rn[S_IDX(1,0)] = factors[S_IDX(1,0)]*pm1*power_r;
    power_r *= mult_r;
    for (n = 2; n <= nmax; ++n)
    {
        pnm = ((2*n - 1) * x * pm1 - (n - 1) * pm2) / (TYPE) n;
        Y_rn[S_IDX(n,0)] = factors[n*n+n]*pnm*power_r;
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
        pmm *= u * twomm1;
        pm2 = pmm;

        Y_rn[S_IDX(m,m)] = factors[S_IDX(m,m)]*pmm*e_imp*power_r;
        Y_rn[S_IDX(m,-m)] = neg_pow_n(m)*TYPE_CONJ(Y_rn[S_IDX(m,m)]);
        power_r *= mult_r;

        pm1 = x * pmm * (2*m + 1);
        Y_rn[S_IDX(m+1,m)] = factors[S_IDX(m+1,m)]*pm1*e_imp*power_r;
        Y_rn[S_IDX(m+1,-m)] = neg_pow_n(m)*TYPE_CONJ(Y_rn[S_IDX(m+1,m)]);

        TYPE power_r2 = power_r*mult_r;
        for (n = m + 2; n <= nmax; ++n)
        {
            pnm = ((2*n - 1) * x * pm1 - (n + m - 1) * pm2) / (TYPE) (n - m);
            pm2 = pm1;
            pm1 = pnm;
            Y_rn[S_IDX(n,m)] = factors[S_IDX(n,m)]*pnm*e_imp*power_r2;
            Y_rn[S_IDX(n,-m)] = neg_pow_n(m)*TYPE_CONJ(Y_rn[S_IDX(n,m)]);
            power_r2 *= mult_r;
        }
        e_imp *= e_ip;
    }
    twomm1 += TYPE_TWO;
    pmm *= u * twomm1;
    Y_rn[S_IDX(m,m)] = factors[S_IDX(m,m)]*pmm*TYPE_CEXP(I*m*phi)*power_r;
    Y_rn[S_IDX(m,-m)] = neg_pow_n(m)*TYPE_CONJ(Y_rn[S_IDX(m,m)]);
}


void compute_spharm_deriv(t_fmm_params* params, TYPE base_r, TYPE mult_r, TYPE theta, TYPE phi, TYPE_COMPLEX* factors, TYPE_COMPLEX* Y_rn, TYPE_COMPLEX* Y_rn_deriv)
{
    const size_t nmax = params->num_terms-1;
    const TYPE x = TYPE_COS(theta);

    //used to change the Condon-Shortley phase factor - csphase=-1.0 includes phase factor
    const TYPE csphase = 1.0;
    const TYPE u = TYPE_SIN(theta);
    const TYPE u_phase = csphase*u;
    
    const TYPE u_inv = 1.0/u;
    const TYPE xbyu = x * u_inv;
    size_t n, m;
    TYPE pnm, pmm;
    TYPE twomm1;
    TYPE pm1 = x, pm2 = TYPE_ONE;    
    TYPE power_r = base_r;

    Y_rn[S_IDX(0,0)] = factors[S_IDX(0,0)]*pm2*power_r;
    Y_rn_deriv[S_IDX(0,0)] = factors[S_IDX(0,0)]*TYPE_ZERO;
    power_r *= mult_r;

    Y_rn[S_IDX(1,0)] = factors[S_IDX(1,0)]*pm1*power_r;
    Y_rn_deriv[S_IDX(1,0)] = factors[S_IDX(1,0)]*-u*power_r;
    power_r *= mult_r;

    for (n = 2; n <= nmax; ++n)
    {
        pnm = ((2*n - 1) * x * pm1 - (n - 1) * pm2) / (TYPE) n;
        TYPE_COMPLEX coeff = factors[S_IDX(n,0)]*power_r;
        Y_rn[S_IDX(n,0)] = coeff*pnm;
        TYPE dpnm = -(TYPE)n * (pm1 - x * pnm) * u_inv;
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
        pmm *= u_phase * twomm1;
        pm2 = pmm;

        TYPE_COMPLEX coeff = factors[S_IDX(m,m)]*e_imp*power_r;
        Y_rn[S_IDX(m,m)] = pmm*coeff;
        Y_rn[S_IDX(m,-m)] = neg_pow_n(m)*TYPE_CONJ(Y_rn[S_IDX(m,m)]);
        TYPE dpnm = m * xbyu * pmm;
        Y_rn_deriv[S_IDX(m,m)] = coeff*dpnm;
        power_r *= mult_r;

        pm1 = x * pmm * (2*m + 1);
        coeff = factors[S_IDX(m+1,m)]*e_imp*power_r;
        Y_rn[S_IDX(m+1,m)] = coeff*pm1;
        Y_rn[S_IDX(m+1,-m)] = neg_pow_n(m)*TYPE_CONJ(Y_rn[S_IDX(m+1,m)]);
        dpnm = -u_inv * ((2*m + 1) * pmm - (m+1) * x * pm1);
        Y_rn_deriv[S_IDX(m+1,m)] = coeff*dpnm;
        
        TYPE power_r2 = power_r*mult_r;
        for (n = m + 2; n <= nmax; ++n)
        {
            pnm = ((2*n - 1) * x * pm1 - (n + m - 1) * pm2) / (TYPE) (n - m);
            coeff = factors[S_IDX(n,m)]*e_imp*power_r2;
            Y_rn[S_IDX(n,m)] = coeff*pnm;
            Y_rn[S_IDX(n,-m)] = neg_pow_n(m)*TYPE_CONJ(Y_rn[S_IDX(n,m)]);
            dpnm = -u_inv * ((n + m) * pm1 - n * x * pnm);
            Y_rn_deriv[S_IDX(n,m)] = coeff*dpnm;
            pm2 = pm1;
            pm1 = pnm;
            power_r2 *= mult_r;
        }
        e_imp *= e_ip;
    }
    twomm1 += TYPE_TWO;
    pmm *= u_phase * twomm1;
    TYPE_COMPLEX coeff = factors[S_IDX(m,m)]*TYPE_CEXP(I*m*phi)*power_r;
    Y_rn[S_IDX(m,m)] = coeff*pmm;
    Y_rn[S_IDX(m,-m)] = neg_pow_n(m)*TYPE_CONJ(Y_rn[S_IDX(m,m)]);
    TYPE dpnm = nmax * x * pmm * u_inv;
    Y_rn_deriv[S_IDX(m,m)] = coeff*dpnm;
}

void compute_inner(t_fmm_params* params, TYPE r, TYPE theta, TYPE phi, TYPE_COMPLEX* inner)
{
    compute_spharm(params, TYPE_ONE, r, theta, phi, params->inner_factors, inner);
}

void compute_outer(t_fmm_params* params, TYPE r, TYPE theta, TYPE phi, TYPE_COMPLEX* outer)
{
    compute_spharm(params, TYPE_ONE/r, TYPE_ONE/r, theta, phi, params->outer_factors, outer);
}

void compute_inner_deriv(t_fmm_params* params, TYPE r, TYPE theta, TYPE phi, TYPE_COMPLEX* inner, TYPE_COMPLEX* inner_deriv)
{
    compute_spharm_deriv(params, TYPE_ONE, r, theta, phi, params->inner_factors, inner, inner_deriv);
}

void compute_outer_deriv(t_fmm_params* params, TYPE r, TYPE theta, TYPE phi, TYPE_COMPLEX* outer, TYPE_COMPLEX* outer_deriv)
{
    compute_spharm_deriv(params, TYPE_ONE/r, TYPE_ONE/r, theta, phi, params->inner_factors, outer, outer_deriv);
}

