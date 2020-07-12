#include <stdlib.h>
#include <stdio.h>

#include "params.h"
#include "type.h"
#include "spharm_gpu.h"
#include "cuda_utils.h"

#define S_IDX(n,m) ((n)*(n)+(n)+(m))

static inline __device__
TYPE neg_pow_n(int n)
{
    return (TYPE)(1 + ((n & 0x01)*-2));
}

static inline __device__
void cmul(TYPE a, TYPE b, TYPE c, TYPE d, TYPE* r, TYPE* i)
{
    *r = (a*c - b*d);
    *i = (a*d + b*c);
}

__device__ TYPE* d_inner_factors_real;
__device__ TYPE* d_inner_factors_imag;
__device__ TYPE* d_outer_factors_real;
__device__ TYPE* d_outer_factors_imag;

void init_array(TYPE* arr, TYPE* d_arr, size_t sz)
{
    TYPE* temp;
    CUDACHK(cudaMalloc((void**)&temp, sz));
    CUDACHK(cudaMemcpy(temp, arr, sz, cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpyToSymbol(d_arr, &temp, sizeof(TYPE*), 0, cudaMemcpyHostToDevice));
}

void init_spharm_gpu(t_fmm_params* params)
{
    const size_t sz = params->num_terms*params->num_terms*sizeof(TYPE); 
    TYPE* inner_factors_real;
    TYPE* inner_factors_imag;
    TYPE* outer_factors_real;
    TYPE* outer_factors_imag;

    CUDACHK(cudaMalloc((void**) &inner_factors_real, sz));
    CUDACHK(cudaMalloc((void**) &inner_factors_imag, sz));
    CUDACHK(cudaMalloc((void**) &outer_factors_real, sz));
    CUDACHK(cudaMalloc((void**) &outer_factors_imag, sz));

    CUDACHK(cudaMemcpy(inner_factors_real, params->inner_factors_real, sz, cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(inner_factors_imag, params->inner_factors_imag, sz, cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(outer_factors_real, params->outer_factors_real, sz, cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(outer_factors_imag, params->outer_factors_imag, sz, cudaMemcpyHostToDevice));

    CUDACHK(cudaMemcpyToSymbol(d_inner_factors_real, &inner_factors_real, sizeof(TYPE*)));
    CUDACHK(cudaMemcpyToSymbol(d_inner_factors_imag, &inner_factors_imag, sizeof(TYPE*)));
    CUDACHK(cudaMemcpyToSymbol(d_outer_factors_real, &outer_factors_real, sizeof(TYPE*)));
    CUDACHK(cudaMemcpyToSymbol(d_outer_factors_imag, &outer_factors_imag, sizeof(TYPE*)));
}

__device__
void compute_outer_gpu(int num_terms, TYPE r, TYPE theta, TYPE phi, TYPE* outer_real, TYPE* outer_imag)
{
    compute_spharm_gpu(num_terms, TYPE_ONE/r, TYPE_ONE/r, theta, phi, d_outer_factors_real, d_outer_factors_imag, outer_real, outer_imag);
}

__device__
void compute_spharm_gpu(int num_terms, TYPE base_r, TYPE mult_r, TYPE theta, TYPE phi, TYPE* factors_real, TYPE* factors_imag, TYPE* Y_rn_real, TYPE* Y_rn_imag)
{
    const size_t nmax = num_terms-1;
    const TYPE x = TYPE_COS(theta);
    //used to change the Condon-Shortley phase factor - csphase=-1.0 includes phase factor
    const TYPE csphase = 1.0;
    
    const TYPE u = csphase*TYPE_SIN(theta);
    size_t n, m;
    TYPE pnm, pmm;
    TYPE twomm1;
    TYPE pm1 = x, pm2 = TYPE_ONE;    
    TYPE power_r = base_r;

    //Y_rn[S_IDX(0,0)] = factors[S_IDX(0,0)]*pm2*power_r;
    Y_rn_real[S_IDX(0,0)] = factors_real[S_IDX(0,0)]*pm2*power_r;
    Y_rn_imag[S_IDX(0,0)] = factors_imag[S_IDX(0,0)]*pm2*power_r;
    power_r *= mult_r;
    //Y_rn[S_IDX(1,0)] = factors[S_IDX(1,0)]*pm1*power_r;
    Y_rn_real[S_IDX(1,0)] = factors_real[S_IDX(1,0)]*pm1*power_r;
    Y_rn_imag[S_IDX(1,0)] = factors_imag[S_IDX(1,0)]*pm1*power_r;
    power_r *= mult_r;
    for (n = 2; n <= nmax; ++n)
    {
        pnm = ((2*n - 1) * x * pm1 - (n - 1) * pm2) / (TYPE) n;
        //Y_rn[S_IDX(n,0)] = factors[n*n+n]*pnm*power_r;
        Y_rn_real[S_IDX(n,0)] = factors_real[n*n+n]*pnm*power_r;
        Y_rn_imag[S_IDX(n,0)] = factors_imag[n*n+n]*pnm*power_r;
        pm2 = pm1;
        pm1 = pnm;
        power_r *= mult_r;
    }

    pmm = TYPE_ONE;
    twomm1 = -TYPE_ONE;
    power_r = base_r*mult_r;
    //TYPE_COMPLEX e_ip = TYPE_CEXP(I*phi);
    //TYPE e_ip_real = TYPE_COS(phi);
    //TYPE e_ip_imag = TYPE_SIN(phi);
    //TYPE_COMPLEX e_imp = e_ip;
    //TYPE e_imp_real = e_ip_real;
    //TYPE e_imp_imag = e_ip_imag;
    for (m = 1; m <= nmax - 1; ++m)
    {
        TYPE e_imp_real = TYPE_COS(m*phi);
        TYPE e_imp_imag = TYPE_SIN(m*phi);
        twomm1 += TYPE_TWO;
        pmm *= u * twomm1;
        pm2 = pmm;

        //Y_rn[S_IDX(m,m)] = factors[S_IDX(m,m)]*pmm*e_imp*power_r;
        cmul(factors_real[S_IDX(m,m)], factors_imag[S_IDX(m,m)], 
            e_imp_real, e_imp_imag, &Y_rn_real[S_IDX(m,m)], &Y_rn_imag[S_IDX(m,m)]);
        Y_rn_real[S_IDX(m,m)] *= pmm*power_r;
        Y_rn_imag[S_IDX(m,m)] *= pmm*power_r;

        //Y_rn[S_IDX(m,-m)] = neg_pow_n(m)*TYPE_CONJ(Y_rn[S_IDX(m,m)]);
        Y_rn_real[S_IDX(m,-m)] = neg_pow_n(m)*Y_rn_real[S_IDX(m,m)];
        Y_rn_imag[S_IDX(m,-m)] = neg_pow_n(m)*-Y_rn_imag[S_IDX(m,m)];
        power_r *= mult_r;

        pm1 = x * pmm * (2*m + 1);
        //Y_rn[S_IDX(m+1,m)] = factors[S_IDX(m+1,m)]*pm1*e_imp*power_r;
        cmul(factors_real[S_IDX(m+1,m)], factors_imag[S_IDX(m+1,m)], 
            e_imp_real, e_imp_imag, &Y_rn_real[S_IDX(m+1,m)], &Y_rn_imag[S_IDX(m+1,m)]);
        Y_rn_real[S_IDX(m+1,m)] *= pm1*power_r;
        Y_rn_imag[S_IDX(m+1,m)] *= pm1*power_r;
        
        //Y_rn[S_IDX(m+1,-m)] = neg_pow_n(m)*TYPE_CONJ(Y_rn[S_IDX(m+1,m)]);
        Y_rn_real[S_IDX(m+1,-m)] = neg_pow_n(m)*Y_rn_real[S_IDX(m+1,m)];
        Y_rn_imag[S_IDX(m+1,-m)] = neg_pow_n(m)*-Y_rn_imag[S_IDX(m+1,m)];

        TYPE power_r2 = power_r*mult_r;
        for (n = m + 2; n <= nmax; ++n)
        {
            pnm = ((2*n - 1) * x * pm1 - (n + m - 1) * pm2) / (TYPE) (n - m);
            pm2 = pm1;
            pm1 = pnm;
            //Y_rn[S_IDX(n,m)] = factors[S_IDX(n,m)]*pnm*e_imp*power_r2;
            cmul(factors_real[S_IDX(n,m)], factors_imag[S_IDX(n,m)], 
                    e_imp_real, e_imp_imag, &Y_rn_real[S_IDX(n,m)], &Y_rn_imag[S_IDX(n,m)]);
            Y_rn_real[S_IDX(n,m)] *= pnm*power_r2;
            Y_rn_imag[S_IDX(n,m)] *= pnm*power_r2;
            
            //Y_rn[S_IDX(n,-m)] = neg_pow_n(m)*TYPE_CONJ(Y_rn[S_IDX(n,m)]);
            Y_rn_real[S_IDX(n,-m)] = neg_pow_n(m)*Y_rn_real[S_IDX(n,m)];
            Y_rn_imag[S_IDX(n,-m)] = neg_pow_n(m)*-Y_rn_imag[S_IDX(n,m)];
            power_r2 *= mult_r;
        }
        //e_imp *= e_ip;
        //e_imp_real = e_imp_real*e_ip_real - e_imp_imag*e_ip_imag;
        //e_imp_imag = e_imp_real*e_ip_imag + e_imp_imag*e_ip_real;
    }
    twomm1 += TYPE_TWO;
    pmm *= u * twomm1;
    //Y_rn[S_IDX(m,m)] = factors[S_IDX(m,m)]*pmm*TYPE_CEXP(I*m*phi)*power_r;
    TYPE exp_real = TYPE_COS(m*phi);
    TYPE exp_imag = TYPE_SIN(m*phi);
    cmul(factors_real[S_IDX(m,m)], factors_imag[S_IDX(m,m)], exp_real, exp_imag, 
        &Y_rn_real[S_IDX(m,m)], &Y_rn_imag[S_IDX(m,m)]);
    Y_rn_real[S_IDX(m,m)] *= pmm*power_r;
    Y_rn_imag[S_IDX(m,m)] *= pmm*power_r;

    //Y_rn[S_IDX(m,-m)] = neg_pow_n(m)*TYPE_CONJ(Y_rn[S_IDX(m,m)]);
    Y_rn_real[S_IDX(m,-m)] = neg_pow_n(m)*Y_rn_real[S_IDX(m,m)];
    Y_rn_imag[S_IDX(m,-m)] = neg_pow_n(m)*-Y_rn_imag[S_IDX(m,m)];
}


//void compute_spharm_deriv(t_fmm_params* params, TYPE base_r, TYPE mult_r, TYPE theta, TYPE phi, TYPE* factors_real, TYPE* factors_imag, TYPE* Y_rn_real, TYPE*Y_rn_imag, TYPE* Y_rn_deriv_real, TYPE* Y_rn_deriv_imag)
//{
//    const size_t nmax = params->num_terms-1;
//    const TYPE x = TYPE_COS(theta);
//
//    //used to change the Condon-Shortley phase factor - csphase=-1.0 includes phase factor
//    const TYPE csphase = 1.0;
//    const TYPE u = TYPE_SIN(theta);
//    const TYPE u_phase = csphase*u;
//    
//    const TYPE u_inv = 1.0/u;
//    const TYPE xbyu = x * u_inv;
//    size_t n, m;
//    TYPE pnm, pmm;
//    TYPE twomm1;
//    TYPE pm1 = x, pm2 = TYPE_ONE;    
//    TYPE power_r = base_r;
//
//    //Y_rn[S_IDX(0,0)] = factors[S_IDX(0,0)]*pm2*power_r;
//    Y_rn_real[S_IDX(0,0)] = factors_real[S_IDX(0,0)]*pm2*power_r;
//    Y_rn_imag[S_IDX(0,0)] = factors_imag[S_IDX(0,0)]*pm2*power_r;
//    //Y_rn_deriv[S_IDX(0,0)] = factors[S_IDX(0,0)]*TYPE_ZERO;
//    Y_rn_deriv_real[S_IDX(0,0)] = factors_real[S_IDX(0,0)]*TYPE_ZERO;
//    Y_rn_deriv_imag[S_IDX(0,0)] = factors_imag[S_IDX(0,0)]*TYPE_ZERO;
//    power_r *= mult_r;
//
//    //Y_rn[S_IDX(1,0)] = factors[S_IDX(1,0)]*pm1*power_r;
//    Y_rn_real[S_IDX(1,0)] = factors_real[S_IDX(1,0)]*pm1*power_r;
//    Y_rn_imag[S_IDX(1,0)] = factors_imag[S_IDX(1,0)]*pm1*power_r;
//    //Y_rn_deriv[S_IDX(1,0)] = factors[S_IDX(1,0)]*-u*power_r;
//    Y_rn_deriv_real[S_IDX(1,0)] = factors_real[S_IDX(1,0)]*-u*power_r;
//    Y_rn_deriv_imag[S_IDX(1,0)] = factors_imag[S_IDX(1,0)]*-u*power_r;
//    power_r *= mult_r;
//
//    for (n = 2; n <= nmax; ++n)
//    {
//        pnm = ((2*n - 1) * x * pm1 - (n - 1) * pm2) / (TYPE) n;
//        //TYPE_COMPLEX coeff = factors[S_IDX(n,0)]*power_r;
//        TYPE coeff_real = factors_real[S_IDX(n,0)]*power_r;
//        TYPE coeff_imag = factors_imag[S_IDX(n,0)]*power_r;
//        //Y_rn[S_IDX(n,0)] = coeff*pnm;
//        Y_rn_real[S_IDX(n,0)] = coeff_real*pnm;
//        Y_rn_imag[S_IDX(n,0)] = coeff_imag*pnm;
//        TYPE dpnm = -(TYPE)n * (pm1 - x * pnm) * u_inv;
//        //Y_rn_deriv[S_IDX(n,0)] = coeff*dpnm;
//        Y_rn_deriv_real[S_IDX(n,0)] = coeff_real*dpnm;
//        Y_rn_deriv_imag[S_IDX(n,0)] = coeff_imag*dpnm;
//        pm2 = pm1;
//        pm1 = pnm;
//        power_r *= mult_r;
//    }
//
//    pmm = TYPE_ONE;
//    twomm1 = -TYPE_ONE;
//    power_r = base_r*mult_r;
//    //TYPE_COMPLEX e_ip = TYPE_CEXP(I*phi);
//    //TYPE_COMPLEX e_imp = e_ip;
//    for (m = 1; m <= nmax - 1; ++m)
//    {
//        TYPE e_imp_real = TYPE_COS(m*phi);
//        TYPE e_imp_imag = TYPE_SIN(m*phi);
//
//        twomm1 += TYPE_TWO;
//        pmm *= u_phase * twomm1;
//        pm2 = pmm;
//
//        //TYPE_COMPLEX coeff = factors[S_IDX(m,m)]*e_imp*power_r;
//        TYPE coeff_real; TYPE coeff_imag;
//        cmul(factors_real[S_IDX(m,m)], factors_imag[S_IDX(m,m)],
//            e_imp_real, e_imp_imag, &coeff_real, &coeff_imag);
//        coeff_real *= power_r; coeff_imag *= power_r;
//        //Y_rn[S_IDX(m,m)] = pmm*coeff;
//        Y_rn_real[S_IDX(m,m)] = pmm*coeff_real;
//        Y_rn_imag[S_IDX(m,m)] = pmm*coeff_imag;
//        //Y_rn[S_IDX(m,-m)] = neg_pow_n(m)*TYPE_CONJ(Y_rn[S_IDX(m,m)]);
//        Y_rn_real[S_IDX(m,-m)] = neg_pow_n(m)*Y_rn_real[S_IDX(m,m)];
//        Y_rn_imag[S_IDX(m,-m)] = neg_pow_n(m)*-Y_rn_imag[S_IDX(m,m)];
//        TYPE dpnm = m * xbyu * pmm;
//        //Y_rn_deriv[S_IDX(m,m)] = coeff*dpnm;
//        Y_rn_deriv_real[S_IDX(m,m)] = coeff_real*dpnm;
//        Y_rn_deriv_imag[S_IDX(m,m)] = coeff_imag*dpnm;
//        power_r *= mult_r;
//
//        pm1 = x * pmm * (2*m + 1);
//        //coeff = factors[S_IDX(m+1,m)]*e_imp*power_r;
//        cmul(factors_real[S_IDX(m+1,m)], factors_imag[S_IDX(m+1,m)],
//            e_imp_real, e_imp_imag, &coeff_real, &coeff_imag);
//        coeff_real *= power_r; coeff_imag *= power_r;
//
//        //Y_rn[S_IDX(m+1,m)] = coeff*pm1;
//        Y_rn_real[S_IDX(m+1,m)] = coeff_real*pm1;
//        Y_rn_imag[S_IDX(m+1,m)] = coeff_imag*pm1;
//        //Y_rn[S_IDX(m+1,-m)] = neg_pow_n(m)*TYPE_CONJ(Y_rn[S_IDX(m+1,m)]);
//        Y_rn_real[S_IDX(m+1,-m)] = neg_pow_n(m)*Y_rn_real[S_IDX(m+1,m)];
//        Y_rn_imag[S_IDX(m+1,-m)] = neg_pow_n(m)*-Y_rn_imag[S_IDX(m+1,m)];
//        dpnm = -u_inv * ((2*m + 1) * pmm - (m+1) * x * pm1);
//        //Y_rn_deriv[S_IDX(m+1,m)] = coeff*dpnm;
//        Y_rn_deriv_real[S_IDX(m+1,m)] = coeff_real*dpnm;
//        Y_rn_deriv_imag[S_IDX(m+1,m)] = coeff_imag*dpnm;
//        
//        TYPE power_r2 = power_r*mult_r;
//        for (n = m + 2; n <= nmax; ++n)
//        {
//            pnm = ((2*n - 1) * x * pm1 - (n + m - 1) * pm2) / (TYPE) (n - m);
//            //coeff = factors[S_IDX(n,m)]*e_imp*power_r2;
//            cmul(factors_real[S_IDX(n,m)], factors_imag[S_IDX(n,m)],
//                    e_imp_real, e_imp_imag, &coeff_real, &coeff_imag);
//            coeff_real *= power_r2; coeff_imag *= power_r2;
//
//            //Y_rn[S_IDX(n,m)] = coeff*pnm;
//            Y_rn_real[S_IDX(n,m)] = coeff_real*pnm;
//            Y_rn_imag[S_IDX(n,m)] = coeff_imag*pnm;
//            //Y_rn[S_IDX(n,-m)] = neg_pow_n(m)*TYPE_CONJ(Y_rn[S_IDX(n,m)]);
//            Y_rn_real[S_IDX(n,-m)] = neg_pow_n(m)*Y_rn_real[S_IDX(n,m)];
//            Y_rn_imag[S_IDX(n,-m)] = neg_pow_n(m)*-Y_rn_imag[S_IDX(n,m)];
//            dpnm = -u_inv * ((n + m) * pm1 - n * x * pnm);
//            //Y_rn_deriv[S_IDX(n,m)] = coeff*dpnm;
//            Y_rn_deriv_real[S_IDX(n,m)] = coeff_real*dpnm;
//            Y_rn_deriv_imag[S_IDX(n,m)] = coeff_imag*dpnm;
//            pm2 = pm1;
//            pm1 = pnm;
//            power_r2 *= mult_r;
//        }
//        //e_imp *= e_ip;
//    }
//    twomm1 += TYPE_TWO;
//    pmm *= u_phase * twomm1;
//    //TYPE_COMPLEX coeff = factors[S_IDX(m,m)]*TYPE_CEXP(I*m*phi)*power_r;
//    TYPE coeff_real; TYPE coeff_imag;
//    cmul(factors_real[S_IDX(m,m)], factors_imag[S_IDX(m,m)],
//            TYPE_COS(m*phi), TYPE_SIN(m*phi), &coeff_real, &coeff_imag);
//    coeff_real *= power_r; coeff_imag *= power_r;
//    //Y_rn[S_IDX(m,m)] = coeff*pmm;
//    Y_rn_real[S_IDX(m,m)] = coeff_real*pmm;
//    Y_rn_imag[S_IDX(m,m)] = coeff_imag*pmm;
//    //Y_rn[S_IDX(m,-m)] = neg_pow_n(m)*TYPE_CONJ(Y_rn[S_IDX(m,m)]);
//    Y_rn_real[S_IDX(m,-m)] = neg_pow_n(m)*Y_rn_real[S_IDX(m,m)];
//    Y_rn_imag[S_IDX(m,-m)] = neg_pow_n(m)*-Y_rn_imag[S_IDX(m,m)];
//    TYPE dpnm = nmax * x * pmm * u_inv;
//    //Y_rn_deriv[S_IDX(m,m)] = coeff*dpnm;
//    Y_rn_deriv_real[S_IDX(m,m)] = coeff_real*dpnm;
//    Y_rn_deriv_imag[S_IDX(m,m)] = coeff_imag*dpnm;
//}

//void compute_inner(t_fmm_params* params, TYPE r, TYPE theta, TYPE phi, TYPE* inner_real, TYPE* inner_imag)
//{
//    compute_spharm(params, TYPE_ONE, r, theta, phi, params->inner_factors_real, params->inner_factors_imag, inner_real, inner_imag);
//}
//
//void compute_outer(t_fmm_params* params, TYPE r, TYPE theta, TYPE phi, TYPE* outer_real, TYPE* outer_imag)
//{
//    compute_spharm(params, TYPE_ONE/r, TYPE_ONE/r, theta, phi, params->outer_factors_real, params->outer_factors_imag, outer_real, outer_imag);
//}
//
//void compute_inner_deriv(t_fmm_params* params, TYPE r, TYPE theta, TYPE phi, TYPE* inner_real, TYPE* inner_imag, TYPE* inner_deriv_real, TYPE* inner_deriv_imag)
//{
//    //compute_spharm(params, TYPE_ONE, r, theta, phi, params->inner_factors_real, params->inner_factors_imag, inner_real, inner_imag);
//    compute_spharm_deriv(params, TYPE_ONE, r, theta, phi, params->inner_factors_real, params->inner_factors_imag, inner_real, inner_imag, inner_deriv_real, inner_deriv_imag);
//}
//
//void compute_outer_deriv(t_fmm_params* params, TYPE r, TYPE theta, TYPE phi, TYPE* outer_real, TYPE* outer_imag, TYPE* outer_deriv_real, TYPE* outer_deriv_imag)
//{
//    // TODO should this be inner or outer?
//    compute_spharm_deriv(params, TYPE_ONE/r, TYPE_ONE/r, theta, phi, params->outer_factors_real, params->outer_factors_imag, outer_real, outer_imag, outer_deriv_real, outer_deriv_imag);
//}
//
