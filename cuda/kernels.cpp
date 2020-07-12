#include <stdlib.h>
#include <stdio.h>

#include "type.h"
#include "params.h"
#include "tree.h"
#include "util.h"
#include "kernels.h"
#include "spharm.h"

static inline
void cfma(TYPE a, TYPE b, TYPE c, TYPE d, TYPE* r, TYPE* i)
{
    *r += (a*c - b*d);
    *i += (a*d + b*c);
}

static inline
TYPE cmul_real(TYPE a, TYPE b, TYPE c, TYPE d)
{
    return (a*c - b*d);
}

static inline
TYPE cmul_imag(TYPE a, TYPE b, TYPE c, TYPE d)
{
    return (a*d + b*c);
}

    static inline
void cart_to_sph(TYPE x, TYPE y, TYPE z, TYPE* pr, TYPE* ptheta, TYPE* pphi)
{
    *pr = TYPE_SQRT(x*x+y*y+z*z);
    *ptheta = (*pr == TYPE_ZERO) ? TYPE_ZERO : TYPE_ACOS(z/(*pr));
    *pphi = TYPE_ATAN2(y, x);
}

    static inline
void sph_unit_to_cart_unit(TYPE r, TYPE theta, TYPE phi, TYPE grad_r, TYPE grad_theta, TYPE grad_phi,
        TYPE* x, TYPE* y, TYPE* z)
{
    *x = TYPE_SIN(theta)*TYPE_COS(phi)*grad_r+TYPE_COS(theta)*TYPE_COS(phi)*grad_theta-TYPE_SIN(phi)*grad_phi;
    *y = TYPE_SIN(theta)*TYPE_SIN(phi)*grad_r+TYPE_COS(theta)*TYPE_SIN(phi)*grad_theta+TYPE_COS(phi)*grad_phi;
    *z = TYPE_COS(theta)*              grad_r-TYPE_SIN(theta)              *grad_theta;
}

#define S_IDX(n,m) ((n)*(n)+(n)+(m))

//__device__
//void p2p_gpu(t_node* target, t_node* source)
//{
//    const TYPE* tx = target->x;
//    const TYPE* ty = target->y;
//    const TYPE* tz = target->z;
//    const TYPE* sx = source->x;
//    const TYPE* sy = source->y;
//    const TYPE* sz = source->z;
//    const TYPE* sw = source->w;
//    TYPE* tax = target->ax;
//    TYPE* tay = target->ay;
//    TYPE* taz = target->az;
//    TYPE* tp = target->p;
//
//    printf("performing p2p\n");
//    
//    for (size_t i = 0; i < target->num_points; ++i)
//    {
//        TYPE ax = 0.0, ay = 0.0, az = 0.0, p = 0.0;
//        const TYPE xi = tx[i], yi = ty[i], zi = tz[i];
//        for (size_t j = 0; j < source->num_points; ++j)
//        {
//            const TYPE dx = sx[j] - xi;
//            const TYPE dy = sy[j] - yi;
//            const TYPE dz = sz[j] - zi;
//            const TYPE wj = sw[j]; 
//            const TYPE r = dx*dx + dy*dy + dz*dz;
//            TYPE inv_r = (r == 0.0) ? 0.0 : rsqrt(r);
//            const TYPE t = wj*inv_r;
//            const TYPE s = t*inv_r*inv_r;
//            ax += dx*s;
//            ay += dy*s;
//            az += dz*s;
//            p += t;
//        }
//        tax[i] += ax;
//        tay[i] += ay;
//        taz[i] += az;
//        tp[i] += p;
//    }
//}

void p2p(t_fmm_params* params, t_node* target, t_node* source)
{
    size_t tidx = target->point_idx;
    size_t sidx = source->point_idx;

    TYPE* tx = &params->x[tidx];
    TYPE* ty = &params->y[tidx];
    TYPE* tz = &params->z[tidx];
    TYPE* tax = &params->ax[tidx];
    TYPE* tay = &params->ay[tidx];
    TYPE* taz = &params->az[tidx];
    TYPE* tp = &params->p[tidx];

    TYPE* sx = &params->x[sidx];
    TYPE* sy = &params->y[sidx];
    TYPE* sz = &params->z[sidx];
    TYPE* sw = &params->w[sidx];

    const size_t t_num_points = target->num_points;
    const size_t s_num_points = source->num_points;

    for (size_t i = 0; i < t_num_points; ++i)
    {
        TYPE ax = 0.0, ay = 0.0, az = 0.0, p = 0.0;
        const TYPE xi = tx[i], yi = ty[i], zi = tz[i];
        for (size_t j = 0; j < s_num_points; ++j)
        {
            const TYPE dx = sx[j] - xi;
            const TYPE dy = sy[j] - yi;
            const TYPE dz = sz[j] - zi;
            const TYPE wj = sw[j];
            const TYPE r = dx*dx + dy*dy + dz*dz;
            TYPE inv_r = TYPE_ONE/TYPE_SQRT(r);
            if (r == TYPE_ZERO) inv_r = TYPE_ZERO;
            const TYPE t = wj*inv_r;
            const TYPE s = t*inv_r*inv_r;
            ax += dx*s;
            ay += dy*s;
            az += dz*s;
            p  += t;
        }
        #pragma omp atomic
        tax[i] += ax;
        #pragma omp atomic
        tay[i] += ay;
        #pragma omp atomic
        taz[i] += az;
        #pragma omp atomic
        tp[i] += p;
    }
}

// Currently unused
//void p2p_one_node(t_fmm_params* params, t_node* target)
//{
//    const TYPE* const __restrict__ tx = target->x;
//    const TYPE* const __restrict__ ty = target->y;
//    const TYPE* const __restrict__ tz = target->z;
//    const TYPE* const __restrict__ sx = target->x;
//    const TYPE* const __restrict__ sy = target->y;
//    const TYPE* const __restrict__ sz = target->z;
//    const TYPE* const __restrict__ sw = target->w;
//    TYPE* const __restrict__ tax = target->ax;
//    TYPE* const __restrict__ tay = target->ay;
//    TYPE* const __restrict__ taz = target->az;
//    TYPE* const __restrict__ tp = target->p;
//
//    const size_t t_num_points = target->num_points;
//    const size_t s_num_points = target->num_points;
//
//    for (size_t i = 0; i < t_num_points; ++i)
//    {
//        TYPE ax = 0.0, ay = 0.0, az = 0.0, p = 0.0;
//        const TYPE xi = tx[i], yi = ty[i], zi = tz[i];
//        for (size_t j = 0; j < s_num_points; ++j)
//        {
//            const TYPE dx = sx[j] - xi;
//            const TYPE dy = sy[j] - yi;
//            const TYPE dz = sz[j] - zi;
//            const TYPE wj = sw[j];
//            const TYPE r = dx*dx + dy*dy + dz*dz;
//            TYPE inv_r = TYPE_ONE/TYPE_SQRT(r);
//            if (r < 0.0000001) inv_r = 0.0;
//            const TYPE t = wj*inv_r;
//            const TYPE s = t*inv_r*inv_r;
//            ax += dx*s;
//            ay += dy*s;
//            az += dz*s;
//            p  += t;
//        }
//        tax[i] += ax;
//        tay[i] += ay;
//        taz[i] += az;
//        tp[i] += p;
//    }
//}

void p2m(t_fmm_params* params, t_node* node)
{
    int num_terms = params->num_terms;

    size_t pidx = node->point_idx;
    size_t midx = node->mult_idx;
    TYPE* nx = &params->x[pidx];
    TYPE* ny = &params->y[pidx];
    TYPE* nz = &params->z[pidx];
    TYPE* M_real = &params->M_array_real[midx];
    TYPE* M_imag = &params->M_array_imag[midx];

    for (size_t i = 0; i < node->num_points; ++i)
    {
        TYPE dx = nx[i] - node->center[0];
        TYPE dy = ny[i] - node->center[1];
        TYPE dz = nz[i] - node->center[2];
        TYPE inner_real[num_terms*num_terms];
        TYPE inner_imag[num_terms*num_terms];
        TYPE r, theta, phi;
        cart_to_sph(dx, dy, dz, &r, &theta, &phi);
        compute_inner(params, r, theta, phi, inner_real, inner_imag);
        for (int n = 0; n < num_terms; ++n)
        {
            for (int m = -n; m <= n; ++m)
            {
                 cfma(neg_pow_n(n), TYPE_ZERO, inner_real[S_IDX(n,m)], inner_imag[S_IDX(n,m)],  
                    &M_real[S_IDX(n,m)], &M_imag[S_IDX(n,m)]);
            }
        }
    }
}


void m2m(t_fmm_params* params, t_node* parent)
{
    int num_terms = params->num_terms;
    size_t pmidx = parent->mult_idx;
    TYPE* parent_M_real = &params->M_array_real[pmidx];
    TYPE* parent_M_imag = &params->M_array_imag[pmidx];
    for (size_t i = 0; i < parent->num_children; ++i)
    {
        TYPE inner_real[num_terms*num_terms];
        TYPE inner_imag[num_terms*num_terms];
        t_node* child = get_node(params, parent->child[i]);
        
        size_t cmidx = child->mult_idx;
        TYPE* child_M_real = &params->M_array_real[cmidx];
        TYPE* child_M_imag = &params->M_array_imag[cmidx];
        
        TYPE dx = parent->center[0] - child->center[0];
        TYPE dy = parent->center[1] - child->center[1];
        TYPE dz = parent->center[2] - child->center[2];
        TYPE r, theta, phi;
        cart_to_sph(dx, dy, dz, &r, &theta, &phi);
        compute_inner(params, r, theta, phi, inner_real, inner_imag);
        for (int j = 0; j < num_terms; ++j)
        {   
            for (int k = -j; k <= j; ++k)
            {
                TYPE tmp_real = TYPE_ZERO;
                TYPE tmp_imag = TYPE_ZERO;
                for (int n = 0; n <= j; ++n)
                {
                    for (int m = -n; m <= n; ++m)
                    {
                        if (abs(k-m) <= j-n)
                        {
                            cfma(child_M_real[S_IDX(n,m)], child_M_imag[S_IDX(n,m)],
                                inner_real[S_IDX(j-n,k-m)], inner_imag[S_IDX(j-n,k-m)],
                                &tmp_real, &tmp_imag);
                        }
                    }
                }
                parent_M_real[S_IDX(j,k)] += tmp_real;
                parent_M_imag[S_IDX(j,k)] += tmp_imag;
            }
        }
    }
}
int m2l_calls = 0;
void m2l(t_fmm_params* params, t_node* target, t_node* source)
{
    m2l_calls++;
    int num_terms = params->num_terms;

    size_t midx = source->mult_idx;
    size_t lidx = target->mult_idx;
    TYPE* M_real = &params->M_array_real[midx];
    TYPE* M_imag = &params->M_array_imag[midx];
    TYPE* L_real = &params->L_array_real[lidx];
    TYPE* L_imag = &params->L_array_imag[lidx];
    
    TYPE dx = target->center[0] - source->center[0];
    TYPE dy = target->center[1] - source->center[1];
    TYPE dz = target->center[2] - source->center[2];

    TYPE outer_real[num_terms*num_terms];
    TYPE outer_imag[num_terms*num_terms];
    TYPE rho, alpha, beta;
    cart_to_sph(dx, dy, dz, &rho, &alpha, &beta);
    compute_outer(params, rho, alpha, beta, outer_real, outer_imag);

    for (int j = 0; j < num_terms; ++j)
    {
        for (int k = -j; k <= j; ++k)
        {
            TYPE tmp_real = TYPE_ZERO;
            TYPE tmp_imag = TYPE_ZERO;
            for (int n = 0; n < num_terms-j; ++n)
            {
                for (int m = -n; m <= n; ++m)
                {
                    cfma(M_real[S_IDX(n,m)], M_imag[S_IDX(n,m)],
                            outer_real[S_IDX(j+n,-k-m)], outer_imag[S_IDX(j+n,-k-m)],
                            &tmp_real, &tmp_imag);
                }
            }
            #pragma omp atomic
            L_real[S_IDX(j,k)] += tmp_real;
            #pragma omp atomic
            L_imag[S_IDX(j,k)] += tmp_imag;
        }
    }
}

void l2l(t_fmm_params* params, t_node* child, t_node* parent)
{
    int num_terms = params->num_terms;
    TYPE dx = child->center[0] - parent->center[0];
    TYPE dy = child->center[1] - parent->center[1];
    TYPE dz = child->center[2] - parent->center[2];

    size_t plidx = parent->mult_idx;
    size_t clidx = child->mult_idx;
    TYPE* parent_L_real = &params->L_array_real[plidx];
    TYPE* parent_L_imag = &params->L_array_imag[plidx];
    TYPE* child_L_real = &params->L_array_real[clidx];
    TYPE* child_L_imag = &params->L_array_imag[clidx];

    TYPE rho, alpha, beta;
    cart_to_sph(dx, dy, dz, &rho, &alpha, &beta);
    
    TYPE inner_real[num_terms*num_terms];
    TYPE inner_imag[num_terms*num_terms];
    compute_inner(params, rho, alpha, beta, inner_real, inner_imag);

    for (int j = 0; j < num_terms; ++j)
    {
        for (int k = -j; k <= j; ++k)
        {
            TYPE tmp_real = TYPE_ZERO;
            TYPE tmp_imag = TYPE_ZERO;
            for (int n = j; n < num_terms; ++n)
            {
                for (int m = -n; m <= n; ++m)
                {
                    if (abs(m-k) <= n - j)
                    {
                        cfma(parent_L_real[S_IDX(n,m)], parent_L_imag[S_IDX(n,m)],
                                inner_real[S_IDX(n-j,m-k)], inner_imag[S_IDX(n-j,m-k)],
                                &tmp_real, &tmp_imag);
                    }
                }
            }
            //child->L[S_IDX(j,k)] += l_tmp;
            child_L_real[S_IDX(j,k)] += tmp_real;
            child_L_imag[S_IDX(j,k)] += tmp_imag;
        }
    }
}
void l2p(t_fmm_params* params, t_node* node)
{
    int num_terms = params->num_terms;

    TYPE inner_real[num_terms*num_terms];
    TYPE inner_imag[num_terms*num_terms];
    TYPE inner_deriv_real[num_terms*num_terms];
    TYPE inner_deriv_imag[num_terms*num_terms];

    size_t pidx = node->point_idx;
    size_t midx = node->mult_idx;
    TYPE* nx = &params->x[pidx];
    TYPE* ny = &params->y[pidx];
    TYPE* nz = &params->z[pidx];
    TYPE* nax = &params->ax[pidx];
    TYPE* nay = &params->ay[pidx];
    TYPE* naz = &params->az[pidx];
    TYPE* np = &params->p[pidx];
    TYPE* L_real = &params->L_array_real[midx];
    TYPE* L_imag = &params->L_array_imag[midx];

    for (size_t i = 0; i < node->num_points; ++i)
    {
        TYPE dx = nx[i] - node->center[0];
        TYPE dy = ny[i] - node->center[1];
        TYPE dz = nz[i] - node->center[2];

        TYPE r, theta, phi;
        cart_to_sph(dx, dy, dz, &r, &theta, &phi);

        compute_inner_deriv(params, r, theta, phi, inner_real, inner_imag, inner_deriv_real, inner_deriv_imag);
        TYPE pot = TYPE_ZERO;
        TYPE rsum = TYPE_ZERO;
        TYPE thetasum = TYPE_ZERO;
        TYPE phisum = TYPE_ZERO;

        for (int n = 0; n < num_terms; ++n)
        {
            int m = 0;
            pot += cmul_real(L_real[S_IDX(n,m)], L_imag[S_IDX(n,m)], 
                inner_real[S_IDX(n,m)], inner_imag[S_IDX(n,m)]);
            rsum += (TYPE)n*cmul_real(L_real[S_IDX(n,m)], L_imag[S_IDX(n,m)], 
                inner_real[S_IDX(n,m)], inner_imag[S_IDX(n,m)]);
            thetasum += cmul_real(L_real[S_IDX(n,m)], L_imag[S_IDX(n,m)], 
                inner_deriv_real[S_IDX(n,m)], inner_deriv_imag[S_IDX(n,m)]);
            phisum += (TYPE)m*-cmul_imag(L_real[S_IDX(n,m)], L_imag[S_IDX(n,m)], 
                inner_real[S_IDX(n,m)], inner_imag[S_IDX(n,m)]);
            for (int m = 1; m <= n; ++m)
            {
                //pot         += 2.0*creal(node->L[S_IDX(n,m)]*inner[S_IDX(n,m)]);
                //rsum        += 2.0*(TYPE)n*creal(node->L[S_IDX(n,m)]*inner[S_IDX(n,m)]);
                //thetasum    += 2.0*creal(node->L[S_IDX(n,m)]*inner_deriv[S_IDX(n,m)]);
                //phisum      += 2.0*(TYPE)m*creal(node->L[S_IDX(n,m)]*inner[S_IDX(n,m)]*I);
                pot += 2.0*cmul_real(L_real[S_IDX(n,m)], L_imag[S_IDX(n,m)], 
                        inner_real[S_IDX(n,m)], inner_imag[S_IDX(n,m)]);
                rsum += 2.0*(TYPE)n*cmul_real(L_real[S_IDX(n,m)], L_imag[S_IDX(n,m)], 
                        inner_real[S_IDX(n,m)], inner_imag[S_IDX(n,m)]);
                thetasum += 2.0*cmul_real(L_real[S_IDX(n,m)], L_imag[S_IDX(n,m)], 
                        inner_deriv_real[S_IDX(n,m)], inner_deriv_imag[S_IDX(n,m)]);
                phisum += 2.0*(TYPE)m*-cmul_imag(L_real[S_IDX(n,m)], L_imag[S_IDX(n,m)], 
                        inner_real[S_IDX(n,m)], inner_imag[S_IDX(n,m)]);
            }
        }
        TYPE inv_r = TYPE_ONE/r;
        rsum *= inv_r;
        thetasum *= inv_r;
        phisum *= inv_r;
        phisum *= TYPE_ONE/TYPE_SIN(theta);

        TYPE ax, ay, az;
        sph_unit_to_cart_unit(r, theta, phi, rsum, thetasum, phisum, &ax, &ay, &az);

        np[i] += pot;
        nax[i] += ax;
        nay[i] += ay;
        naz[i] += az;
    }
}

void m2p(t_fmm_params* params, t_node* target, t_node* source)
{
    //int num_terms = params->num_terms;
    //for (size_t i = 0; i < target->num_points; ++i)
    //{
    //    TYPE dx = target->x[i] - source->center[0];
    //    TYPE dy = target->y[i] - source->center[1];
    //    TYPE dz = target->z[i] - source->center[2];
    //    TYPE r, theta, phi;
    //    cart_to_sph(dx, dy, dz, &r, &theta, &phi);
    //    
    //    TYPE_COMPLEX outer[num_terms*num_terms];
    //    TYPE_COMPLEX outer_deriv[num_terms*num_terms];
    //    compute_outer_deriv(params, r, theta, phi, outer, outer_deriv);
    //    
    //    TYPE pot = TYPE_ZERO;
    //    TYPE rsum = TYPE_ZERO;
    //    TYPE thetasum = TYPE_ZERO;
    //    TYPE phisum = TYPE_ZERO;
    //    for (int n = 0; n < num_terms; ++n)
    //    {
    //        int m = 0;
    //        pot         += creal(outer[S_IDX(n,-m)]*source->M[S_IDX(n,m)]);
    //        rsum        += (TYPE)-(n+1)*creal(outer[S_IDX(n,-m)]*source->M[S_IDX(n,m)]);
    //        thetasum    += creal(outer_deriv[S_IDX(n,-m)]*source->M[S_IDX(n,m)]);
    //        phisum      += (TYPE)m*creal(outer[S_IDX(n,-m)]*I*source->M[S_IDX(n,m)]);
    //        for (m = 1; m <= n; ++m)
    //        {
    //            pot         += 2.0*creal(outer[S_IDX(n,-m)]*source->M[S_IDX(n,m)]);
    //            rsum        += 2.0*(TYPE)-(n+1)*creal(outer[S_IDX(n,-m)]*source->M[S_IDX(n,m)]);
    //            thetasum    += 2.0*creal(outer_deriv[S_IDX(n,-m)]*source->M[S_IDX(n,m)]);
    //            phisum      += 2.0*(TYPE)m*creal(outer[S_IDX(n,-m)]*I*source->M[S_IDX(n,m)]);
    //        }
    //    }
    //    rsum *= TYPE_ONE/r;
    //    thetasum *= TYPE_ONE/r;
    //    phisum *= TYPE_ONE/r;

    //    phisum /= TYPE_SIN(theta);
    //    TYPE x, y, z;
    //    sph_unit_to_cart_unit(r, theta, phi, rsum, thetasum, phisum, &x, &y, &z);
    //    target->p[i] += pot;
    //    target->ax[i] += x;
    //    target->ay[i] += y;
    //    target->az[i] += z;
    //}
}

