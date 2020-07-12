#include <stdlib.h>
#include <stdio.h>

#include "type.h"
#include "params.h"
#include "tree.h"
#include "util.h"
#include "kernels.h"
#include "spharm.h"

    static inline
void cart_to_sph(TYPE x, TYPE y, TYPE z, TYPE* pr, TYPE* ptheta, TYPE* pphi)
{
    *pr = TYPE_SQRT(x*x+y*y+z*z);
    *ptheta = (*pr == TYPE_ZERO) ? TYPE_ZERO : TYPE_ACOS(z/(*pr));
    *pphi = TYPE_ATAN2(y, x);
}

    static inline
void sph_to_cart(TYPE r, TYPE theta, TYPE phi, TYPE*x, TYPE* y, TYPE* z)
{
    *x = r*TYPE_SIN(theta)*TYPE_COS(phi);
    *y = r*TYPE_SIN(theta)*TYPE_SIN(phi);
    *z = r*TYPE_COS(theta);
}

    static inline
void sph_unit_to_cart_unit(TYPE r, TYPE theta, TYPE phi, TYPE grad_r, TYPE grad_theta, TYPE grad_phi,
        TYPE* x, TYPE* y, TYPE* z)
{
    *x = TYPE_SIN(theta)*TYPE_COS(phi)*grad_r+TYPE_COS(theta)*TYPE_COS(phi)*grad_theta-TYPE_SIN(phi)*grad_phi;
    *y = TYPE_SIN(theta)*TYPE_SIN(phi)*grad_r+TYPE_COS(theta)*TYPE_SIN(phi)*grad_theta+TYPE_COS(phi)*grad_phi;
    *z = TYPE_COS(theta)*              grad_r-TYPE_SIN(theta)              *grad_theta;
}

// computes I^n
    static inline
TYPE_COMPLEX ipow(int n)
{
    TYPE_COMPLEX i;
    i = (n & 0x1) ? I : TYPE_ONE;
    i *= (n & 0x2) ? -TYPE_ONE : TYPE_ONE;
    return i;
}


// computes (-1)^n
static inline
TYPE neg_pow_n(int n)
{
    return (TYPE)(1 + ((n & 0x01)*-2));
}

#define S_IDX(n,m) ((n)*(n)+(n)+(m))

void p2p(t_fmm_params* params, t_node* target, t_node* source)
{
    const TYPE* const __restrict__ tx = target->x;
    const TYPE* const __restrict__ ty = target->y;
    const TYPE* const __restrict__ tz = target->z;
    const TYPE* const __restrict__ sx = source->x;
    const TYPE* const __restrict__ sy = source->y;
    const TYPE* const __restrict__ sz = source->z;
    const TYPE* const __restrict__ sw = source->w;
    TYPE* const __restrict__ tax = target->ax;
    TYPE* const __restrict__ tay = target->ay;
    TYPE* const __restrict__ taz = target->az;
    TYPE* const __restrict__ tp = target->p;

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
void p2p_one_node(t_fmm_params* params, t_node* target)
{
    const TYPE* const __restrict__ tx = target->x;
    const TYPE* const __restrict__ ty = target->y;
    const TYPE* const __restrict__ tz = target->z;
    const TYPE* const __restrict__ sx = target->x;
    const TYPE* const __restrict__ sy = target->y;
    const TYPE* const __restrict__ sz = target->z;
    const TYPE* const __restrict__ sw = target->w;
    TYPE* const __restrict__ tax = target->ax;
    TYPE* const __restrict__ tay = target->ay;
    TYPE* const __restrict__ taz = target->az;
    TYPE* const __restrict__ tp = target->p;

    const size_t t_num_points = target->num_points;
    const size_t s_num_points = target->num_points;

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
            if (r < 0.0000001) inv_r = 0.0;
            const TYPE t = wj*inv_r;
            const TYPE s = t*inv_r*inv_r;
            ax += dx*s;
            ay += dy*s;
            az += dz*s;
            p  += t;
        }
        tax[i] += ax;
        tay[i] += ay;
        taz[i] += az;
        tp[i] += p;
    }
}

void p2m(t_fmm_params* params, t_node* node)
{
    int num_terms = params->num_terms;
    for (size_t i = 0; i < node->num_points; ++i)
    {
        TYPE dx = node->x[i] - node->center[0];
        TYPE dy = node->y[i] - node->center[1];
        TYPE dz = node->z[i] - node->center[2];
        TYPE_COMPLEX inner[num_terms*num_terms];
        TYPE r, theta, phi;
        cart_to_sph(dx, dy, dz, &r, &theta, &phi);
        compute_inner(params, r, theta, phi, inner);
        for (int n = 0; n < num_terms; ++n)
        {
            for (int m = -n; m <= n; ++m)
            {
                 node->M[S_IDX(n,m)] += neg_pow_n(n)*inner[S_IDX(n, m)];
            }
        }
    }
}


void m2m(t_fmm_params* params, t_node* parent)
{
    int num_terms = params->num_terms;
    for (size_t i = 0; i < parent->num_children; ++i)
    {
        TYPE_COMPLEX inner[num_terms*num_terms];
        t_node* child = parent->child[i];
        TYPE dx = parent->center[0] - child->center[0];
        TYPE dy = parent->center[1] - child->center[1];
        TYPE dz = parent->center[2] - child->center[2];
        TYPE r, theta, phi;
        cart_to_sph(dx, dy, dz, &r, &theta, &phi);
        compute_inner(params, r, theta, phi, inner);
        for (int j = 0; j < num_terms; ++j)
        {   
            for (int k = -j; k <= j; ++k)
            {
                TYPE_COMPLEX tmp = 0.0 + I*0.0;
                for (int n = 0; n <= j; ++n)
                {
                    for (int m = -n; m <= n; ++m)
                    {
                        if (abs(k-m) <= j-n)
                        {
                            tmp += child->M[S_IDX(n,m)]*inner[S_IDX(j-n,k-m)];
                        }
                    }
                }
                parent->M[S_IDX(j,k)] += tmp;
            }
        }
    }
}
int m2l_calls = 0;
void m2l(t_fmm_params* params, t_node* target, t_node* source)
{
    m2l_calls++;
    int num_terms = params->num_terms;
    
    TYPE dx = target->center[0] - source->center[0];
    TYPE dy = target->center[1] - source->center[1];
    TYPE dz = target->center[2] - source->center[2];

    TYPE_COMPLEX outer[num_terms*num_terms];
    TYPE rho, alpha, beta;
    cart_to_sph(dx, dy, dz, &rho, &alpha, &beta);
    compute_outer(params, rho, alpha, beta, outer);

    for (int j = 0; j < num_terms; ++j)
    {
        for (int k = -j; k <= j; ++k)
        {
            TYPE_COMPLEX l_tmp = TYPE_ZERO;
            for (int n = 0; n < num_terms-j; ++n)
            {
                for (int m = -n; m <= n; ++m)
                {
                    l_tmp += source->M[S_IDX(n,m)]*outer[S_IDX(j+n,-k-m)];
                }
            }
            //#pragma omp atomic
            //target->L[S_IDX(j,k)] += l_tmp;
            #pragma omp atomic
            ((TYPE*)&target->L[S_IDX(j,k)])[0] += creal(l_tmp);
            #pragma omp atomic
            ((TYPE*)&target->L[S_IDX(j,k)])[1] += cimag(l_tmp);
        }
    }
}

void l2l(t_fmm_params* params, t_node* child, t_node* parent)
{
    int num_terms = params->num_terms;
    TYPE dx = child->center[0] - parent->center[0];
    TYPE dy = child->center[1] - parent->center[1];
    TYPE dz = child->center[2] - parent->center[2];

    TYPE rho, alpha, beta;
    cart_to_sph(dx, dy, dz, &rho, &alpha, &beta);
    
    TYPE_COMPLEX inner[num_terms*num_terms];
    compute_inner(params, rho, alpha, beta, inner);

    for (int j = 0; j < num_terms; ++j)
    {
        for (int k = -j; k <= j; ++k)
        {
            TYPE_COMPLEX l_tmp = TYPE_ZERO;
            for (int n = j; n < num_terms; ++n)
            {
                for (int m = -n; m <= n; ++m)
                {
                    if (abs(m-k) <= n - j)
                    {
                        l_tmp += parent->L[S_IDX(n,m)]*inner[S_IDX(n-j, m-k)];
                    }
                }
            }
            child->L[S_IDX(j,k)] += l_tmp;
        }
    }
}
void l2p(t_fmm_params* params, t_node* node)
{
    int num_terms = params->num_terms;

    TYPE_COMPLEX inner[num_terms*num_terms];
    TYPE_COMPLEX inner_deriv[num_terms*num_terms];

    for (size_t i = 0; i < node->num_points; ++i)
    {
        TYPE dx = node->x[i] - node->center[0];
        TYPE dy = node->y[i] - node->center[1];
        TYPE dz = node->z[i] - node->center[2];

        TYPE r, theta, phi;
        cart_to_sph(dx, dy, dz, &r, &theta, &phi);

        compute_inner_deriv(params, r, theta, phi, inner, inner_deriv);
        TYPE pot = TYPE_ZERO;
        TYPE rsum = TYPE_ZERO;
        TYPE thetasum = TYPE_ZERO;
        TYPE phisum = TYPE_ZERO;

        for (int n = 0; n < num_terms; ++n)
        {
            int m = 0;
            pot += creal(node->L[S_IDX(n,m)]*inner[S_IDX(n,m)]);
            rsum += (TYPE)n*creal(node->L[S_IDX(n,m)]*inner[S_IDX(n,m)]);
            thetasum += creal(node->L[S_IDX(n,m)]*inner_deriv[S_IDX(n,m)]);
            phisum += (TYPE)m*creal(node->L[S_IDX(n,m)]*inner[S_IDX(n,m)]*I);
            for (int m = 1; m <= n; ++m)
            {
                // TODO change creal, cimag functions to type agnostic
                pot         += 2.0*creal(node->L[S_IDX(n,m)]*inner[S_IDX(n,m)]);
                rsum        += 2.0*(TYPE)n*creal(node->L[S_IDX(n,m)]*inner[S_IDX(n,m)]);
                thetasum    += 2.0*creal(node->L[S_IDX(n,m)]*inner_deriv[S_IDX(n,m)]);
                phisum      += 2.0*(TYPE)m*creal(node->L[S_IDX(n,m)]*inner[S_IDX(n,m)]*I);
            }
        }
        TYPE inv_r = TYPE_ONE/r;
        rsum *= inv_r;
        thetasum *= inv_r;
        phisum *= inv_r;
        phisum *= TYPE_ONE/TYPE_SIN(theta);

        TYPE ax, ay, az;
        sph_unit_to_cart_unit(r, theta, phi, rsum, thetasum, phisum, &ax, &ay, &az);

        node->p[i] += pot;
        node->ax[i] += ax;
        node->ay[i] += ay;
        node->az[i] += az;
    }
}

void m2p(t_fmm_params* params, t_node* target, t_node* source)
{
    int num_terms = params->num_terms;
    for (size_t i = 0; i < target->num_points; ++i)
    {
        TYPE dx = target->x[i] - source->center[0];
        TYPE dy = target->y[i] - source->center[1];
        TYPE dz = target->z[i] - source->center[2];
        TYPE r, theta, phi;
        cart_to_sph(dx, dy, dz, &r, &theta, &phi);
        
        TYPE_COMPLEX outer[num_terms*num_terms];
        TYPE_COMPLEX outer_deriv[num_terms*num_terms];
        compute_outer_deriv(params, r, theta, phi, outer, outer_deriv);
        
        TYPE pot = TYPE_ZERO;
        TYPE rsum = TYPE_ZERO;
        TYPE thetasum = TYPE_ZERO;
        TYPE phisum = TYPE_ZERO;
        for (int n = 0; n < num_terms; ++n)
        {
            int m = 0;
            pot         += creal(outer[S_IDX(n,-m)]*source->M[S_IDX(n,m)]);
            rsum        += (TYPE)-(n+1)*creal(outer[S_IDX(n,-m)]*source->M[S_IDX(n,m)]);
            thetasum    += creal(outer_deriv[S_IDX(n,-m)]*source->M[S_IDX(n,m)]);
            phisum      += (TYPE)m*creal(outer[S_IDX(n,-m)]*I*source->M[S_IDX(n,m)]);
            for (m = 1; m <= n; ++m)
            {
                pot         += 2.0*creal(outer[S_IDX(n,-m)]*source->M[S_IDX(n,m)]);
                rsum        += 2.0*(TYPE)-(n+1)*creal(outer[S_IDX(n,-m)]*source->M[S_IDX(n,m)]);
                thetasum    += 2.0*creal(outer_deriv[S_IDX(n,-m)]*source->M[S_IDX(n,m)]);
                phisum      += 2.0*(TYPE)m*creal(outer[S_IDX(n,-m)]*I*source->M[S_IDX(n,m)]);
            }
        }
        rsum *= TYPE_ONE/r;
        thetasum *= TYPE_ONE/r;
        phisum *= TYPE_ONE/r;

        phisum /= TYPE_SIN(theta);
        TYPE x, y, z;
        sph_unit_to_cart_unit(r, theta, phi, rsum, thetasum, phisum, &x, &y, &z);
        target->p[i] += pot;
        target->ax[i] += x;
        target->ay[i] += y;
        target->az[i] += z;
    }
}

