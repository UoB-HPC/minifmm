#include <stdlib.h>

#include "type.h"
#include "params.h"
#include "tree.h"
#include "util.h"
#include "kernels.h"
#include "spharm.h"

#define M_INDEX(n, m) ((n)*(n)+((m)+(n)))

#define MS_INDEX(n, m) ((n)*((n)+1)/2 + (m))

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
    *x = TYPE_SIN(theta)*TYPE_COS(phi)*grad_r + TYPE_COS(theta)*TYPE_COS(phi)*grad_theta - TYPE_SIN(phi)*grad_phi;
    *y = TYPE_SIN(theta)*TYPE_SIN(phi)*grad_r + TYPE_COS(theta)*TYPE_SIN(phi)*grad_theta + TYPE_COS(phi)*grad_phi;
    *z = TYPE_COS(theta)*              grad_r - TYPE_SIN(theta)         *grad_theta;
}

// do we need to make this mutual, i.e. add to one, sub from other?
// leaf 1 = target, leaf 2 = source
void p2p(t_fmm_options* options, t_node* target, t_node* source)
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
        TYPE xi = tx[i], yi = ty[i], zi = tz[i];
        for (size_t j = 0; j < s_num_points; ++j)
        {
            TYPE dx = sx[j] - xi;
            TYPE dy = sy[j] - yi;
            TYPE dz = sz[j] - zi;

            TYPE inv_r = TYPE_ONE/TYPE_SQRT(dx*dx + dy*dy + dz*dz);
            TYPE inv_r_3 = inv_r * inv_r * inv_r;
            ax += sw[j]*dx*inv_r_3;
            ay += sw[j]*dy*inv_r_3;
            az += sw[j]*dz*inv_r_3;
            p += sw[j]*inv_r;
        }
        tax[i] += ax;
        tay[i] += ay;
        taz[i] += az;
        tp[i] += p;
    }
}

void p2p_one_node(t_fmm_options* options, t_node* node)
{
    for (size_t i = 0; i < node->num_points; ++i)
    {
        TYPE ax = 0.0, ay = 0.0, az = 0.0, p = 0.0;
        TYPE xi = node->x[i], yi = node->y[i], zi = node->z[i];
        for (size_t j = 0; j < node->num_points; ++j)
        {
            TYPE dx = node->x[j] - xi;
            TYPE dy = node->y[j] - yi;
            TYPE dz = node->z[j] - zi;

            TYPE r = TYPE_SQRT(dx*dx + dy*dy + dz*dz);
            TYPE inv_r = (r == 0.0) ? 0.0 : 1.0/r;
            TYPE inv_r_3 = inv_r * inv_r * inv_r;
            ax += node->w[j]*dx*inv_r_3;
            ay += node->w[j]*dy*inv_r_3;
            az += node->w[j]*dz*inv_r_3;
            p += node->w[j]*inv_r;
        }
        node->ax[i] += ax;
        node->ay[i] += ay;
        node->az[i] += az;
        node->p[i] += p;
    }
}

void p2m(t_fmm_options* options, t_node* node)
{
    TYPE_COMPLEX Y_rn[options->num_spharm_terms];
    for (size_t i = 0; i < node->num_points; ++i)
    {
        TYPE dx = node->x[i] - node->center[0];
        TYPE dy = node->y[i] - node->center[1];
        TYPE dz = node->z[i] - node->center[2];
        TYPE w = node->w[i];
        TYPE rho, alpha, beta;
        cart_to_sph(dx, dy, dz, &rho, &alpha, &beta);
        spharm_r_n(options, 1.0, rho, alpha, beta, Y_rn);
        for (int n = 0; n < options->num_terms; ++n)
        {
            for (int m = 0; m <= n; ++m)
            {
                TYPE_COMPLEX y = Y_rn[n*n+(n-m)];
                node->M[MS_INDEX(n,m)] += y*w;
            }
        }
    }
}

void m2m(t_fmm_options* options, t_node* parent)
{
    TYPE_COMPLEX Y_rn[options->num_spharm_terms];
    TYPE* A = options->A;
    for (size_t i = 0; i < parent->num_children; ++i)
    {
        t_node* child = parent->child[i];
        TYPE dx = parent->center[0] - child->center[0];
        TYPE dy = parent->center[1] - child->center[1];
        TYPE dz = parent->center[2] - child->center[2];
        TYPE rho, alpha, beta;
        cart_to_sph(dx, dy, dz, &rho, &alpha, &beta);
        spharm_r_n(options, TYPE_ONE, rho, alpha, beta, Y_rn);
        for (int j = 0; j < options->num_terms; ++j)
        {
            for (int k = 0; k <= j; ++k)
            { 
                TYPE_COMPLEX m_tmp = 0.0;
                int jk = j*j+j+k;
                for (int n = 0; n <= j; ++n)
                {
                    for (int m = MAX(-n,-j+k+n); m <= MIN(k-1, n); ++m)
                    {
                        int nm = n*n+n+m;
                        int jnkm = (j-n)*(j-n)+(j-n)+(k-m);
                        TYPE real_terms = A[nm]*A[jnkm]/A[jk];
                        int index = MS_INDEX(j-n, k-m);
                        real_terms *= (n & 1) ? -1.0 : 1.0;
                        real_terms *= (m < 0 && (m&1)) ? -1.0 : 1.0;
                        TYPE_COMPLEX complex_terms = child->M[index]*Y_rn[n*n+(n-m)];
                        TYPE_COMPLEX m_curr = real_terms*complex_terms;
                        m_tmp += m_curr;
                    }
                    for (int m = k; m <= MIN(n,j+k-n); ++m)
                    {
                        int nm = n*n+n+m;
                        int jnkm = (j-n)*(j-n)+(j-n)+(k-m);
                        TYPE real_terms = A[nm]*A[jnkm]/A[jk];
                        int index = MS_INDEX(j-n, m-k);
                        TYPE_COMPLEX complex_terms = conj(child->M[index])*Y_rn[n*n+n-m];
                        
                        TYPE sign = ((k+m+n) & 1) ? -1.0 : 1.0;
                        TYPE_COMPLEX m_curr = sign*real_terms*complex_terms;

                        m_tmp += m_curr;
                    }
                }
                int index = MS_INDEX(j,k);
                parent->M[index] += m_tmp;
            }
        }
    }
}

// LOCAL for N1 constructed from MULTIPOLE of N2
// i.e. n1 = target, n2 = source
// TODO CHANGE "A CALCULATION" FOR LOCAL - i.e. -1^n -> -1
int m2l_calls=0;
void m2l(t_fmm_options* options, t_node* target, t_node* source)
{
    ++m2l_calls;
    TYPE dx = target->center[0] - source->center[0];
    TYPE dy = target->center[1] - source->center[1];
    TYPE dz = target->center[2] - source->center[2];

    TYPE rho, alpha, beta;
    cart_to_sph(dx, dy, dz, &rho, &alpha, &beta);

    TYPE_COMPLEX Y_rn[options->num_spharm_terms];
    TYPE inv_rho = TYPE_ONE/rho;
    spharm_r_n(options, inv_rho, inv_rho, alpha, beta, Y_rn);
    for (int j = 0; j < options->num_terms; ++j)
    {
        for (int k = 0; k <= j; ++k)
        {
            TYPE_COMPLEX l_tmp = TYPE_ZERO;
            int jk = j*j+j+k;
            for (int n = 0; n < options->num_terms-j; ++n)
            {
                for (int m = -n; m < 0; ++m)
                {
                    int nm = n*n+n+m;
                    int cindex = jk*options->num_terms*options->num_terms + nm;
                    int index = MS_INDEX(n,-m);
                    int spharm_index = (j+n)*(j+n)+((j+n)+(m-k));
                    l_tmp += options->C[cindex]*conj(source->M[index])*Y_rn[spharm_index];
                }
                for (int m = 0; m <= n; ++m)
                {
                    int nm = n*n+n+m;
                    int cindex = jk*options->num_terms*options->num_terms + nm;
                    int index = MS_INDEX(n,m);
                    int spharm_index = (j+n)*(j+n)+((j+n)+(m-k));
                    l_tmp += options->C[cindex]*source->M[index]*Y_rn[spharm_index];
                }
            }
            int index = MS_INDEX(j,k);
            target->L[index] += l_tmp;
        }
    }
}

// n1 = branch, n2 = parent
void l2l(t_fmm_options* options, t_node* child, t_node* parent)       
{
    // is this the right way around?
    TYPE dx = child->center[0] - parent->center[0];
    TYPE dy = child->center[1] - parent->center[1];
    TYPE dz = child->center[2] - parent->center[2];

    TYPE rho, alpha, beta;
    cart_to_sph(dx, dy, dz, &rho, &alpha, &beta);

    TYPE_COMPLEX Y_rn[options->num_spharm_terms];
    TYPE* A = options->A;
    spharm_r_n(options, TYPE_ONE, rho, alpha, beta, Y_rn);

    for (int j = 0; j < options->num_terms; ++j)
    {
        for (int k = 0; k <= j; ++k)
        {
            int jk = j*j+j+k;
            TYPE_COMPLEX l_tmp = 0.0 + I*0.0;
            for (int n = j; n < options->num_terms; ++n)
            {
                for (int m = j+k-n; m < 0; ++m)
                {
                    TYPE sign = ((k) & 1) ? -1.0 : 1.0;
                    int nm = n*n+n+m;
                    int njmk = (n-j)*(n-j)+(n-j)+(m-k);
                    TYPE real_terms = A[njmk]*A[jk]/A[nm];
                    real_terms *= sign;
                    int index = MS_INDEX(n, -m);
                    TYPE_COMPLEX complex_terms = conj(parent->L[index])*Y_rn[njmk];
                    TYPE_COMPLEX l_curr = real_terms * complex_terms;
                    l_tmp += l_curr;
                }
                for (int m = 0; m <= n; ++m)
                {
                    if (n-j >= abs(m-k))
                    {
                        int nm = n*n+n+m;
                        int njmk = (n-j)*(n-j)+(n-j)+(m-k);
                        TYPE real_terms = A[njmk]*A[jk]/A[nm];

                        int index = MS_INDEX(n, m);
                        TYPE_COMPLEX complex_terms = parent->L[index]*Y_rn[njmk];
                        TYPE_COMPLEX l_curr = real_terms * complex_terms;

                        int ipow = abs(m) - abs(m-k) - abs(k);
                        l_curr *= (ipow & 1) ? I : 1;
                        l_curr *= (ipow & 2) ? -1 : 1;

                        l_tmp += l_curr;
                    }
                }
            }
            int index = MS_INDEX(j,k);
            child->L[index] += l_tmp;
        }
    }
}

void l2p(t_fmm_options* options, t_node* node)
{
    TYPE_COMPLEX Y_rn[options->num_spharm_terms];
    TYPE_COMPLEX Y_rn_div[options->num_spharm_terms];

    for (size_t i = 0; i < node->num_points; ++i)
    {
        TYPE dx = node->x[i] - node->center[0];
        TYPE dy = node->y[i] - node->center[1];
        TYPE dz = node->z[i] - node->center[2];

        TYPE r, theta, phi;
        cart_to_sph(dx, dy, dz, &r, &theta, &phi);
        
        TYPE grad_r = TYPE_ZERO, grad_theta = TYPE_ZERO, grad_phi = TYPE_ZERO;
        TYPE inv_r = TYPE_ONE/r;

        spharm_r_n_d_theta(options, TYPE_ONE, r, theta, phi, Y_rn, Y_rn_div);

        TYPE tmp_pot = 0.0;
        for (int n = 0; n < options->num_terms; ++n)
        {
            int m = 0;
            TYPE_COMPLEX y = Y_rn[n*n+n+m];
            int index = MS_INDEX(n,m);
            TYPE_COMPLEX l_val = node->L[index];

            TYPE_COMPLEX l_y = l_val*y;
            tmp_pot += creal(l_y);

            grad_r += creal(l_y)*n;
            grad_theta += creal(Y_rn_div[n*n+n+m]*l_val);
            grad_phi += creal(l_y*I)*m;

            for (m = 1; m <= n; ++m)
            {
                TYPE_COMPLEX y = Y_rn[n*n+n+m];
                int index = MS_INDEX(n,m);
                TYPE_COMPLEX l_val = node->L[index];

                TYPE_COMPLEX l_y = l_val*y;
                tmp_pot += TYPE_TWO*creal(l_y);
                grad_r += TYPE_TWO*creal(l_y) * n;
                grad_theta += TYPE_TWO*creal(Y_rn_div[n*n+n+m]* l_val);
                grad_phi += TYPE_TWO*creal(l_y*I)*m;
            }
        }
        TYPE sin_theta = TYPE_SIN(theta);
        grad_phi *= (sin_theta == TYPE_ZERO) ? TYPE_ONE : TYPE_ONE/sin_theta;
        grad_r *= inv_r;
        grad_theta *= inv_r;
        grad_phi *= inv_r;

        TYPE grad_x, grad_y, grad_z;
        sph_unit_to_cart_unit(r, theta, phi, grad_r, grad_theta, grad_phi, &grad_x, &grad_y, &grad_z);
        node->ax[i] += grad_x;
        node->ay[i] += grad_y;
        node->az[i] += grad_z;
        node->p[i] += tmp_pot;
    }
}

void m2p(t_fmm_options* options, t_node* target, t_node* source)
{
    TYPE_COMPLEX Y_rn[options->num_spharm_terms];
    TYPE_COMPLEX Y_rn_div[options->num_spharm_terms];

    for (size_t i = 0; i < target->num_points; ++i)
    {
        TYPE dx = target->x[i] - source->center[0];
        TYPE dy = target->y[i] - source->center[1];
        TYPE dz = target->z[i] - source->center[2];

        TYPE r, theta, phi;
        cart_to_sph(dx, dy, dz, &r, &theta, &phi);
        
        TYPE grad_r = TYPE_ZERO, grad_theta = TYPE_ZERO, grad_phi = TYPE_ZERO;
        TYPE inv_r = TYPE_ONE/r;
        TYPE tmp_pot = TYPE_ZERO;
        spharm_r_n_d_theta(options, inv_r, inv_r, theta, phi, Y_rn, Y_rn_div);
        
        for (int n = 0; n < options->num_terms; ++n)
        {
            int m = 0;
            TYPE_COMPLEX y = Y_rn[n*n+n+m];
            int index = MS_INDEX(n,m);
            TYPE_COMPLEX m_val = source->M[index];
            TYPE_COMPLEX m_y = m_val*y;
                
            tmp_pot += creal(m_y);

            grad_r += creal(m_y) * -(n+1);
            TYPE_COMPLEX d_y = Y_rn_div[n*n+n+m];
            grad_theta += creal(d_y * m_val);
            grad_phi += creal(m_y*I) * m;

            for (m = 1; m <= n; ++m)
            {
                TYPE_COMPLEX y = Y_rn[n*n+n+m];
                int index = MS_INDEX(n,m);
                TYPE_COMPLEX m_val = source->M[index];
                TYPE_COMPLEX m_y = m_val*y;
                
                tmp_pot += 2.0*creal(m_y);
                // the following 3 lines need careful checking
                grad_r += 2.0*creal(m_y) * -(n+1);
                TYPE_COMPLEX d_y = Y_rn_div[n*n+n+m];
                grad_theta += 2.0*creal(d_y * m_val);
                grad_phi += 2.0*creal(m_y*I) * m;
            }
        }
        TYPE sin_theta = sin(theta);
        grad_phi *= (sin_theta == 0.0) ? 1.0 : 1.0/sin_theta;
        grad_r *= inv_r;
        grad_phi *= inv_r;
        grad_theta *= inv_r;

        TYPE grad_x, grad_y, grad_z;
        sph_unit_to_cart_unit(r, theta, phi, grad_r, grad_theta, grad_phi, &grad_x, &grad_y, &grad_z);
        // should this be += or =?
        target->ax[i] += grad_x;
        target->ay[i] += grad_y;
        target->az[i] += grad_z;
        target->p[i] += tmp_pot;
    }
}
