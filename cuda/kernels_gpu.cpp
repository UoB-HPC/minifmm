#include "spharm_gpu.h"
#include "tree.h"

//typedef Kokkos::View<TYPE*> t_view;
#define S_IDX(n,m) ((n)*(n)+(n)+(m))

//__device__
//void cart_to_sph(TYPE x, TYPE y, TYPE z, TYPE* pr, TYPE* ptheta, TYPE* pphi)
//{
//    *pr = TYPE_SQRT(x*x+y*y+z*z);
//    *ptheta = (*pr == TYPE_ZERO) ? TYPE_ZERO : TYPE_ACOS(z/(*pr));
//    *pphi = TYPE_ATAN2(y, x);
//}

//__noinline__    __device__
//void m2l_kokkos_team(t_node* target, t_node* source, long tid, long tsz,
//    t_view& d_m_real, t_view& d_m_imag, t_view& d_l_real, t_view& d_l_imag)
//{
//    if (tid != 0) return;
//    const int num_terms = 4;
//    size_t midx = source->mult_idx;
//    size_t lidx = target->mult_idx;
//    TYPE* M_real = &d_m_real[midx];
//    TYPE* M_imag = &d_m_imag[midx];
//    TYPE* L_real = &d_l_real[lidx];
//    TYPE* L_imag = &d_l_imag[lidx];
//
//    TYPE dx = target->center[0] - source->center[0];
//    TYPE dy = target->center[1] - source->center[1];
//    TYPE dz = target->center[2] - source->center[2];
//
//    __shared__ TYPE outer_real[4*num_terms*num_terms];
//    __shared__ TYPE outer_imag[4*num_terms*num_terms];
//
//    const long wid = threadIdx.z;
//    const long woff = wid*num_terms*num_terms;
//
//    TYPE rho, alpha, beta;
//    cart_to_sph(dx, dy, dz, &rho, &alpha, &beta);
//    compute_outer_gpu(num_terms, rho, alpha, beta, &outer_real[wid*num_terms*num_terms], &outer_imag[wid*num_terms*num_terms]);
//
//    //for (int i = 0; i < num_terms*num_terms; ++i)
//    //{
//    //    atomicAdd(&L_real[i], outer_real[wid*num_terms*num_terms+i]);
//    //    atomicAdd(&L_imag[i], outer_imag[wid*num_terms*num_terms+i]);
//    //}
//
//    //printf("performing m2l\n");
//
//    for (int j = 0; j < num_terms; ++j)
//    {
//        for (int k = -j; k <= j; ++k)
//        {
//            TYPE tmp_real = TYPE_ZERO;
//            TYPE tmp_imag = TYPE_ZERO;
//            for (int n = 0; n < num_terms-j; ++n)
//            {
//                for (int m = -n; m <= n; ++m)
//                {
//                    tmp_real += M_real[S_IDX(n,m)]*outer_real[woff+S_IDX(j+n,-k-m)] - 
//                        M_imag[S_IDX(n,m)]*outer_imag[woff+S_IDX(j+n,-k-m)];
//                    tmp_imag += M_real[S_IDX(n,m)]*outer_imag[woff+S_IDX(j+n,-k-m)] + 
//                        M_imag[S_IDX(n,m)]*outer_real[woff+S_IDX(j+n,-k-m)];
//                }
//            }
//            //L_real[S_IDX(j,k)] += tmp_real;
//            //L_imag[S_IDX(j,k)] += tmp_imag;
//            atomicAdd(&L_real[S_IDX(j,k)], tmp_real);
//            atomicAdd(&L_imag[S_IDX(j,k)], tmp_imag);
//        }
//    }
//}
//
//__noinline__  __device__
//void olved extern function2p_kokkos_team(t_node* target, long tid, long tsz,
//    t_view& d_x, t_view& d_y, t_view& d_z, t_view& d_w,
//    t_view& d_ax, t_view& d_ay, t_view& d_az, t_view& d_p)
//{
//    size_t tidx = target->point_idx;
//
//    const TYPE* __restrict__ tx    = &d_x[tidx];
//    const TYPE* __restrict__ ty    = &d_y[tidx];
//    const TYPE* __restrict__ tz    = &d_z[tidx];
//    TYPE* __restrict__ tax   = &d_ax[tidx];
//    TYPE* __restrict__ tay   = &d_ay[tidx];
//    TYPE* __restrict__ taz   = &d_az[tidx];
//    TYPE* __restrict__ tp    = &d_p[tidx];
//
//    const TYPE* __restrict sx    = &d_x[tidx];
//    const TYPE* __restrict sy    = &d_y[tidx];
//    const TYPE* __restrict sz    = &d_z[tidx];
//    const TYPE* __restrict sw    = &d_w[tidx];
//
//    const size_t t_num_points = target->num_points;
//
//    const long shmem_sz = 128;
//    __shared__ TYPE shmem[shmem_sz*4*4];
//
//    int warp_id = threadIdx.z;
//
//    for (size_t i = tid; i < t_num_points; i += tsz)
//    {
//        if (i >= t_num_points) continue;
//
//        TYPE ax = 0.0, ay = 0.0, az = 0.0, p = 0.0;
//        const TYPE xi = tx[i], yi = ty[i], zi = tz[i];
//
//        for (long jb = 0; jb < t_num_points; jb += shmem_sz)
//        {
//            #pragma unroll 32
//            for (long j = tid; j < shmem_sz; j += tsz)
//            {
//                if (j+jb >= t_num_points) continue;
//                shmem[warp_id*shmem_sz*4 + 0*shmem_sz + j] = sx[j+jb];
//                shmem[warp_id*shmem_sz*4 + 1*shmem_sz + j] = sy[j+jb];
//                shmem[warp_id*shmem_sz*4 + 2*shmem_sz + j] = sz[j+jb];
//                shmem[warp_id*shmem_sz*4 + 3*shmem_sz + j] = sw[j+jb];
//            }
//
//            #pragma unroll 32
//            for (long j = 0; j < shmem_sz; ++j)
//            {
//                if (j+jb >= t_num_points) continue;
//                const TYPE dx = shmem[warp_id*shmem_sz*4 + 0*shmem_sz + j] - xi;
//                const TYPE dy = shmem[warp_id*shmem_sz*4 + 1*shmem_sz + j] - yi;
//                const TYPE dz = shmem[warp_id*shmem_sz*4 + 2*shmem_sz + j] - zi;
//                const TYPE wj = shmem[warp_id*shmem_sz*4 + 3*shmem_sz + j];
//                const TYPE r = dx*dx + dy*dy + dz*dz;
//                //TYPE inv_r = TYPE_ONE/TYPE_SQRT(r);
//                TYPE inv_r = (r == 0) ? 0.0 : rsqrtf(r);
//                const TYPE t = wj*inv_r;
//                const TYPE s = t*inv_r*inv_r;
//                ax += dx*s;
//                ay += dy*s;
//                az += dz*s;
//                p  += t;
//            }
//        }
//        atomicAdd(&tax[i], ax);
//        atomicAdd(&tay[i], ay);
//        atomicAdd(&taz[i], az);
//        atomicAdd(&tp[i], p);
//        //tax[i] += ax;
//        //tay[i] += ay;
//        //taz[i] += az;
//        //tp[i] += p;
//    }
//}
//
//__noinline__  __device__
//void p2p_kokkos_team(t_node* target, t_node* source, long tid, long tsz,
//    t_view& d_x, t_view& d_y, t_view& d_z, t_view& d_w,
//    t_view& d_ax, t_view& d_ay, t_view& d_az, t_view& d_p)
//{
//    size_t tidx = target->point_idx;
//    size_t sidx = source->point_idx;
//
//    const TYPE* __restrict__ tx    = &d_x[tidx];
//    const TYPE* __restrict__ ty    = &d_y[tidx];
//    const TYPE* __restrict__ tz    = &d_z[tidx];
//    TYPE* __restrict__ tax   = &d_ax[tidx];
//    TYPE* __restrict__ tay   = &d_ay[tidx];
//    TYPE* __restrict__ taz   = &d_az[tidx];
//    TYPE* __restrict__ tp    = &d_p[tidx];
//
//    const TYPE* __restrict sx    = &d_x[sidx];
//    const TYPE* __restrict sy    = &d_y[sidx];
//    const TYPE* __restrict sz    = &d_z[sidx];
//    const TYPE* __restrict sw    = &d_w[sidx];
//
//    const size_t t_num_points = target->num_points;
//    const size_t s_num_points = source->num_points;
//
//    const long shmem_sz = 128;
//    __shared__ TYPE shmem[shmem_sz*4*4];
//
//    int warp_id = threadIdx.z;
//
//    for (size_t i = tid; i < t_num_points; i += tsz)
//    {
//        if (i >= t_num_points) continue;
//
//        TYPE ax = 0.0, ay = 0.0, az = 0.0, p = 0.0;
//        const TYPE xi = tx[i], yi = ty[i], zi = tz[i];
//
//        for (long jb = 0; jb < s_num_points; jb += shmem_sz)
//        {
//            #pragma unroll 32
//            for (long j = tid; j < shmem_sz; j += tsz)
//            {
//                if (j+jb >= s_num_points) continue;
//                shmem[warp_id*shmem_sz*4 + 0*shmem_sz + j] = sx[j+jb];
//                shmem[warp_id*shmem_sz*4 + 1*shmem_sz + j] = sy[j+jb];
//                shmem[warp_id*shmem_sz*4 + 2*shmem_sz + j] = sz[j+jb];
//                shmem[warp_id*shmem_sz*4 + 3*shmem_sz + j] = sw[j+jb];
//            }
//
//            #pragma unroll 32
//            for (long j = 0; j < shmem_sz; ++j)
//            {
//                if (j+jb >= s_num_points) continue;
//                const TYPE dx = shmem[warp_id*shmem_sz*4 + 0*shmem_sz + j] - xi;
//                const TYPE dy = shmem[warp_id*shmem_sz*4 + 1*shmem_sz + j] - yi;
//                const TYPE dz = shmem[warp_id*shmem_sz*4 + 2*shmem_sz + j] - zi;
//                const TYPE wj = shmem[warp_id*shmem_sz*4 + 3*shmem_sz + j];
//                const TYPE r = dx*dx + dy*dy + dz*dz;
//                //TYPE inv_r = TYPE_ONE/TYPE_SQRT(r);
//                TYPE inv_r = rsqrtf(r);
//                const TYPE t = wj*inv_r;
//                const TYPE s = t*inv_r*inv_r;
//                ax += dx*s;
//                ay += dy*s;
//                az += dz*s;
//                p  += t;
//            }
//        }
//        atomicAdd(&tax[i], ax);
//        atomicAdd(&tay[i], ay);
//        atomicAdd(&taz[i], az);
//        atomicAdd(&tp[i], p);
//        //tax[i] += ax;
//        //tay[i] += ay;
//        //taz[i] += az;
//        //tp[i] += p;
//    }
//}

