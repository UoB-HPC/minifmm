#include <stdlib.h>

#include <vector>

#include "type.h"
#include "tree.h"
#include "kernels.h"
#include "timer.h"
#include "dtt.h"

#include "spharm_gpu.h"
#include "cuda_utils.h"

void dual_tree_traversal_core(t_fmm_params* params, t_node* target, t_node* source, 
    std::vector<std::vector<int>>& vec_p2p_interactions, std::vector<std::vector<int>>& vec_m2l_interactions)
{
    TYPE dx = source->center[0] - target->center[0];
    TYPE dy = source->center[1] - target->center[1];
    TYPE dz = source->center[2] - target->center[2];
    TYPE r2 = dx*dx + dy*dy + dz*dz;
    TYPE d1 = source->rad*2.0;
    TYPE d2 = target->rad*2.0;

    if ((d1+d2)*(d1+d2) < params->theta2*r2)
    {
        //m2l(params, target, source);
        vec_m2l_interactions[target->offset].push_back(source->offset);

    }
    else if (is_leaf(source) && is_leaf(target))
    {
        //p2p(params, target, source);
        vec_p2p_interactions[target->offset].push_back(source->offset);
    }
    else
    {
        TYPE target_sz = target->rad;
        TYPE source_sz = source->rad;

        if (is_leaf(source) || (target_sz >= source_sz && !is_leaf(target))) 
        {
            for (size_t i = 0; i < target->num_children; ++i)
                dual_tree_traversal_core(params, get_node(params, target->child[i]), source, vec_p2p_interactions, vec_m2l_interactions);
        }
        else
        {
             for (size_t i = 0; i < source->num_children; ++i)
                //dual_tree_traversal_core(params, target, source->child[i]);
                dual_tree_traversal_core(params, target, get_node(params, source->child[i]), vec_p2p_interactions, vec_m2l_interactions);
        }
    }
}

__global__
void direct_gpu(
    TYPE* d_x, TYPE* d_y, TYPE* d_z, TYPE* d_w,
    TYPE* d_ax, TYPE* d_ay, TYPE* d_az, TYPE* d_p, int n)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= n) return;

    TYPE xi = d_x[i];
    TYPE yi = d_y[i];
    TYPE zi = d_z[i];
    TYPE ax = 0.0f;
    TYPE ay = 0.0f;
    TYPE az = 0.0f;
    TYPE p = 0.0f;
    for (int j = 0; j < n; ++j)
    {
        TYPE dx = d_x[j] - xi;
        TYPE dy = d_y[j] - yi;
        TYPE dz = d_z[j] - zi;
        TYPE wj = d_w[j];
        TYPE r = dx*dx + dy*dy + dz*dz;
        TYPE inv_r = (r == 0.0f) ? 0.0f : rsqrtf(r);
        TYPE inv_r_3 = inv_r*inv_r*inv_r;
        ax += dx*wj*inv_r_3;
        ay += dy*wj*inv_r_3;
        az += dz*wj*inv_r_3;
        p += wj*inv_r;
    }
    d_ax[i] += ax;
    d_ay[i] += ay;
    d_az[i] += az;
    d_p[i] += p;
}

__global__
void gpu_p2p_interactions(
    t_node* d_nodes,
    TYPE* d_x, TYPE* d_y, TYPE* d_z, TYPE* d_w,
    TYPE* d_ax, TYPE* d_ay, TYPE* d_az, TYPE* d_p, 
    int* d_p2p_interactions, int* d_p2p_sizes, int* d_p2p_offsets,
    size_t num_nodes)
{
    long wid = threadIdx.x/32;
    long lane = threadIdx.x % 32;
    const long num_warps = 256/32;

    long target_offset = blockIdx.x*(blockDim.x/32)+wid;

    if (target_offset >= num_nodes) return;

    t_node* target = &d_nodes[target_offset];

    size_t tidx = target->point_idx;
    const TYPE*  tx    = &d_x[tidx];
    const TYPE*  ty    = &d_y[tidx];
    const TYPE*  tz    = &d_z[tidx];
    TYPE*  tax   = &d_ax[tidx];
    TYPE*  tay   = &d_ay[tidx];
    TYPE*  taz   = &d_az[tidx];
    TYPE*  tp    = &d_p[tidx];

    const long shmem_sz = 256;
    __shared__ float4 shmem_base[shmem_sz*num_warps];
    float4* shmem = &shmem_base[wid*shmem_sz];

    const int interaction_offset = d_p2p_offsets[target_offset];
    const int* interaction_list = &d_p2p_interactions[interaction_offset];
    const int num_interacts = d_p2p_sizes[target_offset];

    const long num_target_points = target->num_points;

    for (size_t ii = lane; ii < target->num_points+31; ii += 32)
    { 
        TYPE ax = 0.0, ay = 0.0, az = 0.0, p = 0.0;
        const TYPE xi = (ii >= num_target_points) ? 0.0f : tx[ii];
        const TYPE yi = (ii >= num_target_points) ? 0.0f : ty[ii];
        const TYPE zi = (ii >= num_target_points) ? 0.0f : tz[ii];

        for (size_t j = 0; j < num_interacts; ++j)
        {
            int source_offset = interaction_list[j];
            t_node* source = &d_nodes[source_offset];

            size_t sidx = source->point_idx;
            const TYPE* sx = &d_x[sidx];
            const TYPE* sy = &d_y[sidx];
            const TYPE* sz = &d_z[sidx];
            const TYPE* sw = &d_w[sidx];

            const size_t num_source_points = source->num_points;

            for (size_t jb = 0; jb < num_source_points; jb += shmem_sz)
            {
                #pragma unroll 32
                for (size_t jj = lane; jj < shmem_sz; jj += 32)
                {
                    if (jj+jb >= num_source_points) break;
                    shmem[jj].x = sx[jj+jb];
                    shmem[jj].y = sy[jj+jb];
                    shmem[jj].z = sz[jj+jb];
                    shmem[jj].w = sw[jj+jb];
                }

                #pragma unroll 32
                for (size_t jj = 0; jj < shmem_sz; ++jj)
                {
                    if (jj+jb >= num_source_points) break;
                    TYPE dx = shmem[jj].x - xi;
                    TYPE dy = shmem[jj].y - yi;
                    TYPE dz = shmem[jj].z - zi;
                    TYPE wj = shmem[jj].w;
                    TYPE r = dx*dx + dy*dy + dz*dz;
                    TYPE inv_r = (r == 0.0f) ? 0.0f : rsqrtf(r);
                    TYPE inv_r_3 = inv_r*inv_r*inv_r;
                    ax += dx*wj*inv_r_3;
                    ay += dy*wj*inv_r_3;
                    az += dz*wj*inv_r_3;
                    p += wj*inv_r;
                }
            }

        }
        if (ii >= num_target_points) break;
        tax[ii] += ax;
        tay[ii] += ay;
        taz[ii] += az;
        tp[ii]  += p;
    }
}

#define S_IDX(n,m) ((n)*(n)+(n)+(m))

__device__
void cart_to_sph(TYPE x, TYPE y, TYPE z, TYPE* pr, TYPE* ptheta, TYPE* pphi)
{
    *pr = TYPE_SQRT(x*x+y*y+z*z);
    *ptheta = (*pr == TYPE_ZERO) ? TYPE_ZERO : TYPE_ACOS(z/(*pr));
    *pphi = TYPE_ATAN2(y, x);
}

__global__
void gpu_m2l_interactions(
    t_node* d_nodes,
    TYPE* d_m_real, TYPE* d_m_imag, TYPE* d_l_real, TYPE* d_l_imag,
    int* d_m2l_interactions, int* d_m2l_sizes, int* d_m2l_offsets, 
    size_t num_nodes)
{
    long wid = threadIdx.x/32;
    long lane = threadIdx.x % 32;
    const long num_warps = 256/32;

    long target_offset = blockIdx.x*(blockDim.x/32)+wid;

    if (target_offset >= num_nodes) return;

    t_node* target = &d_nodes[target_offset];

    size_t tidx = target->mult_idx;
    TYPE* L_real = &d_l_real[tidx];
    TYPE* L_imag = &d_l_imag[tidx];

    const int num_terms = 4;

    __shared__ TYPE outer_real[num_warps*num_terms*num_terms];
    __shared__ TYPE outer_imag[num_warps*num_terms*num_terms];

    const int num_ints = d_m2l_sizes[target_offset];
    const int interact_offset = d_m2l_offsets[target_offset];
    const int* interact_list = &d_m2l_interactions[interact_offset];

    size_t woff = wid*num_terms*num_terms;
    
    //if (lane == 0) printf("target %ld has %ld interactions\n", target_offset, num_ints);
    if (lane != 0) return;
    //if (lane == 0)
    for (size_t i = 0; i < num_ints; ++i)
    {
        size_t source_offset = interact_list[i];
        t_node* source = &d_nodes[source_offset];

        TYPE dx = target->center[0] - source->center[0];
        TYPE dy = target->center[1] - source->center[1];
        TYPE dz = target->center[2] - source->center[2];

        size_t sidx = source->mult_idx;
        
        TYPE* M_real = &d_m_real[sidx];
        TYPE* M_imag = &d_m_imag[sidx];

        TYPE rho, alpha, beta;
        cart_to_sph(dx, dy, dz, &rho, &alpha, &beta);
        compute_outer_gpu(num_terms, rho, alpha, beta, 
            &outer_real[wid*num_terms*num_terms], &outer_imag[wid*num_terms*num_terms]);

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
                        tmp_real += M_real[S_IDX(n,m)]*outer_real[woff+S_IDX(j+n,-k-m)] - 
                            M_imag[S_IDX(n,m)]*outer_imag[woff+S_IDX(j+n,-k-m)];
                        tmp_imag += M_real[S_IDX(n,m)]*outer_imag[woff+S_IDX(j+n,-k-m)] + 
                            M_imag[S_IDX(n,m)]*outer_real[woff+S_IDX(j+n,-k-m)];
                    }
                }
                L_real[S_IDX(j,k)] += tmp_real;
                L_imag[S_IDX(j,k)] += tmp_imag;
            }
        }
    }
}
void dual_tree_traversal(t_fmm_params* params)
{   
    t_node* d_nodes;
    TYPE* d_x;
    TYPE* d_y;
    TYPE* d_z;
    TYPE* d_w;
    TYPE* d_m_real;
    TYPE* d_m_imag;
    TYPE* d_l_real;
    TYPE* d_l_imag;
    TYPE* d_ax;
    TYPE* d_ay;
    TYPE* d_az;
    TYPE* d_p;
    
    size_t np = params->num_points;
    CUDACHK(cudaMalloc((void**)&d_x,    sizeof(TYPE)*np));
    CUDACHK(cudaMalloc((void**)&d_y,    sizeof(TYPE)*np));
    CUDACHK(cudaMalloc((void**)&d_z,    sizeof(TYPE)*np));
    CUDACHK(cudaMalloc((void**)&d_w,    sizeof(TYPE)*np));
    CUDACHK(cudaMalloc((void**)&d_ax,   sizeof(TYPE)*np));
    CUDACHK(cudaMalloc((void**)&d_ay,   sizeof(TYPE)*np));
    CUDACHK(cudaMalloc((void**)&d_az,   sizeof(TYPE)*np));
    CUDACHK(cudaMalloc((void**)&d_p,    sizeof(TYPE)*np));

    size_t nm = params->num_multipoles*params->num_nodes;
    CUDACHK(cudaMalloc((void**)&d_m_real,   sizeof(TYPE)*nm));
    CUDACHK(cudaMalloc((void**)&d_m_imag,   sizeof(TYPE)*nm));
    CUDACHK(cudaMalloc((void**)&d_l_real,   sizeof(TYPE)*nm));
    CUDACHK(cudaMalloc((void**)&d_l_imag,   sizeof(TYPE)*nm));

    CUDACHK(cudaMalloc((void**)&d_nodes, sizeof(t_node)*params->num_nodes));

    CUDACHK(cudaMemcpy(d_x,     &params->x[0],  sizeof(TYPE)*np, cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(d_y,     &params->y[0],  sizeof(TYPE)*np, cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(d_z,     &params->z[0],  sizeof(TYPE)*np, cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(d_w,     &params->w[0],  sizeof(TYPE)*np, cudaMemcpyHostToDevice));

    CUDACHK(cudaMemcpy(d_ax,    &params->ax[0], sizeof(TYPE)*np, cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(d_ay,    &params->ay[0], sizeof(TYPE)*np, cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(d_az,    &params->az[0], sizeof(TYPE)*np, cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(d_p,     &params->p[0],  sizeof(TYPE)*np, cudaMemcpyHostToDevice));

    CUDACHK(cudaMemcpy(d_l_real, &params->L_array_real[0], sizeof(TYPE)*nm, cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(d_l_imag, &params->L_array_imag[0], sizeof(TYPE)*nm, cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(d_m_real, &params->M_array_real[0], sizeof(TYPE)*nm, cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(d_m_imag, &params->M_array_imag[0], sizeof(TYPE)*nm, cudaMemcpyHostToDevice));

    CUDACHK(cudaMemcpy(d_nodes, &params->node_array[0], sizeof(t_node)*params->num_nodes, cudaMemcpyHostToDevice));

    init_spharm_gpu(params);

    //size_t* p2p_interactions = (size_t*)malloc(sizeof(size_t)*params->num_nodes*params->num_nodes);
    //size_t* num_p2p_interactions = (size_t*)malloc(sizeof(size_t)*params->num_nodes);
    //size_t* m2l_interactions = (size_t*)malloc(sizeof(size_t)*params->num_nodes*params->num_nodes);
    //size_t* num_m2l_interactions = (size_t*)malloc(sizeof(size_t)*params->num_nodes);

    //size_t* d_p2p_interactions;
    //size_t* d_num_p2p_interactions;
    //size_t* d_m2l_interactions;
    //size_t* d_num_m2l_interactions;

    //CUDACHK(cudaMalloc((void**)&d_p2p_interactions, sizeof(size_t)*nn*nn));
    //CUDACHK(cudaMalloc((void**)&d_num_p2p_interactions, sizeof(size_t)*nn));
    //CUDACHK(cudaMalloc((void**)&d_m2l_interactions, sizeof(size_t)*nn*nn));
    //CUDACHK(cudaMalloc((void**)&d_num_m2l_interactions, sizeof(size_t)*nn));
    
    size_t nn = params->num_nodes;

    std::vector<std::vector<int>> vec_p2p_interactions(nn, std::vector<int>());
    std::vector<std::vector<int>> vec_m2l_interactions(nn, std::vector<int>());

    t_timer dtt_timer;
    start(&dtt_timer);
    dual_tree_traversal_core(params, get_node(params, params->root), get_node(params, params->root), vec_p2p_interactions, vec_m2l_interactions);
    stop(&dtt_timer);

    t_timer transfer_timer;
    start(&transfer_timer);

    int* h_p2p_sizes = (int*)malloc(sizeof(int)*nn);
    int* h_m2l_sizes = (int*)malloc(sizeof(int)*nn);
    int* h_p2p_offsets = (int*)malloc(sizeof(int)*nn);
    int* h_m2l_offsets = (int*)malloc(sizeof(int)*nn);

    size_t tot_p2p = 0, tot_m2l = 0;
    for (int i = 0; i < params->num_nodes; ++i)
    {
        int num_p2p = vec_p2p_interactions[i].size(); 
        int num_m2l = vec_m2l_interactions[i].size(); 
        h_p2p_offsets[i] = tot_p2p;
        h_m2l_offsets[i] = tot_m2l;
        tot_p2p += num_p2p;
        tot_m2l += num_m2l;
    }
    int* h_p2p_interactions = (int*)malloc(sizeof(int)*tot_p2p);
    int* h_m2l_interactions = (int*)malloc(sizeof(int)*tot_m2l);

    for (int i = 0; i < params->num_nodes; ++i)
    {
        int num_p2p = vec_p2p_interactions[i].size(); 
        int num_m2l = vec_m2l_interactions[i].size(); 
        int p2p_offset = h_p2p_offsets[i];
        int m2l_offset = h_m2l_offsets[i];
        h_p2p_sizes[i] = num_p2p;
        h_m2l_sizes[i] = num_m2l;
        for (int j = 0; j < num_p2p; ++j)
            h_p2p_interactions[p2p_offset+j] = vec_p2p_interactions[i][j];
        for (int j = 0; j < num_m2l; ++j)
            h_m2l_interactions[m2l_offset+j] = vec_m2l_interactions[i][j];
    }


    int* d_p2p_sizes;
    int* d_m2l_sizes;
    int* d_p2p_offsets;
    int* d_m2l_offsets;
    int* d_p2p_interactions;
    int* d_m2l_interactions;

    CUDACHK(cudaMalloc((void**)&d_p2p_sizes,        sizeof(int)*nn));
    CUDACHK(cudaMalloc((void**)&d_m2l_sizes,        sizeof(int)*nn));
    CUDACHK(cudaMalloc((void**)&d_p2p_offsets,      sizeof(int)*nn));
    CUDACHK(cudaMalloc((void**)&d_m2l_offsets,      sizeof(int)*nn));
    CUDACHK(cudaMalloc((void**)&d_p2p_interactions, sizeof(int)*tot_p2p));
    CUDACHK(cudaMalloc((void**)&d_m2l_interactions, sizeof(int)*tot_m2l));

    CUDACHK(cudaMemcpy(d_p2p_sizes,         h_p2p_sizes,         sizeof(int)*nn,       cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(d_m2l_sizes,         h_m2l_sizes,         sizeof(int)*nn,       cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(d_p2p_offsets,       h_p2p_offsets,       sizeof(int)*nn,       cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(d_m2l_offsets,       h_m2l_offsets,       sizeof(int)*nn,       cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(d_p2p_interactions,  h_p2p_interactions,  sizeof(int)*tot_p2p,  cudaMemcpyHostToDevice));
    CUDACHK(cudaMemcpy(d_m2l_interactions,  h_m2l_interactions,  sizeof(int)*tot_m2l,  cudaMemcpyHostToDevice));

    stop(&transfer_timer);

    //direct_gpu<<<num_blocks, block_sz>>>(d_x, d_y, d_z, d_w, d_ax, d_ay, d_az, d_p, np);
    
    //CUDACHK(cudaMemcpy(d_p2p_interactions, p2p_interactions, sizeof(size_t)*nn*nn, cudaMemcpyHostToDevice));
    //CUDACHK(cudaMemcpy(d_num_p2p_interactions, num_p2p_interactions, sizeof(size_t)*nn, cudaMemcpyHostToDevice));
    //CUDACHK(cudaMemcpy(d_m2l_interactions, m2l_interactions, sizeof(size_t)*nn*nn, cudaMemcpyHostToDevice));
    //CUDACHK(cudaMemcpy(d_num_m2l_interactions, num_m2l_interactions, sizeof(size_t)*nn, cudaMemcpyHostToDevice));

    const int block_sz = 256;
    const int num_blocks = nn/4 + ((nn % 4) ? 1 : 0);

    t_timer timer;
    start(&timer);

    gpu_p2p_interactions<<<num_blocks, block_sz>>>(d_nodes, d_x, d_y, d_z, d_w, d_ax, d_ay, d_az, d_p, d_p2p_interactions, d_p2p_sizes, d_p2p_offsets, nn);
    gpu_m2l_interactions<<<num_blocks, block_sz>>>(d_nodes, d_m_real, d_m_imag, d_l_real, d_l_imag, d_m2l_interactions, d_m2l_sizes, d_m2l_offsets, nn);

    CUDACHK(cudaPeekAtLastError());
    CUDACHK(cudaDeviceSynchronize());

    stop(&timer);
    printf("tot p2p = %zu, tot_m2l = %zu\n", tot_p2p, tot_m2l);
    printf("total memory to allocate to GPU = %zu MB\n", (sizeof(int)*(size_t)(4*nn + tot_p2p + tot_m2l))/1024/1024);
    printf("----------\n");
    printf("GPU elapsed time = %f\n", timer.elapsed);
    printf("DTT elapsed time = %f\n", dtt_timer.elapsed);
    printf("MEM elapsed time = %f\n", transfer_timer.elapsed);
    printf("Total elapsed time = %f\n", timer.elapsed + dtt_timer.elapsed + transfer_timer.elapsed);
    printf("----------\n");

    CUDACHK(cudaMemcpy(&params->ax[0], d_ax, sizeof(TYPE)*np, cudaMemcpyDeviceToHost));
    CUDACHK(cudaMemcpy(&params->ay[0], d_ay, sizeof(TYPE)*np, cudaMemcpyDeviceToHost));
    CUDACHK(cudaMemcpy(&params->az[0], d_az, sizeof(TYPE)*np, cudaMemcpyDeviceToHost));
    CUDACHK(cudaMemcpy(&params->p[0] , d_p,  sizeof(TYPE)*np, cudaMemcpyDeviceToHost));

    TYPE* l_real = (TYPE*)malloc(sizeof(TYPE)*nm);
    TYPE* l_imag = (TYPE*)malloc(sizeof(TYPE)*nm);

    CUDACHK(cudaMemcpy(l_real, d_l_real, sizeof(TYPE)*nm, cudaMemcpyDeviceToHost));
    CUDACHK(cudaMemcpy(l_imag, d_l_imag, sizeof(TYPE)*nm, cudaMemcpyDeviceToHost));

    CUDACHK(cudaMemcpy(&params->L_array_real[0], d_l_real, sizeof(TYPE)*nm, cudaMemcpyDeviceToHost));
    CUDACHK(cudaMemcpy(&params->L_array_imag[0], d_l_imag, sizeof(TYPE)*nm, cudaMemcpyDeviceToHost));
}

