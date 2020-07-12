#include <stdlib.h>
#include <Kokkos_Core.hpp>

#include "type.h"
#include "tree.h"
#include "kernels.h"
#include "timer.h"
#include "dtt.h"

#include "spharm_gpu.h"

#include "scheduler.hpp"

void dual_tree_traversal_core(t_fmm_params* params, t_node* target, t_node* source)
{
    TYPE dx = source->center[0] - target->center[0];
    TYPE dy = source->center[1] - target->center[1];
    TYPE dz = source->center[2] - target->center[2];
    TYPE r2 = dx*dx + dy*dy + dz*dz;
    TYPE d1 = source->rad*2.0;
    TYPE d2 = target->rad*2.0;

    if ((d1+d2)*(d1+d2) < params->theta2*r2)
    {
        m2l(params, target, source);
    }
    else if (is_leaf(source) && is_leaf(target))
    {
        //p2p(params, target, source);
    }
    else
    {
        TYPE target_sz = target->rad;
        TYPE source_sz = source->rad;

        if (is_leaf(source) || (target_sz >= source_sz && !is_leaf(target))) 
        {
            for (size_t i = 0; i < target->num_children; ++i)
                dual_tree_traversal_core(params, get_node(params, target->child[i]), source);
        }
        else
        {
             for (size_t i = 0; i < source->num_children; ++i)
                //dual_tree_traversal_core(params, target, source->child[i]);
                dual_tree_traversal_core(params, target, get_node(params, source->child[i]));
        }
    }
}

typedef Kokkos::View<TYPE*> t_view;
typedef Kokkos::View<TYPE*>::HostMirror t_hview;

void kokkos_alloc(t_fmm_params* params,
    Kokkos::View<t_node*>& d_nodes, 
    t_view& d_x, t_view& d_y, t_view& d_z, t_view& d_w,
    t_view& d_m_real, t_view& d_m_imag, t_view& d_l_real, t_view& d_l_imag,
    t_view& d_ax, t_view& d_ay, t_view& d_az, t_view& d_p,
    Kokkos::View<t_node*>::HostMirror& h_nodes, 
    t_hview& h_x, t_hview& h_y, t_hview& h_z, t_hview& h_w,
    t_hview& h_m_real, t_hview& h_m_imag, t_hview& h_l_real, t_hview& h_l_imag,
    t_hview& h_ax, t_hview& h_ay, t_hview& h_az, t_hview& h_p
    )
{
    size_t np = params->num_points;
    size_t nm = params->num_nodes * params->num_multipoles;
    d_nodes    = Kokkos::View<t_node*>("d_nodes", params->num_nodes);
    d_x        = Kokkos::View<TYPE*>("d_x", np);
    d_y        = Kokkos::View<TYPE*>("d_y", np);
    d_z        = Kokkos::View<TYPE*>("d_z", np);
    d_w        = Kokkos::View<TYPE*>("d_w", np);
    d_m_real   = Kokkos::View<TYPE*>("d_m_real", nm);
    d_m_imag   = Kokkos::View<TYPE*>("d_m_imag", nm);
    d_l_real   = Kokkos::View<TYPE*>("d_l_real", nm);
    d_l_imag   = Kokkos::View<TYPE*>("d_l_imag", nm);
    d_ax       = Kokkos::View<TYPE*>("d_ax", np);
    d_ay       = Kokkos::View<TYPE*>("d_ay", np);
    d_az       = Kokkos::View<TYPE*>("d_az", np);
    d_p        = Kokkos::View<TYPE*>("d_p", np);

    h_nodes    = Kokkos::View<t_node*>::HostMirror(create_mirror_view(d_nodes));
    h_x        = Kokkos::View<TYPE*>::HostMirror(create_mirror_view(d_x));
    h_y        = Kokkos::View<TYPE*>::HostMirror(create_mirror_view(d_y));
    h_z        = Kokkos::View<TYPE*>::HostMirror(create_mirror_view(d_z));
    h_w        = Kokkos::View<TYPE*>::HostMirror(create_mirror_view(d_w));
    h_m_real   = Kokkos::View<TYPE*>::HostMirror(create_mirror_view(d_m_real));
    h_m_imag   = Kokkos::View<TYPE*>::HostMirror(create_mirror_view(d_m_imag));
    h_l_real   = Kokkos::View<TYPE*>::HostMirror(create_mirror_view(d_l_real));
    h_l_imag   = Kokkos::View<TYPE*>::HostMirror(create_mirror_view(d_l_imag));
    h_ax       = Kokkos::View<TYPE*>::HostMirror(create_mirror_view(d_ax));
    h_ay       = Kokkos::View<TYPE*>::HostMirror(create_mirror_view(d_ay));
    h_az       = Kokkos::View<TYPE*>::HostMirror(create_mirror_view(d_az));
    h_p        = Kokkos::View<TYPE*>::HostMirror(create_mirror_view(d_p));

    for (size_t i = 0; i < np; ++i)
    {
        h_x[i] = params->x[i];
        h_y[i] = params->y[i];
        h_z[i] = params->z[i];
        h_w[i] = params->w[i];
    }

    for (size_t i = 0; i < params->num_nodes; ++i)
    {
        h_nodes[i] = params->node_array[i];
    }

    for (size_t i = 0; i < nm; ++i)
    {
        h_m_real[i] = params->M_array_real[i];
        h_m_imag[i] = params->M_array_imag[i];
    }

    Kokkos::deep_copy(d_x, h_x);
    Kokkos::deep_copy(d_y, h_y);
    Kokkos::deep_copy(d_z, h_z);
    Kokkos::deep_copy(d_w, h_w);

    Kokkos::deep_copy(d_m_real, h_m_real);
    Kokkos::deep_copy(d_m_imag, h_m_imag);

    Kokkos::deep_copy(d_nodes, h_nodes);

    Kokkos::parallel_for(np, KOKKOS_LAMBDA(const long i)
    {
        d_ax[i] = 0.0;
        d_ay[i] = 0.0;
        d_az[i] = 0.0;
        d_p[i] = 0.0;
    });

    Kokkos::fence();

    Kokkos::parallel_for(nm, KOKKOS_LAMBDA(const long i)
    {
        d_l_real[i] = 0.0;
        d_l_imag[i] = 0.0;
    });

    Kokkos::fence();
}

void kokkos_finish(t_fmm_params* params,
    t_view& d_ax, t_view& d_ay, t_view& d_az, t_view& d_p,
    t_view& d_l_real, t_view& d_l_imag,
    t_hview& h_ax, t_hview& h_ay, t_hview& h_az, t_hview& h_p,
    t_hview& h_l_real, t_hview& h_l_imag)
{
    Kokkos::deep_copy(h_ax, d_ax);
    Kokkos::deep_copy(h_ay, d_ay);
    Kokkos::deep_copy(h_az, d_az);
    Kokkos::deep_copy(h_p, d_p);

    Kokkos::deep_copy(h_l_real, d_l_real);
    Kokkos::deep_copy(h_l_imag, d_l_imag);

    size_t np = params->num_points;
    for (size_t i = 0; i < np; ++i)
    {
        params->ax[i] = h_ax[i];
        params->ay[i] = h_ay[i];
        params->az[i] = h_az[i];
        params->p[i] = h_p[i];
    }

    size_t nm = params->num_nodes * params->num_multipoles;
    printf("nm = %zu\n", nm);
    for (size_t i = 0; i < nm; ++i)
    {
        params->L_array_real[i] = h_l_real[i];
        params->L_array_imag[i] = h_l_imag[i];
        //printf("L real val %d = %f\n", i, params->L_array_real[i]);
    }
}

void dual_tree_traversal(t_fmm_params* params)
{   

    Kokkos::View<t_node*> d_nodes;
    Kokkos::View<TYPE*> d_x;
    Kokkos::View<TYPE*> d_y;
    Kokkos::View<TYPE*> d_z;
    Kokkos::View<TYPE*> d_w;
    Kokkos::View<TYPE*> d_m_real;
    Kokkos::View<TYPE*> d_m_imag;
    Kokkos::View<TYPE*> d_l_real;
    Kokkos::View<TYPE*> d_l_imag;
    Kokkos::View<TYPE*> d_ax;
    Kokkos::View<TYPE*> d_ay;
    Kokkos::View<TYPE*> d_az;
    Kokkos::View<TYPE*> d_p;

    Kokkos::View<t_node*>::HostMirror h_nodes;
    Kokkos::View<TYPE*>::HostMirror h_x;
    Kokkos::View<TYPE*>::HostMirror h_y;
    Kokkos::View<TYPE*>::HostMirror h_z;
    Kokkos::View<TYPE*>::HostMirror h_w;
    Kokkos::View<TYPE*>::HostMirror h_m_real;
    Kokkos::View<TYPE*>::HostMirror h_m_imag;
    Kokkos::View<TYPE*>::HostMirror h_l_real;
    Kokkos::View<TYPE*>::HostMirror h_l_imag;
    Kokkos::View<TYPE*>::HostMirror h_ax;
    Kokkos::View<TYPE*>::HostMirror h_ay;
    Kokkos::View<TYPE*>::HostMirror h_az;
    Kokkos::View<TYPE*>::HostMirror h_p;

    kokkos_alloc(params,
        d_nodes, d_x, d_y, d_z, d_w, d_m_real, d_m_imag, d_l_real, d_l_imag, d_ax, d_ay, d_az, d_p,
        h_nodes, h_x, h_y, h_z, h_w, h_m_real, h_m_imag, h_l_real, h_l_imag, h_ax, h_ay, h_az, h_p);

    //size_t np = params->num_points;
    //Kokkos::parallel_for(params->num_points, KOKKOS_LAMBDA(const long i)
    //{
    //    for (int j = 0; j < np; ++j)
    //    {
    //        TYPE dx = d_x[j] - d_x[i];
    //        TYPE dy = d_y[j] - d_y[i];
    //        TYPE dz = d_z[j] - d_z[i];
    //        TYPE r = dx*dx + dy*dy + dz*dz;
    //        TYPE inv_r = (r == 0.0) ? 0.0 : 1.0/sqrt(r);
    //        TYPE inv_r_3 = inv_r*inv_r*inv_r;
    //        d_ax[i] += dx*inv_r_3;
    //        d_ay[i] += dy*inv_r_3;
    //        d_az[i] += dz*inv_r_3;
    //        d_p[i] += inv_r;
    //    }
    //});

    cudaDeviceSetLimit(cudaLimitStackSize, 2048);

    init_spharm_gpu(params);
    
    t_timer timer;
    start(&timer);
    DTT<Kokkos::Cuda>::run(params->root, d_nodes, d_x, d_y, d_z, d_w, d_ax, d_ay, d_az, d_p, d_m_real, d_m_imag, d_l_real, d_l_imag);
    Kokkos::fence();
    stop(&timer);
    printf("GPU elapsed time = %f\n", timer.elapsed);

    kokkos_finish(params, 
        d_ax, d_ay, d_az, d_p, d_l_real, d_l_imag,
        h_ax, h_ay, h_az, h_p, h_l_real, h_l_imag);

    //dual_tree_traversal_core(params, get_node(params, params->root), get_node(params, params->root));
}

