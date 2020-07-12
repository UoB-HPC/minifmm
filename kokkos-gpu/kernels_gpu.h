#pragma once

#include "type.h"
#include <Kokkos_Core.hpp>

__noinline__   __device__
void m2l_kokkos_team(t_node* target, t_node* source, long tid, long tsz,
    Kokkos::View<TYPE*>& d_m_real, Kokkos::View<TYPE*>& d_m_imag, Kokkos::View<TYPE*>& d_l_real, Kokkos::View<TYPE*>& d_l_imag);
    
__noinline__   __device__
void p2p_kokkos_team(t_node* target, long tid, long tsz,
    Kokkos::View<TYPE*>& d_x, Kokkos::View<TYPE*>& d_y, Kokkos::View<TYPE*>& d_z, Kokkos::View<TYPE*>& d_w,
    Kokkos::View<TYPE*>& d_ax, Kokkos::View<TYPE*>& d_ay, Kokkos::View<TYPE*>& d_az, Kokkos::View<TYPE*>& d_p);

//__noinline__   __device__
//void p2p_kokkos_team(t_node* target, t_node* source, long tid, long tsz,
//    Kokkos::View<TYPE*>& d_x, Kokkos::View<TYPE*>& d_y, Kokkos::View<TYPE*>& d_z, Kokkos::View<TYPE*>& d_w,
//    Kokkos::View<TYPE*>& d_ax, Kokkos::View<TYPE*>& d_ay, Kokkos::View<TYPE*>& d_az, Kokkos::View<TYPE*>& d_p);

__noinline__  __device__
void p2p_kokkos_team(t_node* target, t_node* source, Kokkos::TaskScheduler<Kokkos::Cuda>::member_type& member,
    Kokkos::View<TYPE*>& d_x, Kokkos::View<TYPE*>& d_y, Kokkos::View<TYPE*>& d_z, Kokkos::View<TYPE*>& d_w,
    Kokkos::View<TYPE*>& d_ax, Kokkos::View<TYPE*>& d_ay, Kokkos::View<TYPE*>& d_az, Kokkos::View<TYPE*>& d_p);
