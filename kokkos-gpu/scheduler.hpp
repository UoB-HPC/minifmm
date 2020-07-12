#include <algorithm>
#include <Kokkos_Macros.hpp>
#if defined (KOKKOS_ENABLE_TASKDAG)
#include <Kokkos_Core.hpp>

#include "tree.h"
#include "spharm_gpu.h"
#include "kernels_gpu.h"

#define S_IDX(n,m) ((n)*(n)+(n)+(m))

typedef Kokkos::View<TYPE*> t_view;

__device__
void serial_dtt(size_t target_offset, size_t source_offset,
        long tid, long tsz, Kokkos::View<t_node*>& d_nodes,
        t_view& d_x, t_view& d_y, t_view& d_z, t_view& d_w,
        t_view& d_ax, t_view& d_ay, t_view& d_az, t_view& d_p,
        t_view& d_m_real, t_view& d_m_imag, t_view& d_l_real, t_view& d_l_imag,
        Kokkos::TaskScheduler<Kokkos::Cuda>::member_type& member)
    
{
    t_node* target = &d_nodes[target_offset];
    t_node* source = &d_nodes[source_offset];

    TYPE dx = source->center[0] - target->center[0];
    TYPE dy = source->center[1] - target->center[1];
    TYPE dz = source->center[2] - target->center[2];
    TYPE r2 = dx*dx + dy*dy + dz*dz;
    TYPE d1 = source->rad*2.0;
    TYPE d2 = target->rad*2.0;

    if ((d1+d2)*(d1+d2) < (0.5*0.5)*r2)
    {
        //m2l_kokkos(target, source);
        m2l_kokkos_team(target, source, tid, tsz, d_m_real, d_m_imag, d_l_real, d_l_imag);
    }
    else if (is_leaf(source) && is_leaf(target))
    {
        if (source_offset == target_offset) p2p_kokkos_team(target, tid, tsz, d_x, d_y, d_z, d_w,
                d_ax, d_ay, d_az, d_p);
        else p2p_kokkos_team(target, source, member, d_x, d_y, d_z, d_w, d_ax, d_ay, d_az, d_p);
    }
    else
    {
        TYPE target_sz = target->rad;
        TYPE source_sz = source->rad;

        if (is_leaf(source) || (target_sz >= source_sz && !is_leaf(target))) 
        {
            for (size_t i = 0; i < target->num_children; ++i)
            {
                serial_dtt(target->child[i], source_offset, tid, tsz, d_nodes,
                        d_x, d_y, d_z, d_w, d_ax, d_ay, d_az, d_p,
                        d_m_real, d_m_imag, d_l_real, d_l_imag, member);
            }
        }
        else
        {
            for (size_t i = 0; i < source->num_children; ++i)
            {
                serial_dtt(target_offset, source->child[i], tid, tsz, d_nodes,
                        d_x, d_y, d_z, d_w, d_ax, d_ay, d_az, d_p,
                        d_m_real, d_m_imag, d_l_real, d_l_imag, member);
            }
        }
    }
}

template<typename Space>
struct DTT
{
    typedef Kokkos::TaskScheduler<Space>    sched_type;
    typedef Kokkos::Future<Space, void>     future_type;
    typedef void value_type;
    
    sched_type sched;
    future_type f[8];
    size_t target_offset;
    size_t source_offset;

    Kokkos::View<t_node*> d_nodes;
    Kokkos::View<TYPE*> d_x;
    Kokkos::View<TYPE*> d_y;
    Kokkos::View<TYPE*> d_z;
    Kokkos::View<TYPE*> d_w;
    Kokkos::View<TYPE*> d_ax;
    Kokkos::View<TYPE*> d_ay;
    Kokkos::View<TYPE*> d_az;
    Kokkos::View<TYPE*> d_p;
    Kokkos::View<TYPE*> d_m_real;
    Kokkos::View<TYPE*> d_m_imag;
    Kokkos::View<TYPE*> d_l_real;
    Kokkos::View<TYPE*> d_l_imag;

    const TYPE theta = 0.5;
    const TYPE theta2 = theta*theta;


    KOKKOS_INLINE_FUNCTION
    DTT(const sched_type& arg_sched, size_t arg_target, size_t arg_source, 
        Kokkos::View<t_node*>& arg_d_nodes, 
        t_view& arg_d_x, t_view& arg_d_y, t_view& arg_d_z, t_view& arg_d_w,
        t_view& arg_d_ax, t_view& arg_d_ay, t_view& arg_d_az, t_view& arg_d_p,
        t_view& arg_d_m_real, t_view& arg_d_m_imag, t_view& arg_d_l_real, t_view& arg_d_l_imag) :
            sched(arg_sched), target_offset(arg_target), source_offset(arg_source), d_nodes(arg_d_nodes),
            d_x(arg_d_x), d_y(arg_d_y), d_z(arg_d_z), d_w(arg_d_w), 
            d_ax(arg_d_ax), d_ay (arg_d_ay), d_az(arg_d_az), d_p(arg_d_p),
            d_m_real(arg_d_m_real), d_m_imag(arg_d_m_imag), d_l_real(arg_d_l_real), d_l_imag(arg_d_l_imag)
            {
                for (int i = 0; i < 8; ++i) f[i] = Kokkos::Future<Space>();
            } 

    KOKKOS_INLINE_FUNCTION
    void operator()(typename sched_type::member_type & member)
    {
        t_node* target = &d_nodes[target_offset];
        t_node* source = &d_nodes[source_offset];

        TYPE dx = source->center[0] - target->center[0];
        TYPE dy = source->center[1] - target->center[1];
        TYPE dz = source->center[2] - target->center[2];
        TYPE r2 = dx*dx + dy*dy + dz*dz;
        TYPE d1 = source->rad*2.0;
        TYPE d2 = target->rad*2.0;

        long tid = member.team_rank();
        long tsz = member.team_size();

        if ((d1+d2)*(d1+d2) < theta2*r2)
        {
            //m2l_kokkos(target, source);
            #ifdef __CUDA_ARCH__
            m2l_kokkos_team(target, source, tid, tsz, d_m_real, d_m_imag, d_l_real, d_l_imag);
            #endif
        }
        else if (is_leaf(source) && is_leaf(target))
        {
            #ifdef __CUDA_ARCH__
            if (source_offset == target_offset) p2p_kokkos_team(target, tid, tsz, d_x, d_y, d_z, d_w,
                d_ax, d_ay, d_az, d_p);
            else p2p_kokkos_team(target, source, member, d_x, d_y, d_z, d_w, d_ax, d_ay, d_az, d_p);
            #endif
        }
        else
        {
            TYPE target_sz = target->rad;
            TYPE source_sz = source->rad;

            if (is_leaf(source) || (target_sz >= source_sz && !is_leaf(target))) 
            {
                if (target->num_points > 10000)
                {
                    for (size_t i = 0; i < target->num_children; ++i)
                    {
                        if (member.team_rank() == 0)
                        {
                            f[i] = Kokkos::task_spawn(Kokkos::TaskTeam(sched), 
                                    DTT(sched, target->child[i], source_offset, d_nodes,
                                        d_x, d_y, d_z, d_w, d_ax, d_ay, d_az, d_p,
                                        d_m_real, d_m_imag, d_l_real, d_l_imag));
                            if (f[i].is_null())
                            {
                                printf("failed to allocate task\n");
                            }
                        }
                    }
                }
                else
                {
                    for (size_t i = 0; i < target->num_children; ++i)
                    {
                        #ifdef __CUDA_ARCH__
                        serial_dtt(target->child[i], source_offset, tid, tsz, d_nodes,
                                d_x, d_y, d_z, d_w, d_ax, d_ay, d_az, d_p,
                                d_m_real, d_m_imag, d_l_real, d_l_imag, member);
                        #endif
                    }
                }
                //Kokkos::respawn(this, Kokkos::when_all(f, target->num_children));
            }
            else
            {
                if (source->num_points >10000)
                {
                    for (size_t i = 0; i < source->num_children; ++i)
                    {
                        if (member.team_rank() == 0)
                        {
                            f[i] = Kokkos::task_spawn(Kokkos::TaskTeam(sched), 
                                    DTT(sched, target_offset, source->child[i], d_nodes,
                                        d_x, d_y, d_z, d_w, d_ax, d_ay, d_az, d_p,
                                        d_m_real, d_m_imag, d_l_real, d_l_imag));
                            if (f[i].is_null())
                            {
                                printf("failed to allocate task\n");
                            }
                        }
                    }
                }
                else
                {
                    for (size_t i = 0; i < source->num_children; ++i)
                    {
                        #ifdef __CUDA_ARCH__
                        serial_dtt(target_offset, source->child[i], tid, tsz, d_nodes,
                                d_x, d_y, d_z, d_w, d_ax, d_ay, d_az, d_p,
                                d_m_real, d_m_imag, d_l_real, d_l_imag, member);
                        #endif
                    }
                }
                //Kokkos::respawn(this, Kokkos::when_all(f, source->num_children));
            }
        }
    }

    static void run(size_t root, Kokkos::View<t_node*>& d_nodes,
        t_view& d_x, t_view& d_y, t_view& d_z, t_view& d_w,
        t_view& d_ax, t_view& d_ay, t_view& d_az, t_view& d_p,
        t_view& d_m_real, t_view& d_m_imag, t_view& d_l_real, t_view& d_l_imag)
    {
        typedef typename sched_type::memory_space memory_space;

        size_t MemoryCapacity = 16384000;

        enum { MinBlockSize   =   64 };
        enum { MaxBlockSize   = 1024 };
        enum { SuperBlockSize = 4096 };

        sched_type root_sched( memory_space()
                , MemoryCapacity
                , MinBlockSize
                , std::min(size_t(MaxBlockSize),MemoryCapacity)
                , std::min(size_t(SuperBlockSize),MemoryCapacity) );

        future_type f = Kokkos::host_spawn(Kokkos::TaskTeam(root_sched),
            DTT(root_sched, root, root, d_nodes, d_x, d_y, d_z, d_w, d_ax, d_ay, d_az, d_p,
                d_m_real, d_m_imag, d_l_real, d_l_imag));

        Kokkos::wait(root_sched);
    }
};
#else
#warning task dag not enabled
#endif
