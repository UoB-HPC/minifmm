#include <stdlib.h>
#include <Kokkos_Core.hpp>

#include "type.h"
#include "tree.h"
#include "kernels.h"
#include "timer.h"
#include "dtt.h"

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
        p2p(params, target, source);
    }
    else
    {
        TYPE target_sz = target->rad;
        TYPE source_sz = source->rad;

        if (is_leaf(source) || (target_sz >= source_sz && !is_leaf(target))) 
        {
            for (size_t i = 0; i < target->num_children; ++i)
                dual_tree_traversal_core(params, target->child[i], source);
        }
        else
        {
             for (size_t i = 0; i < source->num_children; ++i)
                //dual_tree_traversal_core(params, target, source->child[i]);
                dual_tree_traversal_core(params, target, source->child[i]);
        }
    }
}

struct DTT
{
    typedef Kokkos::TaskScheduler<Kokkos::OpenMP>    sched_type;
    typedef Kokkos::Future<Kokkos::OpenMP, void>           future_type;
    typedef void value_type;
    
    sched_type sched;
    future_type f[8];
    t_node* target;
    t_node* source;
    t_fmm_params* params;
    int spawned = 0;

    KOKKOS_INLINE_FUNCTION
    DTT(const sched_type& arg_sched, t_node* arg_target, t_node* arg_source, 
        t_fmm_params* arg_params) :
            sched(arg_sched), target(arg_target), source(arg_source), 
            params(arg_params) 
            {
                for (int i = 0; i < 8; ++i) f[i] = Kokkos::Future<Kokkos::OpenMP>();
            } 

    KOKKOS_INLINE_FUNCTION
    void operator()(typename sched_type::member_type &)
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
            p2p(params, target, source);
        }
        else
        {
            TYPE target_sz = target->rad;
            TYPE source_sz = source->rad;

            if (is_leaf(source) || (target_sz >= source_sz && !is_leaf(target))) 
            {
                for (size_t i = 0; i < target->num_children; ++i)
                {
                    //dual_tree_traversal_core(params, target->child[i], source);
                    if (target->num_points > 10000)
                    {
                        f[i] = Kokkos::task_spawn(Kokkos::TaskSingle(sched), 
                                DTT(sched, target->child[i], source, params));
                        if (f[i].is_null())
                        {
                            printf("failed to allocate task\n");
                            exit(1);
                        };
                    }
                    else
                    {
                        dual_tree_traversal_core(params, target->child[i], source);
                    }
                }
                //Kokkos::respawn(this, Kokkos::when_all(f, target->num_children));
            }
            else
            {
                for (size_t i = 0; i < source->num_children; ++i)
                {
                    //dual_tree_traversal_core(params, target, source->child[i]);
                    if (target->num_points > 10000)
                    {
                        f[i] = Kokkos::task_spawn(Kokkos::TaskSingle(sched), 
                                DTT(sched, target, source->child[i], params));
                        if (f[i].is_null())
                        {
                            printf("failed to allocate task\n");
                            exit(1);
                        }
                    }
                    else
                    {
                        dual_tree_traversal_core(params, target, source->child[i]);
                    }
                }
                //Kokkos::respawn(this, Kokkos::when_all(f, source->num_children));
            }
        }
    }

    static void run(t_fmm_params* params, t_node* root)
    {
        typedef typename sched_type::memory_space memory_space;

        size_t MemoryCapacity = 163840;

        enum { MinBlockSize   =   64 };
        enum { MaxBlockSize   = 1024 };
        enum { SuperBlockSize = 4096 };

        sched_type root_sched( memory_space()
                , MemoryCapacity
                , MinBlockSize
                , std::min(size_t(MaxBlockSize),MemoryCapacity)
                , std::min(size_t(SuperBlockSize),MemoryCapacity) );

        future_type f = Kokkos::host_spawn(Kokkos::TaskSingle(root_sched),
            DTT(root_sched, root, root, params));

        Kokkos::wait(root_sched);
    }
};

void dual_tree_traversal(t_fmm_params* params)
{   
    Kokkos::initialize();
    DTT::run(params, params->root);
    Kokkos::finalize();
    //dual_tree_traversal_core(params, params->root, params->root);
}

