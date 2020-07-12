#pragma once

#include <omp.h>

#include <kernels.hh>
#include <node.hh>
#include <get-deps.hh>

template <class T>
void upwards_pass(FMM<T>* fmm, node_t<T>* node)
{
  for (size_t i = 0; i < node->num_children; ++i) {
#pragma omp task
    upwards_pass(fmm, &fmm->nodes[node->child[i]]);
  }
#pragma omp taskwait

  if (node->is_leaf())
    p2m(fmm, node);
  else
    m2m(fmm, node);
}

template <class T>
void downwards_pass(FMM<T>* fmm, node_t<T>* node)
{
  if (node->is_leaf())
    l2p(fmm, node);
  else {
    l2l(fmm, node);
    for (size_t i = 0; i < node->num_children; ++i) {
#pragma omp task
      downwards_pass(fmm, &fmm->nodes[node->child[i]]);
    }
  }
#pragma omp taskwait
}

template <class T>
void dtt(FMM<T>* fmm)
{
  node_t<T>* nodes = fmm->nodes;
  T* x = fmm->x;
  T* y = fmm->y;
  T* z = fmm->z;
  T* w = fmm->w;
  T* ax = fmm->ax;
  T* ay = fmm->ay;
  T* az = fmm->az;
  T* aw = fmm->p;
  complex_t<T>* mm = fmm->m;
  complex_t<T>* ml = fmm->l;

  size_t nn = fmm->num_nodes;
  size_t np = fmm->num_points;
  size_t nm = fmm->num_multipoles;

#pragma omp target enter data map(to: nodes[:nn], x[:np], y[:np], z[:np], \
                                      w[:np], ax[:np], ay[:np], az[:np],  \
                                      aw[:np], mm[:nm * nn], ml[:nm * nn])

  Timer deps_timer;
  deps_timer.start();

  std::vector<std::vector<size_t>> p2p_deps(fmm->num_nodes);
  std::vector<std::vector<size_t>> m2l_deps(fmm->num_nodes);

  get_deps_omp(fmm, &p2p_deps, &m2l_deps);

  deps_timer.stop();
  printf("    %-16s %12.8f\n", "Deps. Time (s) ", deps_timer.elapsed());

  deps_timer.start();
  size_t* p2p_nodes;
  size_t* p2p_deps_array;
  size_t* p2p_deps_offsets;
  size_t* p2p_deps_sizes;
  size_t p2p_deps_tot;
  size_t p2p_num_nodes;

  size_t* m2l_nodes;
  size_t* m2l_deps_array;
  size_t* m2l_deps_offsets;
  size_t* m2l_deps_sizes;
  size_t m2l_deps_tot;
  size_t m2l_num_nodes;

  pack_deps(p2p_deps, &p2p_nodes, &p2p_deps_array, &p2p_deps_offsets,
            &p2p_deps_sizes, &p2p_deps_tot, &p2p_num_nodes);
  pack_deps(m2l_deps, &m2l_nodes, &m2l_deps_array, &m2l_deps_offsets,
            &m2l_deps_sizes, &m2l_deps_tot, &m2l_num_nodes);
  deps_timer.stop();
  printf("%-20s %12.8f\n", "  Pack Time (s) ", deps_timer.elapsed());

#pragma omp target enter data map(to: p2p_nodes[:p2p_num_nodes],\
                                  p2p_deps_array[:p2p_deps_tot],\
                                  p2p_deps_offsets[:p2p_num_nodes],\
                                  p2p_deps_sizes[:p2p_num_nodes])

#pragma omp target enter data map(to: m2l_nodes[:m2l_num_nodes],\
                                  m2l_deps_array[:m2l_deps_tot],\
                                  m2l_deps_offsets[:m2l_num_nodes],\
                                  m2l_deps_sizes[:m2l_num_nodes])

  Timer compute_timer;
  compute_timer.start();

#pragma omp target teams distribute 
  for (size_t ni = 0; ni < p2p_num_nodes; ++ni) {
    node_t<T>* target = &nodes[p2p_nodes[ni]];
    size_t p2p_size = p2p_deps_sizes[ni];
    size_t p2p_offset = p2p_deps_offsets[ni];

    for (size_t nj = 0; nj < p2p_size; ++nj) {
      size_t source_idx = p2p_deps_array[p2p_offset + nj];
      node_t<T>* source = nodes + source_idx;
//
//      if (target == source) p2p_tiled(fmm, target);
//      else p2p_tiled(fmm, target, source);
//    }
//  }

      //static __attribute((address_space(3))) 
        T shmem[512 * 4];
      T* source_pos = (T*)shmem;

#pragma omp parallel for
      for (size_t j = 0; j < source->num_points; ++j) {
        const size_t jj = j + source->point_idx;
        source_pos[j * 4 + 0] = x[jj];
        source_pos[j * 4 + 1] = y[jj];
        source_pos[j * 4 + 2] = z[jj];
        source_pos[j * 4 + 3] = w[jj];
      }

#pragma omp parallel for
      for (size_t i = 0; i < target->num_points; ++i) {
        const size_t ip = i + target->point_idx;
        const T xi = x[ip];
        const T yi = y[ip];
        const T zi = z[ip];
        T tax = static_cast<T>(0.0);
        T tay = static_cast<T>(0.0);
        T taz = static_cast<T>(0.0);
        T taw = static_cast<T>(0.0);
        for (size_t j = 0; j < source->num_points; ++j) {
          const size_t jp = j + source->point_idx;
          const T dx = source_pos[j * 4 + 0] - xi;
          const T dy = source_pos[j * 4 + 1] - yi;
          const T dz = source_pos[j * 4 + 2] - zi;
          const T wj = source_pos[j * 4 + 3];
          const T r = dx * dx + dy * dy + dz * dz;
          const T inv_r = (r == 0.0) ? 0.0 : 1.0/std::sqrt(r);
          const T inv_r_3 = inv_r * inv_r * inv_r * wj;
          tax += dx * inv_r_3;
          tay += dy * inv_r_3;
          taz += dz * inv_r_3;
          taw += inv_r * wj;
        }
        ax[ip] += tax;
        ay[ip] += tay;
        az[ip] += taz;
        aw[ip] += taw;
      }
    }
  }

//#pragma omp parallel for schedule(guided)
//  for (size_t i = 0; i < fmm->num_nodes; ++i) {
//    node_t<T>* target = &fmm->nodes[i];
//    for (size_t j = 0; j < p2p_deps[i].size(); ++j) {
//      node_t<T>* source = fmm->nodes + p2p_deps[i][j];
//      if (target == source) {
//        p2p_tiled(fmm, target);
//      }
//      else {
//        p2p_tiled(fmm, target, source);
//      }
//    }
//  }
  compute_timer.stop();
  printf("    %-16s %12.8f\n", "P2P Time (s) ", compute_timer.elapsed());

  compute_timer.start();
#pragma omp parallel for schedule(guided)
  for (size_t i = 0; i < fmm->num_nodes; ++i) {
    node_t<T>* target = &fmm->nodes[i];
    for (size_t j = 0; j < m2l_deps[i].size(); ++j) {
      node_t<T>* source = fmm->nodes + m2l_deps[i][j];
      m2l(fmm, target, source);
    }
  }
  compute_timer.stop();
  printf("    %-16s %12.8f\n", "M2L Time (s) ", compute_timer.elapsed());

#pragma omp target exit data map(from: ax[:np], ay[:np], az[:np], aw[:np])
}


template <class T>
void perform_traversals(FMM<T>* fmm)
{
#pragma omp parallel
#pragma omp single
  printf("Running on %d threads\n", omp_get_num_threads());

  Timer timer;
  Timer tot_timer;

  timer.start();
  tot_timer.start();
#pragma omp parallel
#pragma omp single
  upwards_pass(fmm, &fmm->nodes[fmm->root]);
  timer.stop();
  printf("\n");
  printf("%-20s %12.8f\n", "Upwards Time (s) ", timer.elapsed());

  dtt(fmm);

  timer.start();
#pragma omp parallel
#pragma omp single
  downwards_pass(fmm, &fmm->nodes[fmm->root]);
  timer.stop();
  printf("%-20s %12.8f\n", "Downwards Time (s) ", timer.elapsed());

  tot_timer.stop();
  printf("--------------------\n");
  printf("%-20s %12.8f\n", "Total Time (s) ", tot_timer.elapsed());
  printf("--------------------\n\n");
}
