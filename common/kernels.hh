#pragma once

#include <spharm.hh>
#include <utils.hh>

template <class T>
void p2p_tiled(FMM<T>* fmm, node_t<T>* target)
{
  const size_t ip = target->point_idx;
  for (size_t ii = 0; ii < target->num_points; ii += TILE_SIZE) {
    T xi[TILE_SIZE] = {0};
    T yi[TILE_SIZE] = {0};
    T zi[TILE_SIZE] = {0};
    T ax[TILE_SIZE] = {0};
    T ay[TILE_SIZE] = {0};
    T az[TILE_SIZE] = {0};
    T aw[TILE_SIZE] = {0};
    const int ilim = std::min((size_t)TILE_SIZE, target->num_points - ii);
    for (int i = 0; i < ilim; ++i) {
      xi[i] = fmm->x[i + ii + ip];
      yi[i] = fmm->y[i + ii + ip];
      zi[i] = fmm->z[i + ii + ip];
    }
    for (size_t j = 0; j < target->num_points; ++j) {
      for (int i = 0; i < TILE_SIZE; ++i) {
        const T dx = fmm->x[j + ip] - xi[i];
        const T dy = fmm->y[j + ip] - yi[i];
        const T dz = fmm->z[j + ip] - zi[i];
        const T sw = fmm->w[j + ip];
        const T r = dx * dx + dy * dy + dz * dz;
        const T inv_r = (r == 0.0) ? 0.0 : 1.0 / std::sqrt(r);
        const T inv_r_3 = sw * inv_r * inv_r * inv_r;
        ax[i] += dx * inv_r_3;
        ay[i] += dy * inv_r_3;
        az[i] += dz * inv_r_3;
        aw[i] += sw * inv_r;
      }
    }
    for (int i = 0; i < ilim; ++i) {
      fmm->ax[i + ii + ip] += ax[i];
      fmm->ay[i + ii + ip] += ay[i];
      fmm->az[i + ii + ip] += az[i];
      fmm->p[i + ii + ip] += aw[i];
    }
  }
}

template <class T>
void p2p(FMM<T>* fmm, node_t<T>* target)
{
  for (size_t i = 0; i < target->num_points; ++i) {
    const size_t ip = i + target->point_idx;
    const T xi = fmm->x[ip];
    const T yi = fmm->y[ip];
    const T zi = fmm->z[ip];
    T ax = static_cast<T>(0.0);
    T ay = static_cast<T>(0.0);
    T az = static_cast<T>(0.0);
    T p = static_cast<T>(0.0);
    for (size_t j = 0; j < target->num_points; ++j) {
      const size_t jp = j + target->point_idx;
      const T dx = fmm->x[jp] - xi;
      const T dy = fmm->y[jp] - yi;
      const T dz = fmm->z[jp] - zi;
      const T r = dx * dx + dy * dy + dz * dz;
      const T inv_r = (r == 0.0) ? 0.0 : 1.0 / std::sqrt(r);
      const T inv_r_3 = inv_r * inv_r * inv_r * fmm->w[jp];
      ax += dx * inv_r_3;
      ay += dy * inv_r_3;
      az += dz * inv_r_3;
      p += fmm->w[jp] * inv_r;
    }
    fmm->ax[ip] += ax;
    fmm->ay[ip] += ay;
    fmm->az[ip] += az;
    fmm->p[ip] += p;
  }
}

template <class T>
void p2p_tiled(FMM<T>* fmm, node_t<T>* target, node_t<T>* source)
{
  const size_t jp = source->point_idx;
  const size_t ip = target->point_idx;
  for (size_t ii = 0; ii < target->num_points; ii += TILE_SIZE) {
    T xi[TILE_SIZE] = {0};
    T yi[TILE_SIZE] = {0};
    T zi[TILE_SIZE] = {0};
    T ax[TILE_SIZE] = {0};
    T ay[TILE_SIZE] = {0};
    T az[TILE_SIZE] = {0};
    T aw[TILE_SIZE] = {0};
    const int ilim = std::min((size_t)TILE_SIZE, target->num_points - ii);
    for (int i = 0; i < ilim; ++i) {
      xi[i] = fmm->x[i + ii + ip];
      yi[i] = fmm->y[i + ii + ip];
      zi[i] = fmm->z[i + ii + ip];
    }
    for (size_t j = 0; j < source->num_points; ++j) {
      for (int i = 0; i < TILE_SIZE; ++i) {
        const T dx = fmm->x[j + jp] - xi[i];
        const T dy = fmm->y[j + jp] - yi[i];
        const T dz = fmm->z[j + jp] - zi[i];
        const T sw = fmm->w[j + jp];
        const T r = dx * dx + dy * dy + dz * dz;
        const T inv_r = 1.0 / std::sqrt(r);
        const T inv_r_3 = sw * inv_r * inv_r * inv_r;
        ax[i] += dx * inv_r_3;
        ay[i] += dy * inv_r_3;
        az[i] += dz * inv_r_3;
        aw[i] += sw * inv_r;
      }
    }
    for (int i = 0; i < ilim; ++i) {
      fmm->ax[i + ii + ip] += ax[i];
      fmm->ay[i + ii + ip] += ay[i];
      fmm->az[i + ii + ip] += az[i];
      fmm->p[i + ii + ip] += aw[i];
    }
  }
}

template <class T>
void p2p(FMM<T>* fmm, node_t<T>* target, node_t<T>* source)
{
  for (size_t i = 0; i < target->num_points; ++i) {
    const size_t ip = i + target->point_idx;
    const T xi = fmm->x[ip];
    const T yi = fmm->y[ip];
    const T zi = fmm->z[ip];
    T ax = static_cast<T>(0.0);
    T ay = static_cast<T>(0.0);
    T az = static_cast<T>(0.0);
    T p = static_cast<T>(0.0);
    for (size_t j = 0; j < source->num_points; ++j) {
      const size_t jp = j + source->point_idx;
      const T dx = fmm->x[jp] - xi;
      const T dy = fmm->y[jp] - yi;
      const T dz = fmm->z[jp] - zi;
      const T r = dx * dx + dy * dy + dz * dz;
      const T inv_r = 1.0 / std::sqrt(r);
      const T inv_r_3 = fmm->w[jp] * inv_r * inv_r * inv_r;
      ax += dx * inv_r_3;
      ay += dy * inv_r_3;
      az += dz * inv_r_3;
      p += fmm->w[jp] * inv_r;
    }
    fmm->ax[ip] += ax;
    fmm->ay[ip] += ay;
    fmm->az[ip] += az;
    fmm->p[ip] += p;
  }
}

template <class T>
void m2l(FMM<T>* fmm, node_t<T>* target, node_t<T>* source)
{
  int num_terms = fmm->num_terms;
  T dx = target->cx - source->cx;
  T dy = target->cy - source->cy;
  T dz = target->cz - source->cz;
  complex_t<T> outer[num_terms * num_terms];
  T rho, alpha, beta;
  cart_to_sph(dx, dy, dz, rho, alpha, beta);
  compute_outer<1>(fmm, rho, alpha, beta, outer);
  complex_t<T>* Msource = &fmm->m[source->mult_idx];
  complex_t<T>* Ltarget = &fmm->l[target->mult_idx];
  for (int j = 0; j < num_terms; ++j) {
    for (int k = -j; k <= j; ++k) {
      complex_t<T> tmp(static_cast<T>(0.0), static_cast<T>(0.0));
      for (int n = 0; n < num_terms - j; ++n) {
        for (int m = -n; m <= n; ++m) {
          tmp += Msource[mult_idx(n, m)] * outer[mult_idx(j + n, -k - m)];
          // blah
        }
      }
      Ltarget[mult_idx(j, k)] += tmp;
    }
  }
}

template <class T>
void p2m(FMM<T>* fmm, node_t<T>* node)
{
  int num_terms = fmm->num_terms;
  size_t pt_offset = node->point_idx;
  size_t mt_offset = node->mult_idx;
  for (size_t i = 0; i < node->num_points; ++i) {
    T dx = fmm->x[i + pt_offset] - node->cx;
    T dy = fmm->y[i + pt_offset] - node->cy;
    T dz = fmm->z[i + pt_offset] - node->cz;
    complex_t<T> inner[num_terms * num_terms];
    T r, theta, phi;
    cart_to_sph(dx, dy, dz, r, theta, phi);
    compute_inner<1>(fmm, r, theta, phi, inner);
    for (int n = 0; n < num_terms; ++n) {
      for (int m = -n; m <= n; ++m) {
        fmm->m[mt_offset + mult_idx(n, m)] +=
            fmm->w[i + pt_offset] *
            std::pow(static_cast<T>(-1.0), static_cast<T>(n)) *
            inner[mult_idx(n, m)];
      }
    }
  }
}

template <class T>
void m2m(FMM<T>* fmm, node_t<T>* node)
{
  int num_terms = fmm->num_terms;
  for (size_t i = 0; i < node->num_children; ++i) {
    complex_t<T> inner[num_terms * num_terms];
    node_t<T>* child = &fmm->nodes[node->child[i]];
    T dx = node->cx - child->cx;
    T dy = node->cy - child->cy;
    T dz = node->cz - child->cz;
    T r, theta, phi;
    cart_to_sph(dx, dy, dz, r, theta, phi);
    compute_inner<1>(fmm, r, theta, phi, inner);
    const complex_t<T>* Mchild = &fmm->m[child->mult_idx];
    complex_t<T>* Mnode = &fmm->m[node->mult_idx];
    for (int j = 0; j < num_terms; ++j) {
      for (int k = -j; k <= j; ++k) {
        complex_t<T> tmp(static_cast<T>(0.0), static_cast<T>(0.0));
        for (int n = 0; n <= j; ++n) {
          for (int m = -n; m <= n; ++m) {
            if (abs(k - m) <= j - n)
              tmp += Mchild[mult_idx(n, m)] * inner[mult_idx(j - n, k - m)];
          }
        }
        Mnode[mult_idx(j, k)] += tmp;
      }
    }
  }
}

template <class T>
void l2l(FMM<T>* fmm, node_t<T>* node)
{
  int num_terms = fmm->num_terms;
  complex_t<T> inner[num_terms * num_terms];
  for (size_t i = 0; i < node->num_children; ++i) {
    node_t<T>* child = &fmm->nodes[node->child[i]];
    // TODO flip these?
    T dx = child->cx - node->cx;
    T dy = child->cy - node->cy;
    T dz = child->cz - node->cz;
    T rho, alpha, beta;
    cart_to_sph(dx, dy, dz, rho, alpha, beta);
    compute_inner<1>(fmm, rho, alpha, beta, inner);
    complex_t<T>* Lnode = &fmm->l[node->mult_idx];
    complex_t<T>* Lchild = &fmm->l[child->mult_idx];
    for (int j = 0; j < num_terms; ++j) {
      for (int k = -j; k <= j; ++k) {
        complex_t<T> tmp(static_cast<T>(0.0), static_cast<T>(0.0));
        for (int n = j; n < num_terms; ++n) {
          for (int m = -n; m <= n; ++m) {
            if (std::abs(m - k) <= n - j) {
              tmp += Lnode[mult_idx(n, m)] * inner[mult_idx(n - j, m - k)];
            }
          }
        }
        Lchild[mult_idx(j, k)] += tmp;
      }
    }
  }
}

template <class T>
void l2p(FMM<T>* fmm, node_t<T>* node)
{
  int num_terms = fmm->num_terms;
  complex_t<T> inner[num_terms * num_terms];
  complex_t<T> inner_deriv[num_terms * num_terms];
  size_t pt_offset = node->point_idx;
  complex_t<T>* Lnode = &fmm->l[node->mult_idx];
  for (size_t i = 0; i < node->num_points; ++i) {
    T dx = fmm->x[pt_offset + i] - node->cx;
    T dy = fmm->y[pt_offset + i] - node->cy;
    T dz = fmm->z[pt_offset + i] - node->cz;
    T r, theta, phi;
    cart_to_sph(dx, dy, dz, r, theta, phi);
    compute_inner<2>(fmm, r, theta, phi, inner, inner_deriv);

    T Psum = static_cast<T>(0.0);
    T rsum = static_cast<T>(0.0);
    T tsum = static_cast<T>(0.0);
    T psum = static_cast<T>(0.0);
    T two = static_cast<T>(2.0);
    complex_t<T> ci(static_cast<T>(0.0), static_cast<T>(1.0));
    for (int n = 0; n < num_terms; ++n) {
      int m = 0;
      Psum += complex_real(Lnode[mult_idx(n, m)] * inner[mult_idx(n, m)]);
      rsum += static_cast<T>(n) *
              complex_real(Lnode[mult_idx(n, m)] * inner[mult_idx(n, m)]);
      tsum += complex_real(Lnode[mult_idx(n, m)] * inner_deriv[mult_idx(n, m)]);
      psum += static_cast<T>(m) *
              complex_real(Lnode[mult_idx(n, m)] * inner[mult_idx(n, m)] * ci);
      for (int m = 1; m <= n; ++m) {
        Psum +=
            two * complex_real(Lnode[mult_idx(n, m)] * inner[mult_idx(n, m)]);
        rsum += two * static_cast<T>(n) *
                complex_real(Lnode[mult_idx(n, m)] * inner[mult_idx(n, m)]);
        tsum += two * complex_real(Lnode[mult_idx(n, m)] *
                                   inner_deriv[mult_idx(n, m)]);
        psum +=
            two * static_cast<T>(m) *
            complex_real(Lnode[mult_idx(n, m)] * inner[mult_idx(n, m)] * ci);
      }
    }
    T inv_r = (r == static_cast<T>(0.0)) ? 0.0 : static_cast<T>(1.0) / r;
    rsum *= inv_r;
    tsum *= inv_r;
    psum *= inv_r;
    psum *= (theta == static_cast<T>(0.0))
                ? 0.0
                : static_cast<T>(1.0) / std::sin(theta);
    T ax, ay, az;
    sph_unit_to_cart_unit(r, theta, phi, rsum, tsum, psum, ax, ay, az);
    fmm->p[pt_offset + i] += Psum;
    fmm->ax[pt_offset + i] += ax;
    fmm->ay[pt_offset + i] += ay;
    fmm->az[pt_offset + i] += az;
  }
}

template <class T>
void m2p(FMM<T>* fmm, node_t<T>* target, node_t<T>* source)
{
  int num_terms = fmm->num_terms;
  size_t target_pt_offset = target->point_idx;
  size_t source_mt_offset = source->mult_idx;
  for (size_t i = 0; i < target->num_points; ++i) {
    T dx = fmm->x[target_pt_offset + i] - source->cx;
    T dy = fmm->y[target_pt_offset + i] - source->cy;
    T dz = fmm->z[target_pt_offset + i] - source->cz;
    T r, theta, phi;
    cart_to_sph(dx, dy, dz, r, theta, phi);
    complex_t<T> outer[num_terms * num_terms];
    complex_t<T> outer_deriv[num_terms * num_terms];
    compute_outer<2>(fmm, r, theta, phi, outer, outer_deriv);
    T Psum = static_cast<T>(0.0);
    T rsum = static_cast<T>(0.0);
    T tsum = static_cast<T>(0.0);
    T psum = static_cast<T>(0.0);
    T two = static_cast<T>(2.0);
    const complex_t<T>* M = &fmm->m[source_mt_offset];
    const complex_t<T> ci(static_cast<T>(0.0), static_cast<T>(1.0));
    for (int n = 0; n < num_terms; ++n) {
      int m = 0;
      Psum += (outer[mult_idx(n, -m)] * M[mult_idx(n, m)]).real();
      rsum += -static_cast<T>(n + 1) *
              complex_real(outer[mult_idx(n, -m)] * M[mult_idx(n, m)]);
      tsum += complex_real(outer_deriv[mult_idx(n, -m)] * M[mult_idx(n, m)]);
      psum += static_cast<T>(m) *
              complex_real(ci * outer[mult_idx(n, -m)] * M[mult_idx(n, m)]);
      for (m = 1; m <= n; ++m) {
        Psum += two * complex_real(outer[mult_idx(n, -m)] *
                                   fmm->m[source_mt_offset + mult_idx(n, m)]);
        rsum += two * -static_cast<T>(n + 1) *
                complex_real(outer[mult_idx(n, -m)] * M[mult_idx(n, m)]);
        tsum += two *
                complex_real(outer_deriv[mult_idx(n, -m)] * M[mult_idx(n, m)]);
        psum += two * static_cast<T>(m) *
                complex_real(ci * outer[mult_idx(n, -m)] * M[mult_idx(n, m)]);
      }
    }
    rsum *= static_cast<T>(1.0) / r;
    tsum *= static_cast<T>(1.0) / r;
    psum *= static_cast<T>(1.0) / r;
    psum /= std::sin(theta);
    T ax, ay, az;
    sph_unit_to_cart_unit(r, theta, phi, rsum, tsum, psum, ax, ay, az);
    fmm->p[target_pt_offset + i] += Psum;
    fmm->ax[target_pt_offset + i] += ax;
    fmm->ay[target_pt_offset + i] += ay;
    fmm->az[target_pt_offset + i] += az;
  }
}
