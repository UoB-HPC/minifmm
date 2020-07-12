#pragma once

#include <fmm.hh>
#include <utils.hh>

template <int order, class T>
HOSTDEVICE void compute_legendre(size_t nmax, T x, T* P, T* P_deriv = nullptr);

template <int order, class T>
void compute_inner(FMM<T>* fmm, T r, T theta, T phi, complex_t<T>* inner,
                   complex_t<T>* inner_deriv = nullptr)
{
  if (fmm->num_terms == 0) return;
  //// TODO this can be reduced as we only calculate the 'positive' legendre
  /// vals
  T legendre[fmm->num_terms * fmm->num_terms];
  T legendre_deriv[fmm->num_terms * fmm->num_terms];
  // TODO forward order to compute_legendre
  if (order == 1) {
    compute_legendre<1>(fmm->num_terms - 1, std::cos(theta), legendre);
  }
  if (order == 2) {
    compute_legendre<2>(fmm->num_terms - 1, std::cos(theta), legendre,
                        legendre_deriv);
  }

  const complex_t<T> i = complex_t<T>(static_cast<T>(0.0), static_cast<T>(1.0));

  for (int n = 0; n < fmm->num_terms; ++n) {
    inner[mult_idx(n, 0)] = fmm->inner_factors[mult_idx(n, 0)] *
                            legendre[leg_idx(n, 0)] *
                            complex_exp(i * static_cast<T>(0) * phi) *
                            std::pow(r, static_cast<T>(n));
    if (order == 2)
      inner_deriv[mult_idx(n, 0)] = fmm->inner_factors[mult_idx(n, 0)] *
                                    legendre_deriv[leg_idx(n, 0)] *
                                    complex_exp(i * static_cast<T>(0) * phi) *
                                    std::pow(r, static_cast<T>(n));
    for (int m = 1; m <= n; ++m) {
      inner[mult_idx(n, m)] = fmm->inner_factors[mult_idx(n, m)] *
                              legendre[leg_idx(n, m)] *
                              complex_exp(i * static_cast<T>(m) * phi) *
                              std::pow(r, static_cast<T>(n));
      inner[mult_idx(n, -m)] =
          std::pow(static_cast<T>(-1.0), static_cast<T>(m)) *
          complex_conj(inner[mult_idx(n, m)]);
      if (order == 2) {
        inner_deriv[mult_idx(n, m)] = fmm->inner_factors[mult_idx(n, m)] *
                                      legendre_deriv[leg_idx(n, m)] *
                                      complex_exp(i * static_cast<T>(m) * phi) *
                                      std::pow(r, static_cast<T>(n));
        inner_deriv[mult_idx(n, -m)] =
            std::pow(static_cast<T>(-1.0), static_cast<T>(m)) *
            complex_conj(inner_deriv[mult_idx(n, m)]);
      }
    }
  }
}

template <int order, class T>
void compute_outer(FMM<T>* fmm, T r, T theta, T phi, complex_t<T>* outer,
                   complex_t<T>* outer_deriv = nullptr)
{
  if (fmm->num_terms == 0) return;
  // TODO this can be reduced as we only calculate the 'positive' legendre vals
  T legendre[fmm->num_terms * fmm->num_terms];
  T legendre_deriv[fmm->num_terms * fmm->num_terms];

  if (order == 1)
    compute_legendre<1>(fmm->num_terms - 1, std::cos(theta), legendre);
  if (order == 2) {
    compute_legendre<2>(fmm->num_terms - 1, std::cos(theta), legendre,
                        legendre_deriv);
  }

  const complex_t<T> i = complex_t<T>(static_cast<T>(0.0), static_cast<T>(1.0));

  for (int n = 0; n < fmm->num_terms; ++n) {
    for (int m = 0; m <= n; ++m) {
      outer[mult_idx(n, m)] =
          fmm->outer_factors[mult_idx(n, m)] * legendre[leg_idx(n, m)] *
          complex_exp(i * static_cast<T>(m) * phi) *
          (static_cast<T>(1.0) / std::pow(r, static_cast<T>(n + 1)));
      outer[mult_idx(n, -m)] =
          std::pow(static_cast<T>(-1.0), static_cast<T>(m)) *
          complex_conj(outer[mult_idx(n, m)]);
      if (order == 2) {
        outer_deriv[mult_idx(n, m)] =
            fmm->outer_factors[mult_idx(n, m)] * legendre_deriv[leg_idx(n, m)] *
            complex_exp(i * static_cast<T>(m) * phi) *
            (static_cast<T>(1.0) / std::pow(r, static_cast<T>(n + 1)));
        outer_deriv[mult_idx(n, -m)] =
            std::pow(static_cast<T>(-1.0), static_cast<T>(m)) *
            complex_conj(outer_deriv[mult_idx(n, m)]);
        // TODO negative derivs may need to be calculated
      }
    }
  }
}

// TODO test with 'if constexpr' with order (C++17 feature)
template <int order, class T>
HOSTDEVICE void compute_legendre(size_t nmax, T x, T* P, T* P_deriv)
{
  const T csphase = static_cast<T>(-1.0);
  const T one = static_cast<T>(1.0);
  const T u = (x == 1.0) ? 0.0 : std::sqrt((one - x) * (one + x));
  const T uinv = (u == static_cast<T>(0.0)) ? static_cast<T>(0.0) : one / u;
  const T xbyu = x * uinv;
  size_t n, m;
  size_t k, idxmm;
  T pnm, pmm, pm1, pm2, twomm1;
  pm2 = one;
  pm1 = x;

  P[0] = pm2;
  if (order >= 2) P_deriv[0] = static_cast<T>(0.0);
  if (nmax == 0) return;
  P[1] = pm1;
  if (order >= 2) P_deriv[1] = -u;

  k = 1;
  for (n = 2; n <= nmax; ++n) {
    k += n;
    pnm = (static_cast<T>(2 * n - 1) * x * pm1 - static_cast<T>(n - 1) * pm2) /
          static_cast<T>(n);
    P[k] = pnm;
    if (order >= 2) P_deriv[k] = -static_cast<T>(n) * (pm1 - x * pnm) * uinv;
    pm2 = pm1;
    pm1 = pnm;
  }

  pmm = one;
  twomm1 = -one;
  idxmm = 0;
  for (m = 1; m <= nmax - 1; ++m) {
    idxmm += m + 1;
    twomm1 += static_cast<T>(2.0);
    pmm *= csphase * u * twomm1;
    P[idxmm] = pmm;
    if (order >= 2) P_deriv[idxmm] = static_cast<T>(m) * xbyu * pmm;
    pm2 = pmm;
    k = idxmm + m + 1;
    pm1 = x * pmm * static_cast<T>(2 * m + 1);
    P[k] = pm1;
    if (order >= 2)
      P_deriv[k] = -uinv * (static_cast<T>(2 * m + 1) * pmm -
                            static_cast<T>(m + 1) * x * pm1);

    for (n = m + 2; n <= nmax; ++n) {
      k += n;
      pnm = (static_cast<T>(2 * n - 1) * x * pm1 -
             static_cast<T>(n + m - 1) * pm2) /
            static_cast<T>(n - m);
      P[k] = pnm;
      if (order >= 2)
        P_deriv[k] =
            -uinv * (static_cast<T>(n + m) * pm1 - static_cast<T>(n) * x * pnm);
      pm2 = pm1;
      pm1 = pnm;
    }
  }

  idxmm += m + 1;
  twomm1 += static_cast<T>(2.0);
  pmm *= csphase * u * twomm1;
  P[idxmm] = pmm;
  if (order >= 2) P_deriv[idxmm] = static_cast<T>(nmax) * x * pmm * uinv;
}
