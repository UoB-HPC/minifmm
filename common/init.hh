#include <algorithm>
#include <complex.hh>
#include <random>
#include <tree.hh>
#include <utils.hh>

template <class T>
T rand_dist(T low, T high)
{
  const T randn = (T)rand() / (T)RAND_MAX;
  return (low + randn * (high - low));
}

template <class T>
void init(FMM<T>* fmm)
{
  fmm->theta2 = fmm->theta * fmm->theta;
  fmm->num_multipoles = fmm->num_terms * fmm->num_terms;
  fmm->num_spharm_terms = fmm->num_terms * fmm->num_terms;

  fmm->x = (T*)malloc(sizeof(T) * fmm->num_points);
  fmm->y = (T*)malloc(sizeof(T) * fmm->num_points);
  fmm->z = (T*)malloc(sizeof(T) * fmm->num_points);
  fmm->w = (T*)malloc(sizeof(T) * fmm->num_points);
  fmm->ax = (T*)malloc(sizeof(T) * fmm->num_points);
  fmm->ay = (T*)malloc(sizeof(T) * fmm->num_points);
  fmm->az = (T*)malloc(sizeof(T) * fmm->num_points);
  fmm->p = (T*)malloc(sizeof(T) * fmm->num_points);

  srand(42);

  if (fmm->dist == FMM<T>::Dist::Uniform) {
    for (size_t i = 0; i < fmm->num_points; ++i) {
      fmm->x[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
      fmm->y[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
      fmm->z[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
      fmm->w[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    }
  }
  else if (fmm->dist == FMM<T>::Dist::Plummer) {
    for (size_t i = 0; i < fmm->num_points; ++i) {
      T randn = (static_cast<T>(rand()) / static_cast<T>(RAND_MAX));
      randn += (randn == static_cast<T>(0.0))
                   ? std::numeric_limits<T>::epsilon()
                   : static_cast<T>(0.0);
      randn -= (randn == static_cast<T>(1.0))
                   ? std::numeric_limits<T>::epsilon()
                   : static_cast<T>(0.0);
      const T radius =
          static_cast<T>(1.0) / std::sqrt(std::pow(randn, (-2.0 / 3.0)) - 1.0);
      const T theta = std::acos(rand_dist(-1.0, 1.0));
      const T phi = rand_dist(0.0, 2.0 * M_PI);
      fmm->x[i] = radius * std::sin(theta) * std::cos(phi);
      fmm->y[i] = radius * std::sin(theta) * std::sin(phi);
      fmm->z[i] = radius * std::cos(theta);
      fmm->w[i] = static_cast<T>(1.0) / static_cast<T>(fmm->num_points);
    }
  }
  else {
    fprintf(stderr, "error: unknown input distribution type\n");
    exit(1);
  }

  std::fill(fmm->ax, fmm->ax + fmm->num_points, 0);
  std::fill(fmm->ay, fmm->ay + fmm->num_points, 0);
  std::fill(fmm->az, fmm->az + fmm->num_points, 0);
  std::fill(fmm->p, fmm->p + fmm->num_points, 0);

  int num_terms = fmm->num_terms;

  fmm->inner_factors =
      (complex_t<T>*)malloc(sizeof(complex_t<T>) * num_terms * num_terms);
  fmm->outer_factors =
      (complex_t<T>*)malloc(sizeof(complex_t<T>) * num_terms * num_terms);

  std::fill(fmm->inner_factors, fmm->inner_factors + (num_terms * num_terms),
            complex_t<T>(static_cast<T>(0.0), static_cast<T>(0.0)));
  std::fill(fmm->outer_factors, fmm->outer_factors + (num_terms * num_terms),
            complex_t<T>(static_cast<T>(0.0), static_cast<T>(0.0)));

  int max = 2 * num_terms - 1;
  T factorial[max];
  factorial[0] = 1.0;
  for (int i = 1; i < max; ++i) factorial[i] = i * factorial[i - 1];

  for (int n = 0; n < num_terms; ++n) {
    for (int m = -n; m <= n; ++m) {
      fmm->inner_factors[mult_idx(n, m)] =
          (std::pow(static_cast<T>(-1.0), static_cast<T>(n)) *
           imag_pow<T>(std::abs(m))) /
          factorial[n + std::abs(m)];
      fmm->outer_factors[mult_idx(n, m)] =
          imag_pow<T>(-std::abs(m)) * factorial[n - std::abs(m)];
    }
  }
  build_tree(fmm);
}
