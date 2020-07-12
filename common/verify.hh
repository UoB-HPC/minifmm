#pragma once

template <class T>
T calc_error(T* p, T* test_p, size_t n)
{
  T diff = static_cast<T>(0.0);
  T norm = static_cast<T>(0.0);
  for (size_t i = 0; i < n; ++i) {
    diff += (p[i] - test_p[i]) * (p[i] - test_p[i]);
    norm += test_p[i] * test_p[i];
  }
  return std::sqrt(diff / norm);
}

template <class T>
T calc_error(T* x, T* y, T* z, T* test_x, T* test_y, T* test_z, size_t n)
{
  T diff = static_cast<T>(0.0);
  T norm = static_cast<T>(0.0);
  for (size_t i = 0; i < n; ++i) {
    T dx = (x[i] - test_x[i]) * (x[i] - test_x[i]);
    T dy = (y[i] - test_y[i]) * (y[i] - test_y[i]);
    T dz = (z[i] - test_z[i]) * (z[i] - test_z[i]);
    diff += dx + dy + dz;
    norm += (test_x[i] * test_x[i]) + (test_y[i] * test_y[i]) +
            (test_z[i] * test_z[i]);
  }
  return std::sqrt(diff/norm);
}

template <class T>
void verify(FMM<T>* fmm)
{
  std::vector<T> test_ax(fmm->num_samples);
  std::vector<T> test_ay(fmm->num_samples);
  std::vector<T> test_az(fmm->num_samples);
  std::vector<T> test_p(fmm->num_samples);

  #pragma omp parallel for
  for (size_t i = 0; i < fmm->num_samples; ++i) {
    for (size_t j = 0; j < fmm->num_points; ++j) {
      if (i == j) continue;
      const T dx = fmm->x[j] - fmm->x[i];
      const T dy = fmm->y[j] - fmm->y[i];
      const T dz = fmm->z[j] - fmm->z[i];
      const T r = std::sqrt(dx * dx + dy * dy + dz * dz);
      const T inv_r = static_cast<T>(1.0) / r;
      const T inv_r_3 = inv_r * inv_r * inv_r;
      const T s = inv_r_3 * fmm->w[j];
      test_ax[i] += s * dx;
      test_ay[i] += s * dy;
      test_az[i] += s * dz;
      test_p[i] += inv_r * fmm->w[j];
    }
  }
 //for (size_t i = 0; i < fmm->num_samples; ++i) {
 //  printf("%f vs %f\n", fmm->p[i], test_p[i]);
 //}
  printf("pot err = %.12e\n", calc_error(fmm->p, &test_p[0], fmm->num_samples));
  printf("acc err = %.12e\n",
         calc_error(fmm->ax, fmm->ay, fmm->az, &test_ax[0], &test_ay[0],
                    &test_az[0], fmm->num_samples));
}
