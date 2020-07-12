#pragma once

#include <limits>
#include <gpu-utils.hh>

#if defined(__x86_64__)
  #if defined (__AVX512F__) 
    #ifdef FMM_DOUBLE
      #define TILE_SIZE 8
    #else
      #define TILE_SIZE 64
    #endif
  #elif defined (__AVX2__)
    #define TILE_SIZE 16
  #else
    #define TILE_SIZE 8
  #endif
#elif defined (__aarch64__)
  #define TILE_SIZE 4
#else
  #warning architecture not supported
  #define TILE_SIZE 32
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//#define XSTR(x) STR(x)
//#define STR(x) #x
//#pragma message "tile size = " XSTR(TILE_SIZE)

HOSTDEVICE inline int mult_idx(const int n, const int m)
{
  return n * n + n + m;
}

HOSTDEVICE inline int leg_idx(const int n, const int m)
{
  return (n * (n + 1)) / 2 + std::abs(m);
}

HOSTDEVICE inline void inv_mult_idx(const int i, int& n, int& m)
{
  n = (int)sqrtf((float)i);
  m = i - n * n - n;
}

template <class T>
HOSTDEVICE void sph_unit_to_cart_unit(T r, T theta, T phi, T rsum, T tsum,
                                      T psum, T& ax, T& ay, T& az)
{
  ax = std::sin(theta) * std::cos(phi) * rsum +
       std::cos(theta) * std::cos(phi) * tsum - std::sin(phi) * psum;
  ay = std::sin(theta) * std::sin(phi) * rsum +
       std::cos(theta) * std::sin(phi) * tsum + std::cos(phi) * psum;
  az = std::cos(theta) * rsum - std::sin(theta) * tsum;
}

template <class T>
HOSTDEVICE T get_eps();
template <>
HOSTDEVICE float get_eps() { return 1e-6; }
template <>
HOSTDEVICE double get_eps() { return 1e-14; }

template <class T>
HOSTDEVICE inline void cart_to_sph(T x, T y, T z, T& r, T& theta, T& phi)
{
  const T eps = get_eps<T>();
  r = std::sqrt(x * x + y * y + z * z) + eps;
  theta = std::acos(z / r);
  phi = std::atan2(y, x);
}

