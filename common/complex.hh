#pragma once

#include <complex>

#include "utils.hh"

#if !defined(__NVCC__)  && defined(__x86_64__) && (__GNUC__ >= 8)
#warning falling back to std complex numbers
template <class T>
using complex_t = std::complex<T>;

template <class T>
T complex_real(const std::complex<T>& a)
{
  return std::real(a);
}

template <class T>
std::complex<T> complex_conj(const std::complex<T>& a)
{
  return std::conj(a);
}

template <class T, class U>
std::complex<T> complex_pow(const std::complex<T>& a, const U& b)
{
  return std::pow(a, b);
}

template <class T>
std::complex<T> complex_exp(const std::complex<T>& a)
{
  return std::exp(a);
}

template <class T>
std::complex<T> imag_pow(const int n)
{
  return std::pow(std::complex<T>(0.0, 1.0), n);
}
#else

template <class T>
struct complex_t {
  T re, im;

  HOSTDEVICE complex_t(T arg_re, T arg_im) : re(arg_re), im(arg_im) {}
  HOSTDEVICE complex_t() : re(static_cast<T>(0.0)), im(static_cast<T>(0.0)) {}

  std::complex<T> convert() { return std::complex<T>(re, im); }

  HOSTDEVICE complex_t<T>& operator+=(const complex_t<T>& rhs)
  {
    this->re += rhs.re;
    this->im += rhs.im;
    return *this;
  }

  HOSTDEVICE T real() { return this->re; }
  HOSTDEVICE T imag() { return this->im; }
};

template <class T>
HOSTDEVICE complex_t<T> operator*(const complex_t<T>& a, const complex_t<T>& b)
{
  T re = a.re * b.re - a.im * b.im;
  T im = a.re * b.im + a.im * b.re;
  return complex_t<T>(re, im);
}

template <class T>
HOSTDEVICE complex_t<T> operator*(const complex_t<T>& a, const T& b)
{
  return complex_t<T>(a.re * b, a.im * b);
}

template <class T>
HOSTDEVICE complex_t<T> operator*(const T& a, const complex_t<T>& b)
{
  return b * a;
}

template <class T>
HOSTDEVICE complex_t<T> operator/(const complex_t<T>& a, T& b)
{
  return complex_t<T>(a.re / b, a.im / b);
}

template <class T>
HOSTDEVICE complex_t<T> complex_exp(const complex_t<T>& a)
{
  const T r = std::exp(a.re);
  return complex_t<T>(r * std::cos(a.im), r * std::sin(a.im));
}

template <class T>
HOSTDEVICE T complex_abs(const complex_t<T>& a)
{
  return std::hypot(a.re, a.im);
}

template <class T>
HOSTDEVICE complex_t<T> complex_pow(const complex_t<T>& a, const T& b)
{
  T r = complex_abs(a);
  if (a.re == 0.0) {
    printf("divide by zero\n");
    exit(1);
  }
  T phi = std::atan(a.im / a.re);
  return std::pow(r, b) * complex_t<T>(std::cos(phi * b), std::sin(phi * b));
}

template <class T>
HOSTDEVICE complex_t<T> imag_pow(int n)
{
  complex_t<T> i = (n & 1) ? complex_t<T>(0.0, 1.0) : complex_t<T>(1.0, 0.0);
  i = (n & 2) ? i * static_cast<T>(-1.0) : i;
  return i;
}

template <class T>
HOSTDEVICE complex_t<T> complex_conj(const complex_t<T>& a)
{
  return complex_t<T>(a.re, -a.im);
}

template <class T>
HOSTDEVICE T complex_real(const complex_t<T>& a)
{
  return a.re;
}

template <class T>
HOSTDEVICE T complex_imag(const complex_t<T>& a)
{
  return a.im;
}

template <class T>
HOSTDEVICE complex_t<T> convert(std::complex<T> a)
{
  return complex_t<T>(((T*)&a)[0], ((T*)&a)[1]);
}
#endif
