#pragma once

#include <vector>

#include <complex.hh>
#include <node.hh>

#ifndef TASK_CUTOFF
#define TASK_CUTOFF 5000
#endif
#define SOURCE_TASK_SPAWN 0

template <class T>
struct FMM
{
  node_t<T>* nodes;
  size_t root;
  T* x = nullptr; 
  T* y = nullptr; 
  T* z = nullptr; 
  T* w = nullptr; 
  T* ax = nullptr;
  T* ay = nullptr;
  T* az = nullptr;
  T* p = nullptr;
  size_t num_points;
  size_t ncrit;
  int num_terms;
  T theta;
  T theta2;
  size_t num_samples;
  size_t num_multipoles;
  size_t num_nodes;
  size_t num_spharm_terms;
  complex_t<T>* inner_factors = nullptr;
  complex_t<T>* outer_factors = nullptr;
  complex_t<T>* m = nullptr;
  complex_t<T>* l = nullptr;
  enum Dist {
    Uniform = 0,
    Plummer,
    NumDist,
  };
  Dist dist;
};
