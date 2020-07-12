#include <cstdio>
#include <iostream>

#include <input.hh>
#include <init.hh>
#include <traversal.hh>
#include <finalise.hh>
#include <verify.hh>
#include <fmm.hh>

template <class T>
void perform_fmm(int argc, char** argv)
{
  FMM<T>* fmm = new FMM<T>();
  read_input(argc, argv, fmm);

  init(fmm);
  perform_traversals(fmm);
  verify(fmm);
  finalise(fmm);

  delete fmm;
} 

int main(int argc, char** argv)
{ 
#ifdef FMM_DOUBLE
  perform_fmm<double>(argc, argv);
#else
  perform_fmm<float>(argc, argv);
#endif
};
