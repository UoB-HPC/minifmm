#pragma once

#include <fmm.hh>

template<class T>
void finalise(FMM<T>* fmm)
{
  free(fmm->x);
  free(fmm->y);
  free(fmm->z);
  free(fmm->w);
  free(fmm->ax);
  free(fmm->ay);
  free(fmm->az);
  free(fmm->p);
  free(fmm->inner_factors);
  free(fmm->outer_factors);
  free(fmm->m);
  free(fmm->l);
  free(fmm->nodes);
}
