/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#pragma once

#include "module.h"

class Linear : public Module {
  Tensor w, b;

 public:
  Linear(int inFeatures, int outFeatures);
  Tensor forward(Tensor inp);
};
