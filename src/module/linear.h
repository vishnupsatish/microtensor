/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#pragma once

#include "module.h"

class Linear : public Module {
  Tensor m_weight;
  Tensor m_bias;

 public:
  Linear(int inFeatures, int outFeatures, bool bias = true);
  Tensor forward(Tensor inp);
};
