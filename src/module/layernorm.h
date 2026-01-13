/**
 *    Author:  Vishnu Satish
 *    Created: Jan 12, 2026
 **/

#pragma once

#include "module.h"

class LayerNorm : public Module {
  // The last dimensions to normalize, also the size of m_gamma and m_beta.
  Shape m_shape;
  size_t m_numReduceElements;
  Tensor m_weight, m_bias;
  std::vector<int> m_reduceDims;
  float m_epsilon = 1e-05;

 public:
  LayerNorm(Shape shape);
  Tensor forward(Tensor inp);
};
