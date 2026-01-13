/**
 *    Author:  Vishnu Satish
 *    Created: Jan 12, 2026
 **/

#include "layernorm.h"

LayerNorm::LayerNorm(Shape shape)
    : m_shape{std::move(shape)},
      m_numReduceElements{sizeFromShape(m_shape)},
      m_weight{Tensor{m_shape, std::vector<float>(m_numReduceElements, 1)}},
      m_bias{Tensor{m_shape, std::vector<float>(m_numReduceElements, 0)}},
      m_reduceDims(shape.size()) {
  insertParameter(m_weight);
  insertParameter(m_bias);
  for (int i = 0; i < shape.size(); ++i) {
    m_reduceDims[i] = -i - 1;
  }
}

Tensor LayerNorm::forward(Tensor inp) {
  auto ex = inp.reduceSum(m_reduceDims, true) / m_numReduceElements;
  auto num = inp - ex;
  auto ex2 = (inp * inp).reduceSum(m_reduceDims, true) / m_numReduceElements;
  auto var = ex2 - (ex * ex);
  auto denom = (var + m_epsilon).pow(0.5);
  return (num / denom) * m_weight + m_bias;
}
