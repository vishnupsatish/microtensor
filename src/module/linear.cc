/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#include "linear.h"

#include <random>

#include "random_gen.h"
#include "tensor.h"

Linear::Linear(int inFeaturesInp, int outFeaturesInp, bool bias) {
  if (inFeaturesInp <= 0 || outFeaturesInp <= 0) {
    throw std::runtime_error{"In and out features must be positive"};
  }
  size_t inFeatures = inFeaturesInp;
  size_t outFeatures = outFeaturesInp;
  m_weight = Tensor{Shape{outFeatures, inFeatures}, true};
  float sk = std::sqrt(1.0f / inFeatures);
  auto fn = std::uniform_real_distribution<float>{-sk, sk};
  auto init = std::bind_front(fn, RNG::get());
  m_weight.fillRandom(init);
  insertParameter(m_weight);

  if (bias) {
    m_bias = Tensor{Shape{outFeatures}, true};
    m_bias.fillRandom(init);
    insertParameter(m_bias);
  }
}

Tensor Linear::forward(Tensor inp) {
  auto wT = m_weight.permute({1, 0});
  auto mm = inp.matmul(wT);
  if (m_bias.isValid()) {
    return mm + m_bias;
  }
  return mm;
}
