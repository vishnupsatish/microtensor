/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#include "linear.h"

#include <random>

#include "random_gen.h"
#include "tensor.h"

Linear::Linear(int inFeaturesInp, int outFeaturesInp) {
  if (inFeaturesInp <= 0 || outFeaturesInp <= 0) {
    throw std::runtime_error{"In and out features must be positive"};
  }
  size_t inFeatures = inFeaturesInp;
  size_t outFeatures = outFeaturesInp;
  w = Tensor{Shape{outFeatures, inFeatures}, true};
  b = Tensor{Shape{outFeatures}, true};
  float sk = std::sqrt(1.0f / inFeatures);
  auto fn = std::uniform_real_distribution<float>{-sk, sk};
  auto init = std::bind_front(fn, RNG::get());
  w.fillRandom(init);
  b.fillRandom(init);
  insertParameter(w);
  insertParameter(b);
}

Tensor Linear::forward(Tensor inp) {
  auto wT = w.permute({1, 0});
  return inp.matmul(wT) + b;
}
