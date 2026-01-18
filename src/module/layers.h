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

class Embedding : public Module {
  Tensor m_embedding;

 public:
  Embedding(int vocabSize, int embeddingSize);
  Tensor forward(Tensor inp);
};

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

class MLP : public Module {
  size_t m_featureSize;
  size_t m_projectionSize;

  std::unique_ptr<Linear> m_linear1;
  std::unique_ptr<Linear> m_linear2;

 public:
  MLP(size_t featureSize, size_t projectionSize);
  Tensor forward(Tensor x);
};
