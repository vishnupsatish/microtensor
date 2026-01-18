/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#include "layers.h"

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

////////////////////////////////////////////////////////////////////////////////

Embedding::Embedding(int vocabSize, int embeddingSize) {
  if (vocabSize <= 0 || embeddingSize <= 0) {
    throw std::runtime_error("Vocab and embedding sizes must be positive");
  }
  size_t vs = vocabSize;
  size_t es = embeddingSize;
  m_embedding = Tensor{Shape{vs, es}, true};
  // PyTorch initialization.
  auto fn = std::normal_distribution<float>{0, 1};
  auto init = std::bind_front(fn, RNG::get());
  m_embedding.fillRandom(init);
  insertParameter(m_embedding);
}

Tensor Embedding::forward(Tensor inp) {
  Shape inpShape = inp.getShape();
  size_t numElements = sizeFromShape(inpShape);
  Tensor indices = inp.reshape({numElements, 1});

  size_t embeddingSize = m_embedding.getShape()[1];
  Tensor index = indices.broadcast({numElements, embeddingSize});

  // Gather along vocab dimension
  Tensor gathered = m_embedding.gather(0, index);

  Shape outShape = inpShape;
  outShape.push_back(embeddingSize);
  return gathered.reshape(outShape);
}

////////////////////////////////////////////////////////////////////////////////

LayerNorm::LayerNorm(Shape shape)
    : m_shape{std::move(shape)},
      m_numReduceElements{sizeFromShape(m_shape)},
      m_weight{
          Tensor{m_shape, std::vector<float>(m_numReduceElements, 1), true}},
      m_bias{Tensor{m_shape, std::vector<float>(m_numReduceElements, 0), true}},
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

////////////////////////////////////////////////////////////////////////////////

MLP::MLP(size_t featureSize, size_t projectionSize)
    : m_featureSize{featureSize},
      m_projectionSize{projectionSize},
      m_linear1{std::make_unique<Linear>(m_featureSize, m_projectionSize)},
      m_linear2{std::make_unique<Linear>(m_projectionSize, m_featureSize)} {
  insertSubModule(m_linear1.get());
  insertSubModule(m_linear2.get());
}

Tensor MLP::forward(Tensor x) {
  auto l1 = m_linear1->forward(x);
  auto act = l1.gelu();
  auto l2 = m_linear2->forward(act);
  return l2;
}
