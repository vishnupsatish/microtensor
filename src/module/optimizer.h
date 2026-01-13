/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#include <vector>

#include "tensor.h"

class Optimizer {
 protected:
  std::vector<Tensor> m_parameters;

 public:
  Optimizer(std::vector<Tensor> parameters);

  void zeroGrad();
  virtual void step() = 0;
};

class SGD : public Optimizer {
  float m_learningRate = 0.001;

 public:
  using Optimizer::Optimizer;
  SGD(std::vector<Tensor> parameters, float learningRate);

  void step() override;
};

class AdamW : public Optimizer {
  float m_learningRate = 0.001;
  float m_beta1 = 0.9;
  float m_beta2 = 0.999;
  float m_epsilon = 1e-8;
  float m_weightDecay = 0.01;
  float m_t = 1;

  std::vector<Tensor> m_moment1;
  std::vector<Tensor> m_moment2;

 public:
  AdamW(std::vector<Tensor> parameters);
  AdamW(std::vector<Tensor> parameters, float learningRate, float beta1,
        float beta2, float epsilon, float weightDecay);

  void step() override;
};
