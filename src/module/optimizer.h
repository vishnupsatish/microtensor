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
