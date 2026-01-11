/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#include "optimizer.h"

#include "grad_mode.h"

Optimizer::Optimizer(std::vector<Tensor> parameters)
    : m_parameters{std::move(parameters)} {}

void Optimizer::zeroGrad() {
  for (auto param : m_parameters) {
    if (param.getGrad().isValid()) {
      // argh... ugly
      param.getGrad().fillRandom([]() { return 0; });
    }
  }
}

SGD::SGD(std::vector<Tensor> parameters, float learningRate)
    : Optimizer{std::move(parameters)}, m_learningRate{learningRate} {}

void SGD::step() {
  NoGrad guard;
  for (auto param : m_parameters) {
    param -= param.getGrad() * m_learningRate;
  }
}
