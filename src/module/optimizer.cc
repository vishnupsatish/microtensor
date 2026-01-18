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
      // TODO: argh... ugly
      param.getGrad().fillRandom([]() { return 0; });
    }
  }
}

////////////////////////////////////////////////////////////////////////////////

SGD::SGD(std::vector<Tensor> parameters, float learningRate)
    : Optimizer{std::move(parameters)}, m_learningRate{learningRate} {}

void SGD::step() {
  NoGrad guard;
  for (auto& param : m_parameters) {
    param -= param.getGrad() * m_learningRate;
  }
}

////////////////////////////////////////////////////////////////////////////////

std::vector<Tensor> initializeMoment(const std::vector<Tensor>& parameters) {
  std::vector<Tensor> moments;
  for (auto& p : parameters) {
    size_t elements = sizeFromShape(p.getShape());
    moments.push_back(Tensor{p.getShape(), std::vector<float>(elements, 0.0f)});
  }
  return moments;
}

AdamW::AdamW(std::vector<Tensor> parameters)
    : Optimizer{std::move(parameters)},
      m_moment1{initializeMoment(m_parameters)},
      m_moment2{initializeMoment(m_parameters)} {}

AdamW::AdamW(std::vector<Tensor> parameters, float learningRate, float beta1,
             float beta2, float epsilon, float weightDecay)
    : Optimizer{std::move(parameters)},
      m_learningRate{learningRate},
      m_beta1{beta1},
      m_beta2{beta2},
      m_epsilon{epsilon},
      m_weightDecay{weightDecay},
      m_moment1{initializeMoment(m_parameters)},
      m_moment2{initializeMoment(m_parameters)} {}

void AdamW::step() {
  // Note: following the algorithm provided in
  // https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html.
  // I have no idea how this works.
  NoGrad guard;
  float m_b1Corr = std::pow(m_beta1, m_t);
  float m_b2Corr = std::pow(m_beta2, m_t);
  for (int i = 0; i < m_parameters.size(); ++i) {
    auto& param = m_parameters[i];
    auto& m1 = m_moment1[i];
    auto& m2 = m_moment2[i];
    auto g = param.getGrad();
    param -= param * m_learningRate * m_weightDecay;
    // TODO: make these in-place operations.
    m1 = m1 * m_beta1 + g * (1 - m_beta1);
    m2 = m2 * m_beta2 + g.pow(2) * (1 - m_beta2);
    auto m1Hat = m1 / (1 - m_b1Corr);
    auto m2Hat = m2 / (1 - m_b2Corr);
    param -= (m1Hat * m_learningRate) / (m2Hat.pow(0.5) + m_epsilon);
  }
  ++m_t;
}
