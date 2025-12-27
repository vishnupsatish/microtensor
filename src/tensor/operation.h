/**
 *    Author:  Vishnu Satish
 *    Created: Dec 26, 2025
 **/

#pragma once

#include <memory>
#include <vector>

class Tensor;

class Operation {
 public:
  virtual std::vector<std::shared_ptr<Tensor>> backward(
      std::shared_ptr<Tensor> grad_output) = 0;

  std::vector<std::shared_ptr<Tensor>> m_parents;
  std::weak_ptr<Tensor> m_output;
};

class MulOp : public Operation {
 public:
  std::vector<std::shared_ptr<Tensor>> backward(
      std::shared_ptr<Tensor> grad_output) override;
};

class AddOp : public Operation {
 public:
  std::vector<std::shared_ptr<Tensor>> backward(
      std::shared_ptr<Tensor> grad_output) override;
};
