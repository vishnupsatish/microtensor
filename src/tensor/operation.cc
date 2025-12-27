/**
 *    Author:  Vishnu Satish
 *    Created: Dec 26, 2025
 **/

#include "operation.h"

#include "tensor.h"

std::vector<std::shared_ptr<Tensor>> MulOp::backward(
    std::shared_ptr<Tensor> grad_output) {
  auto grad_op1 = (*grad_output) * (*m_parents[1]);
  auto grad_op2 = (*grad_output) * (*m_parents[0]);

  return {grad_op1, grad_op2};
}

std::vector<std::shared_ptr<Tensor>> AddOp::backward(
    std::shared_ptr<Tensor> grad_output) {
  return {grad_output, grad_output};
}
