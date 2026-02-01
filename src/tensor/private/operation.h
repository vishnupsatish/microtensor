/**
 *    Author:  Vishnu Satish
 *    Created: Dec 26, 2025
 **/

#pragma once

#include <memory>
#include <vector>

#include "tensor_impl.h"

class Operation {
 public:
  virtual ~Operation() = default;

  Operation(std::vector<std::shared_ptr<TensorImpl>> parents,
            std::shared_ptr<TensorImpl> output)
      : m_parents(std::move(parents)), m_output(output) {}

  virtual std::vector<std::shared_ptr<TensorImpl>> backward(
      std::shared_ptr<TensorImpl> grad_output) = 0;

  std::vector<std::shared_ptr<TensorImpl>> m_parents;
  // Output owns the operation, so this must be non-owning.
  std::weak_ptr<TensorImpl> m_output;
};

std::shared_ptr<TensorImpl> multiply(std::shared_ptr<TensorImpl> a,
                                     std::shared_ptr<TensorImpl> b);

std::shared_ptr<TensorImpl> multiply(std::shared_ptr<TensorImpl> a, float cst);

std::shared_ptr<TensorImpl> divide(std::shared_ptr<TensorImpl> a,
                                   std::shared_ptr<TensorImpl> b);

std::shared_ptr<TensorImpl> divide(std::shared_ptr<TensorImpl> a, float denom);

std::shared_ptr<TensorImpl> divide(float num, std::shared_ptr<TensorImpl> b);

std::shared_ptr<TensorImpl> pow(std::shared_ptr<TensorImpl> a, float exp);

std::shared_ptr<TensorImpl> add(std::shared_ptr<TensorImpl> a,
                                std::shared_ptr<TensorImpl> b);

std::shared_ptr<TensorImpl> add(std::shared_ptr<TensorImpl> a, float b);

std::shared_ptr<TensorImpl> subtract(std::shared_ptr<TensorImpl> a,
                                     std::shared_ptr<TensorImpl> b);

std::shared_ptr<TensorImpl> broadcast(std::shared_ptr<TensorImpl> a,
                                      const Shape& target);

std::shared_ptr<TensorImpl> squeeze(std::shared_ptr<TensorImpl> a,
                                    std::vector<int> dimInputs);

std::shared_ptr<TensorImpl> unsqueeze(std::shared_ptr<TensorImpl> a,
                                      std::vector<int> dimInputs);

std::shared_ptr<TensorImpl> tanh(std::shared_ptr<TensorImpl> a);

std::shared_ptr<TensorImpl> exp(std::shared_ptr<TensorImpl> a);

std::shared_ptr<TensorImpl> log(std::shared_ptr<TensorImpl> a);

std::shared_ptr<TensorImpl> sqrt(std::shared_ptr<TensorImpl> a);

std::shared_ptr<TensorImpl> matmul(std::shared_ptr<TensorImpl> a,
                                   std::shared_ptr<TensorImpl> b);

std::shared_ptr<TensorImpl> permute(std::shared_ptr<TensorImpl> a,
                                    // TODO: change to int, should not be using
                                    // size_t in outside-facing API probably
                                    std::vector<size_t> dims);

std::shared_ptr<TensorImpl> reduceSum(std::shared_ptr<TensorImpl> a,
                                      std::vector<int> dimInputs,
                                      bool keep_dims);

std::shared_ptr<TensorImpl> reduceMax(std::shared_ptr<TensorImpl> a,
                                      int dimInput, bool keep_dim);

std::shared_ptr<TensorImpl> argmax(std::shared_ptr<TensorImpl> a, int dimInput,
                                   bool keep_dim);

std::shared_ptr<TensorImpl> argmax(std::shared_ptr<TensorImpl> a);

std::shared_ptr<TensorImpl> makeContiguous(std::shared_ptr<TensorImpl> a);

std::shared_ptr<TensorImpl> reshape(std::shared_ptr<TensorImpl> a,
                                    Shape newShape);

std::shared_ptr<TensorImpl> slice(std::shared_ptr<TensorImpl> a,
                                  std::vector<int> start, Shape size);

std::shared_ptr<TensorImpl> relu(std::shared_ptr<TensorImpl> a);

std::shared_ptr<TensorImpl> geluApprox(std::shared_ptr<TensorImpl> a);

std::shared_ptr<TensorImpl> softmax(std::shared_ptr<TensorImpl> a, int dim);

std::shared_ptr<TensorImpl> layerNorm(std::shared_ptr<TensorImpl> input,
                                      std::shared_ptr<TensorImpl> weight,
                                      std::shared_ptr<TensorImpl> bias,
                                      int numNormDims, float epsilon);

std::shared_ptr<TensorImpl> gather(std::shared_ptr<TensorImpl> a, int dimInput,
                                   std::shared_ptr<TensorImpl> index);

// Keeps upper triangular part of the matrix. Positive diagonal value: how many
// diagonals to exclude above the main diagonal.
std::shared_ptr<TensorImpl> triu(std::shared_ptr<TensorImpl> a,
                                 int diagonal = 0);

std::shared_ptr<TensorImpl> maskedFill(std::shared_ptr<TensorImpl> a,
                                       std::shared_ptr<TensorImpl> mask,
                                       float val);

void subtract_(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b);

void add_(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b);

void add_(std::shared_ptr<TensorImpl> a, float b);

void multiply_(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b);

void multiply_(std::shared_ptr<TensorImpl> a, float b);

void divide_(std::shared_ptr<TensorImpl> a, std::shared_ptr<TensorImpl> b);

void divide_(std::shared_ptr<TensorImpl> a, float b);

void pow_(std::shared_ptr<TensorImpl> a, float exp);
