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
                                     std::shared_ptr<TensorImpl> b,
                                     bool track_creator = true);

std::shared_ptr<TensorImpl> multiply(std::shared_ptr<TensorImpl> a, float cst,
                                     bool track_creator = true);

std::shared_ptr<TensorImpl> divide(std::shared_ptr<TensorImpl> a,
                                   std::shared_ptr<TensorImpl> b,
                                   bool track_creator = true);

std::shared_ptr<TensorImpl> divide(std::shared_ptr<TensorImpl> a, float denom,
                                   bool track_creator = true);

std::shared_ptr<TensorImpl> divide(float num, std::shared_ptr<TensorImpl> b,
                                   bool track_creator = true);

std::shared_ptr<TensorImpl> pow(std::shared_ptr<TensorImpl> a, float exp,
                                bool track_creator = true);

std::shared_ptr<TensorImpl> add(std::shared_ptr<TensorImpl> a,
                                std::shared_ptr<TensorImpl> b,
                                bool track_creator = true);

std::shared_ptr<TensorImpl> subtract(std::shared_ptr<TensorImpl> a,
                                     std::shared_ptr<TensorImpl> b,
                                     bool track_creator = true);

std::shared_ptr<TensorImpl> broadcast(std::shared_ptr<TensorImpl> a,
                                      const Shape& target,
                                      bool track_creator = true);

std::shared_ptr<TensorImpl> squeeze(std::shared_ptr<TensorImpl> a,
                                    std::vector<int> dimInputs,
                                    bool track_creator = true);

std::shared_ptr<TensorImpl> unsqueeze(std::shared_ptr<TensorImpl> a,
                                      std::vector<int> dimInputs,
                                      bool track_creator = true);

std::shared_ptr<TensorImpl> tanh(std::shared_ptr<TensorImpl> a,
                                 bool track_creator = true);

std::shared_ptr<TensorImpl> exp(std::shared_ptr<TensorImpl> a,
                                bool track_creator = true);

std::shared_ptr<TensorImpl> log(std::shared_ptr<TensorImpl> a,
                                bool track_creator = true);

std::shared_ptr<TensorImpl> matmul(std::shared_ptr<TensorImpl> a,
                                   std::shared_ptr<TensorImpl> b,
                                   bool track_creator = true);

std::shared_ptr<TensorImpl> permute(std::shared_ptr<TensorImpl> a,
                                    // TODO: change to int, should not be using
                                    // size_t in outside-facing API probably
                                    std::vector<size_t> dims,
                                    bool track_creator = true);

std::shared_ptr<TensorImpl> reduceSum(std::shared_ptr<TensorImpl> a,
                                      std::vector<int> dimInputs,
                                      bool keep_dims,
                                      bool track_creator = true);

std::shared_ptr<TensorImpl> reduceMax(std::shared_ptr<TensorImpl> a,
                                      int dimInput, bool keep_dim,
                                      bool track_creator = true);

std::shared_ptr<TensorImpl> makeContiguous(std::shared_ptr<TensorImpl> a,
                                           bool track_creator = true);

std::shared_ptr<TensorImpl> reshape(std::shared_ptr<TensorImpl> a,
                                    Shape newShape, bool track_creator = true);

std::shared_ptr<TensorImpl> slice(std::shared_ptr<TensorImpl> a,
                                  std::vector<int> start, Shape size,
                                  bool track_creator = true);
