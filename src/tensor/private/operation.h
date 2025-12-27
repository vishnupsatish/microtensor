/**
 *    Author:  Vishnu Satish
 *    Created: Dec 26, 2025
 **/

#pragma once

#include <memory>
#include <vector>

#include "tensor_impl.h"

// Stores all the state necessary to perform an operation backwards (is a
// function object, in a sense).
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

std::shared_ptr<TensorImpl> add(std::shared_ptr<TensorImpl> a,
                                std::shared_ptr<TensorImpl> b,
                                bool track_creator = true);
