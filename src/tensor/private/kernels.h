/**
 *    Author:  Vishnu Satish
 *    Created: Jan 11, 2026
 **/

#pragma once

#include <cassert>

#include "tensor.h"

template <typename F>
void elementwiseBinaryKernel(std::shared_ptr<TensorImpl> res,
                             std::shared_ptr<TensorImpl> a,
                             std::shared_ptr<TensorImpl> b, F&& binary_fn);

template <typename F>
void elementwiseUnaryKernel(std::shared_ptr<TensorImpl> res,
                            std::shared_ptr<TensorImpl> a, F&& unary_fn);

// Assumes last two dimensions of c, a are contiguous in memory and the last two
// dimensions of the tranpose of b are contiguous.
void matmulContiguousBatched(std::shared_ptr<TensorImpl> c,
                             std::shared_ptr<TensorImpl> a,
                             std::shared_ptr<TensorImpl> b);

#include "kernels.ipp"
