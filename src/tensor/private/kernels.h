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

void matmulKernel(float* ptr_c, const float* ptr_a, const float* ptr_b,
                  size_t m, size_t k, size_t n, size_t stride_a_m,
                  size_t stride_a_k,  // Strides for A's last 2 dims
                  size_t stride_b_k,
                  size_t stride_b_n,  // Strides for B's last 2 dims
                  size_t stride_c_m,
                  size_t stride_c_n  // Strides for C's last 2 dims
);

void matmulBatched(std::shared_ptr<TensorImpl> c, std::shared_ptr<TensorImpl> a,
                   std::shared_ptr<TensorImpl> b);

#include "kernels.ipp"
