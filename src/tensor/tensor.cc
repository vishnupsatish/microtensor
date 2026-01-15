/**
 *    Author:  Vishnu Satish
 *    Created: Dec 25, 2025
 **/

#include "tensor.h"

#include <iostream>
#include <memory>
#include <vector>

// TODO: this might be a problem... exposing all these operation methods and
// also exposing TensorImpl
#include "private/operation.h"
#include "private/tensor_impl.h"

Tensor::Tensor() : m_impl{nullptr} {}

Tensor::Tensor(const Shape& shape, bool requiresGrad)
    : m_impl{std::make_shared<TensorImpl>(shape, requiresGrad)} {}

Tensor::Tensor(const Tensor& other) : m_impl{other.m_impl} {}

Tensor::Tensor(std::shared_ptr<TensorImpl> otherImpl) : m_impl{otherImpl} {}

Tensor::Tensor(const Shape& shape, const std::vector<float>& data,
               bool requiresGrad)
    : m_impl{std::make_shared<TensorImpl>(shape, data, requiresGrad)} {}

bool Tensor::isValid() const { return m_impl != nullptr; }

void Tensor::print(std::ostream& os) { m_impl->print(os); }

Tensor Tensor::getGrad() const { return Tensor{m_impl->getGrad()}; }

Shape Tensor::getShape() const { return m_impl->m_shape; }

Tensor Tensor::operator+(const Tensor& other) {
  return Tensor{add(m_impl, other.m_impl)};
}

Tensor Tensor::operator+(float other) { return Tensor{add(m_impl, other)}; }

Tensor Tensor::operator-(const Tensor& other) {
  return Tensor{subtract(m_impl, other.m_impl)};
}

Tensor Tensor::operator*(const Tensor& other) {
  return Tensor{multiply(m_impl, other.m_impl)};
}

Tensor Tensor::operator*(float other) {
  return Tensor{multiply(m_impl, other)};
}

Tensor Tensor::operator/(const Tensor& other) {
  return Tensor{divide(m_impl, other.m_impl)};
}

Tensor Tensor::operator/(float other) { return Tensor{divide(m_impl, other)}; }

Tensor Tensor::pow(float exp) { return Tensor{::pow(m_impl, exp)}; }

Tensor Tensor::tanh() { return Tensor{::tanh(m_impl)}; }

Tensor Tensor::exp() { return Tensor{::exp(m_impl)}; }

Tensor Tensor::log() { return Tensor{::log(m_impl)}; }

Tensor Tensor::matmul(const Tensor& other) {
  return Tensor{::matmul(m_impl, other.m_impl)};
}

Tensor Tensor::permute(const std::vector<size_t>& dims) {
  return Tensor{::permute(m_impl, dims)};
}

Tensor Tensor::broadcast(const Shape& target) {
  return Tensor{::broadcast(m_impl, target)};
}

Tensor Tensor::reduceSum(const std::vector<int>& dims, bool keep_dims) {
  return Tensor{::reduceSum(m_impl, dims, keep_dims)};
}

Tensor Tensor::reduceMax(int dim, bool keep_dim) {
  return Tensor{::reduceMax(m_impl, dim, keep_dim)};
}

Tensor Tensor::reshape(const Shape& shape) {
  return Tensor{::reshape(m_impl, shape)};
}

Tensor Tensor::makeContiguous() { return Tensor{::makeContiguous(m_impl)}; }

Tensor Tensor::slice(const std::vector<int>& start, const Shape& size) {
  return Tensor{::slice(m_impl, start, size)};
}

Tensor Tensor::relu() { return Tensor(::relu(m_impl)); }

Tensor Tensor::softmax(int dim) { return Tensor(::softmax(m_impl, dim)); }

Tensor& Tensor::operator-=(const Tensor& other) {
  subtract_(m_impl, other.m_impl);
  return *this;
}

void Tensor::backward() { m_impl->backward(); }

void Tensor::dumpTensor(std::ostream& os) { m_impl->dumpTensor(os); }
