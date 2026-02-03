/**
 *    Author:  Vishnu Satish
 *    Created: Dec 27, 2025
 **/

#include "shape.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <set>
#include <vector>

void Shape::print(std::ostream& os) const {
  os << "[";
  for (size_t i = 0; i < size(); ++i) {
    os << (*this)[i];
    if (i < size() - 1) os << ", ";
  }
  os << "]";
}

std::vector<size_t> defaultStridesFromShape(const Shape& shape) {
  size_t ndim = shape.size();
  std::vector<size_t> strides(ndim, 1);
  if (ndim == 0) return strides;
  for (int i = ndim - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

size_t sizeFromShape(const Shape& shape) {
  return std::accumulate(shape.begin(), shape.end(), 1,
                         std::multiplies<size_t>());
}

std::optional<Shape> getBroadcastShape(const Shape& shape_a,
                                       const Shape& shape_b) {
  Shape result;
  result.reserve(std::max(shape_a.size(), shape_b.size()));
  auto it_a = shape_a.rbegin();
  auto it_b = shape_b.rbegin();
  while (it_a != shape_a.rend() || it_b != shape_b.rend()) {
    size_t dim_a = (it_a != shape_a.rend()) ? *it_a : 1;
    size_t dim_b = (it_b != shape_b.rend()) ? *it_b : 1;
    if (dim_a == dim_b) {
      result.push_back(dim_a);
    } else if (dim_a == 1) {
      result.push_back(dim_b);
    } else if (dim_b == 1) {
      result.push_back(dim_a);
    } else {
      // Invalid broadcast.
      return {};
    }
    if (it_a != shape_a.rend()) ++it_a;
    if (it_b != shape_b.rend()) ++it_b;
  }
  std::reverse(result.begin(), result.end());
  return result;
}

// Returns new shape of `a`, then `b`.
std::optional<std::pair<Shape, Shape>> getBroadcastShapesForMatmul(
    const Shape& shape_a, const Shape& shape_b) {
  assert(shape_a.size() >= 2);
  assert(shape_b.size() >= 2);
  auto rankA = shape_a.size();
  auto rankB = shape_b.size();
  Shape a_res;
  Shape b_res;
  a_res.reserve(std::max(shape_a.size(), shape_b.size()));
  b_res.reserve(std::max(shape_a.size(), shape_b.size()));
  // Check validity of last two dimensions. e.g., 5,6 x 6,2 is valid.
  if (shape_a[rankA - 1] != shape_b[rankB - 2]) {
    return {};
  }
  // Broadcast last N-2 dimensions.
  auto aShapeRm2 = Shape(shape_a.begin(), shape_a.end() - 2);
  auto bShapeRm2 = Shape(shape_b.begin(), shape_b.end() - 2);
  auto broadcasted = getBroadcastShape(aShapeRm2, bShapeRm2);
  if (!broadcasted) {
    return {};
  }
  // Add back the last two dimensions of a and b.
  auto aRet = *broadcasted;
  aRet.push_back(shape_a[rankA - 2]);
  aRet.push_back(shape_a[rankA - 1]);
  auto bRet = *broadcasted;
  bRet.push_back(shape_b[rankB - 2]);
  bRet.push_back(shape_b[rankB - 1]);
  return std::make_pair(aRet, bRet);
}
