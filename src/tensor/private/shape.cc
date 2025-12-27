/**
 *    Author:  Vishnu Satish
 *    Created: Dec 27, 2025
 **/

#include "shape.h"

#include <numeric>
#include <set>
#include <vector>

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
  Shape result(std::max(shape_a.size(), shape_b.size()));
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
      return {};
    }
    if (it_a != shape_a.rend()) ++it_a;
    if (it_b != shape_b.rend()) ++it_b;
  }
  return result;
}