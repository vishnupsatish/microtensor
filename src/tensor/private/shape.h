/**
 *    Author:  Vishnu Satish
 *    Created: Dec 27, 2025
 **/

#pragma once

#include <iostream>
#include <set>
#include <vector>

struct Shape : std::vector<size_t> {
  using std::vector<size_t>::vector;

  void print(std::ostream& os) const;
};

std::vector<size_t> defaultStridesFromShape(const Shape& shape);

size_t sizeFromShape(const Shape& shape);

std::optional<Shape> getBroadcastShape(const Shape& shape1,
                                       const Shape& shape2);

std::optional<std::pair<Shape, Shape>> getBroadcastShapesForMatmul(
    const Shape& shape_a, const Shape& shape_b);
