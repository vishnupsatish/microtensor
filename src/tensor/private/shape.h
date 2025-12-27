/**
 *    Author:  Vishnu Satish
 *    Created: Dec 27, 2025
 **/

#pragma once

#include <set>
#include <vector>

using Shape = std::vector<size_t>;

std::vector<size_t> defaultStridesFromShape(const Shape& shape);

size_t sizeFromShape(const Shape& shape);

std::optional<Shape> getBroadcastShape(const Shape& shape1,
                                       const Shape& shape2);
