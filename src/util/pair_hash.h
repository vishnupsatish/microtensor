/**
 *    Author:  Vishnu Satish
 *    Created: Jan 29, 2026
 **/

#pragma once

#include <functional>
#include <utility>

// Boost's `hash_combine` pattern.
struct PairHash {
  size_t operator()(const std::pair<int, int>& p) const noexcept {
    size_t h1 = std::hash<int>{}(p.first);
    size_t h2 = std::hash<int>{}(p.second);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};
