/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#include <random>

// Object that stores a random number generator that can be seeded once.
// Note: not thread-safe, but it is probably okay.
class RNG {
  static std::mt19937 instance;
  static bool seedSet;

 public:
  static void setSeed(size_t seed);
  static std::mt19937& get();
};
