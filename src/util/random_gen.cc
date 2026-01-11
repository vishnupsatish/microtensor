/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#include "random_gen.h"

std::mt19937 RNG::instance;
bool RNG::seedSet = false;

void RNG::setSeed(size_t seed) {
  if (seedSet) {
    throw std::runtime_error("Cannot redefine seed");
  }
  instance.seed(seed);
}

std::mt19937& RNG::get() { return instance; }
