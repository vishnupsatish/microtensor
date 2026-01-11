/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#include "loss_function.h"

Tensor meanSquaredError(Tensor inp, Tensor target) {
  std::vector<int> allDims;
  size_t elts = sizeFromShape(inp.getShape());
  for (int i = 0; i < inp.getShape().size(); ++i) {
    allDims.push_back(i);
  }
  return (inp - target).pow(2).reduceSum(allDims) / elts;
}
