/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#include "loss_function.h"

#include <functional>
#include <numeric>

Tensor meanSquaredError(Tensor inp, Tensor target) {
  std::vector<int> allDims;
  size_t elts = sizeFromShape(inp.getShape());
  for (int i = 0; i < inp.getShape().size(); ++i) {
    allDims.push_back(i);
  }
  return (inp - target).pow(2).reduceSum(allDims) / (float)elts;
}

Tensor crossEntropy(Tensor inp, Tensor target) {
  size_t vocabSize = inp.getShape().back();
  size_t numElements = sizeFromShape(target.getShape());

  Tensor inpReshape = inp.reshape(Shape{numElements, vocabSize});
  Tensor targetReshape = target.reshape(Shape{numElements, 1});

  auto mx = inpReshape.reduceMax(1, true);
  auto shifted = inpReshape - mx;
  auto logSumExp = (shifted.exp().reduceSum({1}, true)).log();
  auto logProbs = shifted - logSumExp;
  auto gathered = logProbs.gather(1, targetReshape);

  return (gathered * -1.0f).reduceSum({0, 1}, false) / (float)numElements;
}
