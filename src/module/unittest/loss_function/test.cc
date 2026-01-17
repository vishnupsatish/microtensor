#include <iostream>
#include <vector>

#include "loss_function.h"
#include "tensor.h"

int main() {
  // Test 1
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto b = Tensor(Shape{2, 2}, std::vector<float>{1.5, 1.5, 3.5, 3.5});
    meanSquaredError(a, b).dumpTensor(std::cout);
  }

  // Test 2
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0}, true);
    auto b = Tensor(Shape{2, 2}, std::vector<float>{1.5, 1.5, 3.5, 3.5});
    auto loss = meanSquaredError(a, b);
    loss.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 3
  {
    auto logits =
        Tensor(Shape{2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto targets = Tensor(Shape{2}, std::vector<float>{2.0, 1.0});
    crossEntropy(logits, targets).dumpTensor(std::cout);
  }

  // Test 4
  {
    auto logits = Tensor(
        Shape{2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, true);
    auto targets = Tensor(Shape{2}, std::vector<float>{2.0, 1.0});
    auto loss = crossEntropy(logits, targets);
    loss.backward();
    logits.getGrad().dumpTensor(std::cout);
  }

  // Test 5
  {
    // [Batch=1, Seq=2, Vocab=3]
    auto logits = Tensor(Shape{1, 2, 3},
                         std::vector<float>{0.1, 0.2, 0.7, 0.9, 0.05, 0.05});
    auto targets = Tensor(Shape{1, 2}, std::vector<float>{2.0, 0.0});
    crossEntropy(logits, targets).dumpTensor(std::cout);
  }

  // Test 6
  {
    auto logits =
        Tensor(Shape{1, 2, 3},
               std::vector<float>{0.1, 0.2, 0.7, 0.9, 0.05, 0.05}, true);
    auto targets = Tensor(Shape{1, 2}, std::vector<float>{2.0, 0.0});
    auto loss = crossEntropy(logits, targets);
    loss.backward();
    logits.getGrad().dumpTensor(std::cout);
  }

  return 0;
}
