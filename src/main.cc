/**
 *    Author:  Vishnu Satish
 *    Created: Dec 25, 2025
 **/

#include <iostream>
#include <memory>

#include "grad_mode.h"
#include "loss_function.h"
#include "models/xor.h"
#include "module.h"
#include "optimizer.h"
#include "random_gen.h"
#include "tensor.h"

int main() {
  RNG::setSeed(42);
  auto model = XOR{};
  auto opt = SGD{model.getParameters(), 0.01};

  Tensor x{Shape{4, 2}, std::vector<float>{0, 0, 0, 1, 1, 0, 1, 1}};
  Tensor y{Shape{4, 1}, std::vector<float>{0, 1, 1, 0}};

  std::cout << "Training...\n";
  for (int epoch = 1; epoch <= 5000; ++epoch) {
    opt.zeroGrad();
    auto pred = model.forward(x);
    auto loss = meanSquaredError(pred, y);
    loss.backward();
    opt.step();
  }

  std::cout << "\nInference:\n";
  {
    NoGrad guard;
    auto final_pred = model.forward(x);
    std::cout << "Inputs:\n";
    x.print(std::cout);
    std::cout << "\nPredictions:\n";
    final_pred.print(std::cout);
    std::cout << "\nTargets:\n";
    y.print(std::cout);
    std::cout << "\n";
  }
}
