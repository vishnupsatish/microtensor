/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#include <memory>

#include "grad_mode.h"
#include "layers.h"
#include "loss_function.h"
#include "optimizer.h"
#include "random_gen.h"
#include "tensor.h"

class XOR : public Module {
  std::unique_ptr<Linear> l1;
  std::unique_ptr<Linear> l2;

 public:
  XOR()
      : l1{std::make_unique<Linear>(2, 2)}, l2{std::make_unique<Linear>(2, 1)} {
    insertSubModule(l1.get());
    insertSubModule(l2.get());
  }
  Tensor forward(Tensor inp) {
    inp = l1->forward(inp);
    // Note: in certain situations (depending on the random initial values),
    // having a .relu() here does not work.
    inp = inp.tanh();
    inp = l2->forward(inp);
    return inp;
  }
};

int main() {
  RNG::setSeed(42);
  auto model = XOR{};
  // We can use AdamW now!!!
  auto opt = AdamW{model.getParameters(), 0.01, 0.9, 0.999, 1e-8, 0.01};
  // auto opt = SGD{model.getParameters(), 0.01};

  Tensor x{Shape{4, 2}, std::vector<float>{0, 0, 0, 1, 1, 0, 1, 1}};
  Tensor y{Shape{4, 1}, std::vector<float>{0, 1, 1, 0}};

  std::cout << "Training...\n";
  for (int epoch = 1; epoch <= 1000; ++epoch) {
    // Can pick a random data point instead.
    for (int i = 0; i < 4; ++i) {
      Tensor x_single = x.slice({i, 0}, {1, 2});
      Tensor y_single = y.slice({i, 0}, {1, 1});

      opt.zeroGrad();
      auto pred = model.forward(x_single);
      auto loss = meanSquaredError(pred, y_single);
      loss.backward();
      opt.step();
    }
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
