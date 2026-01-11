/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#include <memory>

#include "grad_mode.h"
#include "linear.h"
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
