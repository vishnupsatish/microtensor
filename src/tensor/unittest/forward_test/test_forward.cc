#include <iostream>

#include "tensor.h"

int main() {
  auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
  auto b = Tensor(Shape{2}, std::vector<float>{1.0, 2.0});

  auto c = a + b;

  c.dump_tensor(std::cout);

  return 0;
}