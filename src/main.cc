/**
 *    Author:  Vishnu Satish
 *    Created: Dec 25, 2025
 **/

#include <iostream>
#include <memory>

#include "tensor/tensor.h"

int main() {
  auto a = Tensor(std::vector<size_t>{2, 2});
  a.fill_random();
  a.print(std::cout);
  std::cout << '\n';

  auto b = Tensor(std::vector<size_t>{2, 2});
  b.fill_random();
  b.print(std::cout);
  std::cout << '\n';

  auto c = a + b;
  c.print(std::cout);
  std::cout << '\n';

  auto d = c * a;
  d.print(std::cout);
  std::cout << '\n';

  d.backward();

  std::cout << "print gradients\n";

  c.getGrad().print(std::cout);
  std::cout << '\n';
  b.getGrad().print(std::cout);
  std::cout << '\n';
  a.getGrad().print(std::cout);
  std::cout << '\n';

  return 0;
}
