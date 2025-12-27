/**
 *    Author:  Vishnu Satish
 *    Created: Dec 25, 2025
 **/

#include <iostream>
#include <memory>

#include "tensor/tensor.h"

int main() {
  auto a = std::make_shared<Tensor>(std::vector<size_t>{2, 2});
  a->fill_random();
  a->print(std::cout);
  std::cout << '\n';

  auto b = std::make_shared<Tensor>(std::vector<size_t>{2, 2});
  b->fill_random();
  b->print(std::cout);
  std::cout << '\n';

  auto c = (*a) + (*b);
  c->print(std::cout);
  std::cout << '\n';

  c->backward();
  a->getGrad()->print(std::cout);
  std::cout << '\n';
  b->getGrad()->print(std::cout);
  std::cout << '\n';
  c->getGrad()->print(std::cout);
  std::cout << '\n';

  return 0;
}
