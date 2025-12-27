/**
 *    Author:  Vishnu Satish
 *    Created: Dec 25, 2025
 **/

#include <iostream>
#include <memory>

#include "tensor/tensor.h"

int main() {
  auto a = Tensor(Shape{2, 2});
  a.fill_random();
  std::cout << "a: ";
  a.print(std::cout);
  std::cout << '\n';

  auto b = Tensor(Shape{2, 1, 2});
  b.fill_random();
  std::cout << "b: ";
  b.print(std::cout);
  std::cout << '\n';

  auto c = a + b;
  std::cout << "c: ";
  c.print(std::cout);
  std::cout << '\n';
  std::cout << "c shape: ";
  c.getShape().print(std::cout);
  std::cout << '\n';

  auto d = c * a;
  std::cout << "d: ";
  d.print(std::cout);
  std::cout << '\n';
  std::cout << "d shape: ";
  d.getShape().print(std::cout);
  std::cout << '\n';

  d.backward();

  std::cout << "\nprint gradients\n";

  std::cout << "d grad: ";
  d.getGrad().print(std::cout);
  std::cout << '\n';
  std::cout << "d grad shape: ";
  d.getGrad().getShape().print(std::cout);
  std::cout << '\n';

  std::cout << "c grad: ";
  c.getGrad().print(std::cout);
  std::cout << '\n';
  std::cout << "c grad shape: ";
  c.getGrad().getShape().print(std::cout);
  std::cout << '\n';

  std::cout << "b grad: ";
  b.getGrad().print(std::cout);
  std::cout << '\n';
  std::cout << "b grad shape: ";
  b.getGrad().getShape().print(std::cout);
  std::cout << '\n';

  std::cout << "a grad: ";
  a.getGrad().print(std::cout);
  std::cout << '\n';
  std::cout << "a grad shape: ";
  a.getGrad().getShape().print(std::cout);
  std::cout << '\n';

  return 0;
}
