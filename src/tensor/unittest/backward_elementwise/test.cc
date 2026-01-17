#include <iostream>
#include <vector>

#include "tensor.h"

int main() {
  // Test 1
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0}, true);
    auto b = Tensor(Shape{2}, std::vector<float>{1.0, 2.0}, true);
    auto c = a + b;
    c.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  // Test 2
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0}, true);
    auto b = Tensor(Shape{2}, std::vector<float>{10.0, 20.0}, true);
    auto c = a * b;
    c.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  // Test 3
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0}, true);
    auto b = Tensor(Shape{2}, std::vector<float>{1.0, 2.0}, true);
    auto c = Tensor(Shape{2, 2}, std::vector<float>{5.0, 5.0, 5.0, 5.0}, true);
    auto d = a * b + c;
    d.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
    c.getGrad().dumpTensor(std::cout);
  }

  // Test 4
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0}, true);
    auto b = Tensor(Shape{2}, std::vector<float>{0.5, 0.5}, true);
    auto c = Tensor(Shape{1}, std::vector<float>{2.0}, true);
    auto res = (a * b) + (a * c);
    res.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
    c.getGrad().dumpTensor(std::cout);
  }

  // Test 5
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0}, true);
    auto b = Tensor(Shape{2}, std::vector<float>{1.0, 2.0}, true);
    auto c = Tensor(Shape{1}, std::vector<float>{10.0}, true);
    auto res = (a + b) * (a + c) + b;
    res.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
    c.getGrad().dumpTensor(std::cout);
  }

  // Test 6
  {
    auto a = Tensor(Shape{1, 3}, std::vector<float>{1.0, 2.0, 3.0}, true);
    auto b = Tensor(Shape{3, 1}, std::vector<float>{10.0, 20.0, 30.0}, true);
    auto c = Tensor(Shape{1, 1}, std::vector<float>{5.0}, true);
    auto res = (a + b) * (a + c) * (b + c);
    res.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
    c.getGrad().dumpTensor(std::cout);
  }

  // Test 7
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{-1.0, 0.0, 1.0, 2.0}, true);
    auto b = a.tanh();
    b.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 8
  {
    auto a =
        Tensor(Shape{2, 2}, std::vector<float>{10.0, 20.0, 30.0, 40.0}, true);
    auto b = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0}, true);
    auto res = a - b;
    res.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  // Test 9
  {
    auto a =
        Tensor(Shape{2, 2}, std::vector<float>{10.0, 20.0, 30.0, 40.0}, true);
    auto b = Tensor(Shape{2, 2}, std::vector<float>{2.0, 4.0, 5.0, 8.0}, true);
    auto res = a / b;
    res.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  // Test 10
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0}, true);
    auto res = a.pow(3.0);
    res.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 11
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0}, true);
    auto b = Tensor(Shape{2, 2}, std::vector<float>{5.0, 6.0, 7.0, 8.0}, true);
    auto c = Tensor(Shape{1}, std::vector<float>{2.0}, true);
    auto res = (a + b) / (a * c) - b.pow(2.0);
    res.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
    c.getGrad().dumpTensor(std::cout);
  }

  // Test 12
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{0.5, 1.0, 1.5, 2.0}, true);
    auto b = Tensor(Shape{2}, std::vector<float>{1.0, 2.0}, true);
    auto res = (a / b).tanh() + (a - b) * a;
    res.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  // Test 13
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{0.5, 1.0, 1.5, 2.0}, true);
    auto b = a.exp();
    b.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 14
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0}, true);
    auto b = a.log();
    b.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 15
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0}, true);
    auto b = Tensor(Shape{2}, std::vector<float>{1.0, 2.0}, true);
    auto res = (a / b).exp() + (a * b).log();
    res.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  // Test 16
  {
    auto a =
        Tensor(Shape{2, 2}, std::vector<float>{-2.0, -1.0, 1.0, 2.0}, true);
    auto b = a.gelu();
    b.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  return 0;
}
