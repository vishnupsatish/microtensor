#include <iostream>
#include <vector>

#include "tensor.h"

int main() {
  // Test 1
  {
    auto a = Tensor(Shape{3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6});
    a.reduceSum({0, 1}, false).dumpTensor(std::cout);
  }

  // Test 2
  {
    auto a = Tensor(Shape{3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6});
    a.reduceSum({0}, false).dumpTensor(std::cout);
  }

  // Test 3
  {
    auto a = Tensor(Shape{3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6});
    a.reduceSum({1}, true).dumpTensor(std::cout);
  }

  // Test 4
  {
    auto a = Tensor(Shape{}, std::vector<float>{5.0f});
    a.broadcast(Shape{2, 3}).dumpTensor(std::cout);
  }

  // Test 5
  {
    auto a = Tensor(Shape{1, 3}, std::vector<float>{1, 2, 3});
    a.broadcast(Shape{2, 3}).dumpTensor(std::cout);
  }

  // Test 6
  {
    auto a = Tensor(Shape{3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6});
    auto b = a.reduceSum({0, 1}, false);
    b.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 7
  {
    auto a = Tensor(Shape{}, std::vector<float>{5.0f});
    auto b = a.broadcast(Shape{2, 3});
    b.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 8
  {
    auto a = Tensor(Shape{2, 1}, std::vector<float>{1, 2});
    auto b = a.broadcast(Shape{2, 5});
    auto c = b.reduceSum({1}, false);
    c.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 9
  {
    auto a = Tensor(Shape{2, 3, 4}, std::vector<float>(24, 1.0f));
    auto b = a.reduceSum({2}, true);   // (2, 3, 1)
    auto c = b.reduceSum({0}, false);  // (3, 1)
    c.reduceSum({0}, false).dumpTensor(std::cout);
  }

  // Test 10
  {
    auto a = Tensor(Shape{2, 3, 4, 2}, std::vector<float>(48, 1.0f));
    a.reduceSum({0, 2}, false).dumpTensor(std::cout);  // (3, 2)
  }

  // Test 11
  {
    auto a = Tensor(Shape{3, 1}, std::vector<float>{1, 2, 3});
    auto b = a.broadcast(Shape{3, 4});
    b.reduceSum({0}, false).dumpTensor(std::cout);  // (4)
  }

  // Test 12
  {
    auto a = Tensor(Shape{2, 2, 2}, std::vector<float>(8, 1.0f));
    auto b = a.reduceSum({0, 2}, false);  // (2)
    auto c = b.reduceSum({0}, false);     // scalar
    c.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 13
  {
    auto a = Tensor(Shape{3, 2}, std::vector<float>{1, 6, 3, 4, 5, 2});
    a.reduceMax(0, false).dumpTensor(std::cout);
  }

  // Test 14
  {
    auto a = Tensor(Shape{3, 2}, std::vector<float>{1, 6, 3, 4, 5, 2});
    a.reduceMax(1, true).dumpTensor(std::cout);
  }

  // Test 15
  {
    auto a = Tensor(Shape{3, 2}, std::vector<float>{1, 6, 3, 4, 5, 2});
    auto b = a.reduceMax(0, false);
    b.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 16
  {
    auto a = Tensor(Shape{3, 2}, std::vector<float>{1, 2, 7, 4, 5, 8});
    auto b = a.reduceMax(0, false);
    b.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 17
  {
    auto a = Tensor(Shape{3, 2}, std::vector<float>{5, 2, 5, 4, 1, 4});
    a.reduceMax(0, false).dumpTensor(std::cout);
  }

  // Test 18
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1, 2, 3, 4});
    auto b = Tensor(Shape{2, 2}, std::vector<float>{10, 20, 30, 40});
    auto c = Tensor(Shape{2}, std::vector<float>{0.5, 0.5});
    auto res = (a + b).reduceMax(0, false) * c;
    res.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
    c.getGrad().dumpTensor(std::cout);
  }

  // Test 19
  {
    auto a = Tensor(Shape{3, 2}, std::vector<float>{1, 5, 6, 3, 2, 4});
    auto b = Tensor(Shape{3}, std::vector<float>{1, 1, 1});
    auto res = (a.reduceMax(1, false) - b).pow(2.0);
    res.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  return 0;
}
