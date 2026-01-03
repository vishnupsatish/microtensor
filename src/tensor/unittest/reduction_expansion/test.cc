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

  return 0;
}
