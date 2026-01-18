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
    auto a = Tensor(Shape{3, 2}, std::vector<float>{1, 2, 3, 4, 5, 6}, true);
    auto b = a.reduceSum({0, 1}, false);
    b.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 7
  {
    auto a = Tensor(Shape{}, std::vector<float>{5.0f}, true);
    auto b = a.broadcast(Shape{2, 3});
    b.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 8
  {
    auto a = Tensor(Shape{2, 1}, std::vector<float>{1, 2}, true);
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
    auto a = Tensor(Shape{2, 2, 2}, std::vector<float>(8, 1.0f), true);
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
    auto a = Tensor(Shape{3, 2}, std::vector<float>{1, 6, 3, 4, 5, 2}, true);
    auto b = a.reduceMax(0, false);
    b.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 16
  {
    auto a = Tensor(Shape{3, 2}, std::vector<float>{1, 2, 7, 4, 5, 8}, true);
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
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1, 2, 3, 4}, true);
    auto b = Tensor(Shape{2, 2}, std::vector<float>{10, 20, 30, 40}, true);
    auto c = Tensor(Shape{2}, std::vector<float>{0.5, 0.5}, true);
    auto res = (a + b).reduceMax(0, false) * c;
    res.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
    c.getGrad().dumpTensor(std::cout);
  }

  // Test 19
  {
    auto a = Tensor(Shape{3, 2}, std::vector<float>{1, 5, 6, 3, 2, 4}, true);
    auto b = Tensor(Shape{3}, std::vector<float>{1, 1, 1}, true);
    auto res = (a.reduceMax(1, false) - b).pow(2.0);
    res.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  // Test 20
  {
    auto a = Tensor(Shape{3}, std::vector<float>{1.0f, 2.0f, 3.0f});
    a.softmax(0).dumpTensor(std::cout);
  }

  // Test 21
  {
    auto a = Tensor(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    a.softmax(0).dumpTensor(std::cout);
  }

  // Test 22
  {
    auto a = Tensor(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    a.softmax(1).dumpTensor(std::cout);
  }

  // Test 23
  {
    auto a = Tensor(Shape{3}, std::vector<float>{1.0f, 2.0f, 3.0f}, true);
    auto b = a.softmax(0);
    auto c = b * Tensor(Shape{3}, std::vector<float>{0.1f, 0.2f, 0.3f});
    auto d = c.reduceSum({0}, false);
    d.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 24
  {
    auto a = Tensor(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}, true);
    auto b = a.softmax(1);
    auto c = b.reduceSum({0, 1}, false);
    c.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 25
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1, 2, 3, 4}, true);
    auto b = Tensor(Shape{2, 2}, std::vector<float>{0.5, 1.5, 2.5, 3.5}, true);
    auto res = (a + b).softmax(1).log().reduceSum({0, 1}, false);
    res.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  // Test 26
  {
    auto a = Tensor(Shape{3, 3}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9});
    a.triu(0).dumpTensor(std::cout);
    a.triu(1).dumpTensor(std::cout);
    a.triu(-1).dumpTensor(std::cout);
  }

  // Test 27
  {
    auto a = Tensor(Shape{2, 3, 3}, std::vector<float>(18, 1.0f), true);
    auto b = a.triu(1);
    auto c = b.reduceSum({0, 1, 2}, false);
    c.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 28
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1, 2, 3, 4}, true);
    auto mask = Tensor(Shape{2, 2}, std::vector<float>{0, 1, 1, 0});
    auto b = a.maskedFill(mask, -1e9);
    b.dumpTensor(std::cout);
    b.reduceSum({0, 1}, false).backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 29
  {
    auto a = Tensor(Shape{3, 3}, std::vector<float>(9, 1.0f), true);
    auto ones = Tensor(Shape{3, 3}, std::vector<float>(9, 1.0f));
    auto mask = ones.triu(1);
    auto b = a.maskedFill(mask, -1e9);
    b.dumpTensor(std::cout);
    b.reduceSum({0, 1}, false).backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 30
  {
    auto a = Tensor(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    auto idx = Tensor(Shape{2, 2}, std::vector<float>{0, 2, 1, 1});
    a.gather(1, idx).dumpTensor(std::cout);
  }

  // Test 31
  {
    auto a = Tensor(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    auto idx = Tensor(Shape{1, 3}, std::vector<float>{1, 0, 1});
    a.gather(0, idx).dumpTensor(std::cout);
  }

  // Test 32
  {
    auto a = Tensor(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}, true);
    auto idx = Tensor(Shape{2, 2}, std::vector<float>{0, 2, 1, 1});
    auto b = a.gather(1, idx);
    b.reduceSum({0, 1}, false).backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 33
  {
    auto a = Tensor(Shape{1, 3}, std::vector<float>{1, 2, 3}, true);
    auto idx = Tensor(Shape{1, 4}, std::vector<float>{0, 0, 0, 0});
    auto b = a.gather(1, idx);
    b.reduceSum({0, 1}, false).backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 34
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1, 2, 3, 4}, true);
    auto idx = Tensor(Shape{2, 1}, std::vector<float>{1, 0});
    auto b = a.softmax(1).gather(1, idx);
    b.reduceSum({0, 1}, false).backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 35
  {
    auto a = Tensor(Shape{3, 2}, std::vector<float>{1, 6, 3, 4, 5, 2});
    a.argmax(0, false).dumpTensor(std::cout);
  }

  // Test 36
  {
    auto a = Tensor(Shape{3, 2}, std::vector<float>{1, 6, 3, 4, 5, 2});
    a.argmax(1, true).dumpTensor(std::cout);
  }

  return 0;
}
