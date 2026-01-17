#include <iostream>
#include <vector>

#include "tensor.h"

int main() {
  // Test 1
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto b = Tensor(Shape{2}, std::vector<float>{1.0, 2.0});
    (a + b).dumpTensor(std::cout);
  }

  // Test 2
  {
    auto a = Tensor(Shape{2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0));
    auto b = Tensor(Shape{4}, std::vector<float>{0.0, 1.0, 2.0, 3.0});
    (a + b).dumpTensor(std::cout);
  }

  // Test 3
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto b = Tensor(Shape{1}, std::vector<float>{5.0});
    (a * b).dumpTensor(std::cout);
  }

  // Test 4
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto b = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto c = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    (a + b * c).dumpTensor(std::cout);
  }

  // Test 5
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{-1.0, 0.0, 1.0, 2.0});
    a.tanh().dumpTensor(std::cout);
  }

  // Test 6
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto b = Tensor(Shape{2, 2}, std::vector<float>{5.0, 6.0, 7.0, 8.0});
    (a - b).dumpTensor(std::cout);
  }

  // Test 7
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{10.0, 20.0, 30.0, 40.0});
    auto b = Tensor(Shape{2, 2}, std::vector<float>{2.0, 4.0, 5.0, 8.0});
    (a / b).dumpTensor(std::cout);
  }

  // Test 8
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    a.pow(2.0).dumpTensor(std::cout);
  }

  // Test 9
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{10.0, 20.0, 30.0, 40.0});
    auto b = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto res = (a + b) / (a - b);
    res.dumpTensor(std::cout);
  }

  // Test 10
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto b = Tensor(Shape{1}, std::vector<float>{0.5});
    auto res = (a * b).pow(2.0) + (a - b).tanh();
    res.dumpTensor(std::cout);
  }

  // Test 11
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    a.exp().dumpTensor(std::cout);
  }

  // Test 12
  {
    auto a = Tensor(Shape{2, 2},
                    std::vector<float>{1.0, 2.7182818, 7.389056, 20.085537});
    a.log().dumpTensor(std::cout);
  }

  // Test 13
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto b = Tensor(Shape{2, 2}, std::vector<float>{2.0, 3.0, 4.0, 5.0});
    auto res = a.exp() * b.log() - a.pow(0.5);
    res.dumpTensor(std::cout);
  }

  // Test 14
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{-2.0, -1.0, 1.0, 2.0});
    a.gelu().dumpTensor(std::cout);
  }

  return 0;
}
