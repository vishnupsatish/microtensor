#include <iostream>
#include <vector>

#include "tensor.h"

int main() {
  // Test 1
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto b = Tensor(Shape{2}, std::vector<float>{1.0, 2.0});
    auto c = a + b;
    c.backward();
    a.getGrad().dump_tensor(std::cout);
    b.getGrad().dump_tensor(std::cout);
  }

  // Test 2
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto b = Tensor(Shape{2}, std::vector<float>{10.0, 20.0});
    auto c = a * b;
    c.backward();
    a.getGrad().dump_tensor(std::cout);
    b.getGrad().dump_tensor(std::cout);
  }

  // Test 3
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto b = Tensor(Shape{2}, std::vector<float>{1.0, 2.0});
    auto c = Tensor(Shape{2, 2}, std::vector<float>{5.0, 5.0, 5.0, 5.0});
    auto d = a * b + c;
    d.backward();
    a.getGrad().dump_tensor(std::cout);
    b.getGrad().dump_tensor(std::cout);
    c.getGrad().dump_tensor(std::cout);
  }

  // Test 4
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto b = Tensor(Shape{2}, std::vector<float>{0.5, 0.5});
    auto c = Tensor(Shape{1}, std::vector<float>{2.0});
    auto res = (a * b) + (a * c);
    res.backward();
    a.getGrad().dump_tensor(std::cout);
    b.getGrad().dump_tensor(std::cout);
    c.getGrad().dump_tensor(std::cout);
  }

  // Test 5
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto b = Tensor(Shape{2}, std::vector<float>{1.0, 2.0});
    auto c = Tensor(Shape{1}, std::vector<float>{10.0});
    auto res = (a + b) * (a + c) + b;
    res.backward();
    a.getGrad().dump_tensor(std::cout);
    b.getGrad().dump_tensor(std::cout);
    c.getGrad().dump_tensor(std::cout);
  }

  // Test 6
  {
    auto a = Tensor(Shape{1, 3}, std::vector<float>{1.0, 2.0, 3.0});
    auto b = Tensor(Shape{3, 1}, std::vector<float>{10.0, 20.0, 30.0});
    auto c = Tensor(Shape{1, 1}, std::vector<float>{5.0});
    auto res = (a + b) * (a + c) * (b + c);
    res.backward();
    a.getGrad().dump_tensor(std::cout);
    b.getGrad().dump_tensor(std::cout);
    c.getGrad().dump_tensor(std::cout);
  }

  return 0;
}
