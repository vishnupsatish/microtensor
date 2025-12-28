#include <iostream>
#include <vector>

#include "tensor.h"

int main() {
  // Test 1
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto b = Tensor(Shape{2}, std::vector<float>{1.0, 2.0});
    (a + b).dump_tensor(std::cout);
  }

  // Test 2
  {
    auto a = Tensor(Shape{2, 3, 4}, std::vector<float>(2 * 3 * 4, 1.0));
    auto b = Tensor(Shape{4}, std::vector<float>{0.0, 1.0, 2.0, 3.0});
    (a + b).dump_tensor(std::cout);
  }

  // Test 3
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto b = Tensor(Shape{1}, std::vector<float>{5.0});
    (a * b).dump_tensor(std::cout);
  }

  // Test 4
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto b = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    auto c = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    (a + b * c).dump_tensor(std::cout);
  }

  return 0;
}
