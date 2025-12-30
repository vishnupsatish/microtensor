#include <iostream>
#include <vector>

#include "tensor.h"

int main() {
  // Test 1: Simple 2D Matrix Multiplication (2x3 * 3x2 -> 2x2)
  {
    auto a =
        Tensor(Shape{2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto b = Tensor(Shape{3, 2},
                    std::vector<float>{7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    a.matmul(b).dumpTensor(std::cout);
  }

  // Test 2: Matrix-Vector Multiplication (2x3 * 3 -> 2)
  {
    auto a =
        Tensor(Shape{2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    auto b = Tensor(Shape{3}, std::vector<float>{1.0, 2.0, 3.0});
    a.matmul(b).dumpTensor(std::cout);
  }

  // Test 3: Batched Matmul with Broadcasting (2x1x3x2 * 3x2x4 -> 2x3x3x4)
  // Simplified: (2, 2, 2) * (2, 2) -> (2, 2, 2)
  {
    auto a = Tensor(Shape{2, 2, 2}, std::vector<float>(8, 1.0f));
    auto b = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0});
    a.matmul(b).dumpTensor(std::cout);
  }

  return 0;
}
