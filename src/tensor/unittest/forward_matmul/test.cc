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

  // Test 4: Vector-Vector Dot Product (3 * 3 -> scalar)
  {
    auto a = Tensor(Shape{3}, std::vector<float>{1.0, 2.0, 3.0});
    auto b = Tensor(Shape{3}, std::vector<float>{4.0, 5.0, 6.0});
    a.matmul(b).dumpTensor(std::cout);
  }

  // Test 5: Vector-Matrix Multiplication (2 * 2x3 -> 3)
  {
    auto a = Tensor(Shape{2}, std::vector<float>{1.0, 2.0});
    auto b =
        Tensor(Shape{2, 3}, std::vector<float>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    a.matmul(b).dumpTensor(std::cout);
  }

  // Test 6: 4D Batched Matmul with Broadcasting (2x1x4x3 * 1x3x3x2 -> 2x3x4x2)
  {
    auto a = Tensor(Shape{2, 1, 4, 3}, std::vector<float>(2 * 1 * 4 * 3, 1.0f));
    auto b = Tensor(Shape{1, 3, 3, 2}, std::vector<float>(1 * 3 * 3 * 2, 2.0f));
    a.matmul(b).dumpTensor(std::cout);
  }

  return 0;
}
