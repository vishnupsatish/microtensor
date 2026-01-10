#include <iostream>
#include <vector>

#include "tensor.h"

int main() {
  // Test 1: reshape
  {
    auto a = Tensor(Shape{6}, std::vector<float>{1, 2, 3, 4, 5, 6});
    a.reshape(Shape{2, 3}).dumpTensor(std::cout);
  }

  // Test 2: reshape backward
  {
    auto a = Tensor(Shape{6}, std::vector<float>{1, 2, 3, 4, 5, 6}, true);
    auto b = a.reshape(Shape{3, 2});
    b.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 3: contiguous after permute
  {
    auto a = Tensor(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    auto b = a.permute({1, 0});  // Shape {3, 2}, non-contiguous
    b.makeContiguous().dumpTensor(std::cout);
  }

  // Test 4: contiguous after broadcast
  {
    auto a = Tensor(Shape{3}, std::vector<float>{1, 2, 3});
    auto b = a.broadcast(Shape{2, 3});  // non-contiguous
    b.makeContiguous().dumpTensor(std::cout);
  }

  // Test 5: reshape on non-contiguous tensor
  {
    auto a = Tensor(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}, true);
    auto b = a.permute({1, 0});    // Shape {3, 2}, non-contiguous
    auto c = b.reshape(Shape{6});  // Should call makeContiguous internally
    c.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 6: slice forward
  {
    auto a =
        Tensor(Shape{4, 4}, std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                               11, 12, 13, 14, 15, 16});
    a.slice({1, 1}, Shape{2, 2}).dumpTensor(std::cout);
  }

  // Test 7: slice backward
  {
    auto a = Tensor(Shape{4, 4},
                    std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                       13, 14, 15, 16},
                    true);
    auto b = a.slice({1, 1}, Shape{2, 2});
    b.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 8: Slice -> Permute -> Add
  {
    auto a = Tensor(Shape{4, 4},
                    std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                                       13, 14, 15, 16},
                    true);
    auto b = a.slice({1, 0},
                     Shape{2, 3});  // Rows [1,2], Cols [0,1,2] -> Shape {2, 3}
    auto c = b.permute({1, 0});     // Shape {3, 2}
    auto d = Tensor(Shape{3, 2},
                    std::vector<float>{0.1, 0.2, 0.3, 0.4, 0.5, 0.6}, true);
    auto res = (c + d).pow(2.0);
    res.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 9: Reshape -> Slice -> Backward
  {
    auto a =
        Tensor(Shape{12},
               std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, true);
    auto b = a.reshape(Shape{3, 4});
    auto c = b.slice({1, 1}, Shape{2, 2});  // Shape {2, 2}
    c.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 10: Slice -> Broadcast -> Backward
  {
    auto a = Tensor(Shape{4, 4}, std::vector<float>(16, 1.0f), true);
    auto b = a.slice({1, 1}, Shape{1, 2});  // Shape {1, 2}
    auto c = b.broadcast(Shape{3, 2});      // Shape {3, 2}
    c.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 11: slice -> matmul -> backward
  {
    auto a = Tensor(Shape{4, 4}, std::vector<float>(16, 1.0f), true);
    auto b = Tensor(Shape{2, 2}, std::vector<float>{1, 2, 3, 4}, true);
    auto a_slice = a.slice({1, 1}, Shape{2, 2});
    auto res = a_slice.matmul(b);
    res.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  // Test 12: reshape -> permute -> slice -> backward
  {
    auto a =
        Tensor(Shape{12},
               std::vector<float>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, true);
    auto b = a.reshape(Shape{2, 3, 2});
    auto c = b.permute({1, 0, 2});                // Shape {3, 2, 2}
    auto d = c.slice({1, 0, 0}, Shape{2, 1, 2});  // Shape {2, 1, 2}
    d.backward();
    a.getGrad().dumpTensor(std::cout);
  }

  // Test 13: matmul -> reshape -> slice -> add -> backward
  {
    auto a = Tensor(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6}, true);
    auto b = Tensor(Shape{3, 2}, std::vector<float>{7, 8, 9, 10, 11, 12}, true);
    auto c = a.matmul(b);  // Shape {2, 2}
    auto d = c.reshape(Shape{4});
    auto e = d.slice({1}, Shape{2});  // Shape {2}
    auto res = e + Tensor(Shape{2}, std::vector<float>{0.5, 0.5}, true);
    res.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  return 0;
}
