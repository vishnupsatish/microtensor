#include <iostream>
#include <vector>

#include "tensor.h"

int main() {
  // Test 1
  {
    auto a = Tensor(Shape{2, 3}, std::vector<float>{1, 2, 3, 4, 5, 6});
    auto b = Tensor(Shape{3, 2}, std::vector<float>{7, 8, 9, 10, 11, 12});
    auto c = a.matmul(b);
    c.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  // Test 2
  {
    auto a = Tensor(Shape{2, 2, 2}, std::vector<float>(8, 1.0f));
    auto b = Tensor(Shape{2, 2}, std::vector<float>{1, 2, 3, 4});
    auto c = a.matmul(b);
    c.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  // Test 3
  {
    auto a = Tensor(Shape{3}, std::vector<float>{1, 2, 3});
    auto b = Tensor(Shape{3}, std::vector<float>{4, 5, 6});
    auto c = a.matmul(b);
    c.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  // Test 4
  {
    auto x = Tensor(Shape{1, 2}, std::vector<float>{0.5, -0.5});
    auto w = Tensor(Shape{2, 2}, std::vector<float>{1, 2, 3, 4});
    auto b = Tensor(Shape{2}, std::vector<float>{0.1, 0.2});

    auto y = (x.matmul(w) + b).tanh();
    y.backward();

    x.getGrad().dumpTensor(std::cout);
    w.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  // Test 5
  {
    auto x = Tensor(Shape{2, 3}, std::vector<float>(6, 1.0f));
    auto w1 = Tensor(Shape{3, 2}, std::vector<float>(6, 2.0f));
    auto w2 = Tensor(Shape{3, 2}, std::vector<float>(6, 3.0f));
    auto y = x.matmul(w1) + x.matmul(w2);
    y.backward();
    x.getGrad().dumpTensor(std::cout);
    w1.getGrad().dumpTensor(std::cout);
    w2.getGrad().dumpTensor(std::cout);
  }

  // Test 6
  {
    auto a = Tensor(Shape{2, 1, 2, 2}, std::vector<float>(8, 1.0f));
    auto b = Tensor(Shape{1, 2, 2, 2}, std::vector<float>(8, 2.0f));
    auto c = a.matmul(b);
    c.backward();
    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  // Test 7
  {
    auto x = Tensor(Shape{2, 3, 2}, std::vector<float>(12, 1.0f));
    auto w = Tensor(Shape{2, 2}, std::vector<float>(4, 2.0f));
    auto x_p = x.permute({1, 0, 2});  // (3, 2, 2)
    auto y = x_p.matmul(w);
    y.backward();
    x.getGrad().dumpTensor(std::cout);
    w.getGrad().dumpTensor(std::cout);
  }

  return 0;
}
