#include <iostream>
#include <vector>

#include "grad_mode.h"
#include "tensor.h"

int main() {
  // Test 1: requiresGrad mix 1
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0}, true);
    auto b = Tensor(Shape{2, 2}, std::vector<float>{5.0, 6.0, 7.0, 8.0}, false);
    auto c = a * b;
    c.backward();
    a.getGrad().dumpTensor(std::cout);
    if (b.getGrad().isValid())
      b.getGrad().dumpTensor(std::cout);
    else
      Tensor(b.getShape()).dumpTensor(std::cout);
  }

  // Test 2: requiresGrad mix 2 (chained)
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0}, false);
    auto b = Tensor(Shape{2, 2}, std::vector<float>{5.0, 6.0, 7.0, 8.0}, true);
    auto c = (a + b) * a;
    c.backward();
    if (a.getGrad().isValid())
      a.getGrad().dumpTensor(std::cout);
    else
      Tensor(a.getShape()).dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  // Test 3: all requiresGrad false
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0}, false);
    auto b = Tensor(Shape{2, 2}, std::vector<float>{5.0, 6.0, 7.0, 8.0}, false);
    auto c = a + b;
    c.backward();
    if (a.getGrad().isValid())
      a.getGrad().dumpTensor(std::cout);
    else
      Tensor(a.getShape()).dumpTensor(std::cout);
    if (b.getGrad().isValid())
      b.getGrad().dumpTensor(std::cout);
    else
      Tensor(b.getShape()).dumpTensor(std::cout);
  }

  // Test 4: (a + b) * a where (a + b) is in NoGrad
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0}, true);
    auto b = Tensor(Shape{2, 2}, std::vector<float>{5.0, 6.0, 7.0, 8.0}, true);
    Tensor z(Shape{2, 2});
    {
      NoGrad guard;
      z = a + b;
    }
    auto c = z * a;
    c.backward();

    a.getGrad().dumpTensor(std::cout);
    if (b.getGrad().isValid())
      b.getGrad().dumpTensor(std::cout);
    else
      Tensor(b.getShape()).dumpTensor(std::cout);
  }

  // Test 5: a.exp() * a where a.exp() is in NoGrad
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{0.5, 1.0, 1.5, 2.0}, true);
    Tensor z(Shape{2, 2});
    {
      NoGrad guard;
      z = a.exp();
    }
    auto c = z * a;
    c.backward();

    a.getGrad().dumpTensor(std::cout);
  }

  // Test 6: (a * b) + (a * b) where one is in NoGrad
  {
    auto a = Tensor(Shape{2, 2}, std::vector<float>{1.0, 2.0, 3.0, 4.0}, true);
    auto b = Tensor(Shape{2, 2}, std::vector<float>{0.5, 0.5, 0.5, 0.5}, true);
    Tensor z1(Shape{2, 2});
    {
      NoGrad guard;
      z1 = a * b;
    }
    auto z2 = a * b;
    auto res = z1 + z2;
    res.backward();

    a.getGrad().dumpTensor(std::cout);
    b.getGrad().dumpTensor(std::cout);
  }

  return 0;
}
