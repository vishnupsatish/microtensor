/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#include "xor.h"

#include <memory>

#include "linear.h"

XOR::XOR()
    : l1{std::make_unique<Linear>(2, 2)}, l2{std::make_unique<Linear>(2, 1)} {
  insertSubModule(l1.get());
  insertSubModule(l2.get());
}

Tensor XOR::forward(Tensor inp) {
  inp = l1->forward(inp);
  inp = inp.tanh();
  inp = l2->forward(inp);
  return inp;
}
