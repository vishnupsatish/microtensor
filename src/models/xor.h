/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#pragma once

#include <memory>

#include "linear.h"
#include "tensor.h"

class XOR : public Module {
  std::unique_ptr<Linear> l1;
  std::unique_ptr<Linear> l2;

 public:
  XOR();

  Tensor forward(Tensor inp);
};
