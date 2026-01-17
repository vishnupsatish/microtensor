/**
 *    Author:  Vishnu Satish
 *    Created: Jan 10, 2026
 **/

#pragma once

#include "tensor.h"

Tensor meanSquaredError(Tensor inp, Tensor target);

Tensor crossEntropy(Tensor inp, Tensor target);
