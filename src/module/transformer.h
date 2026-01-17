/**
 *    Author:  Vishnu Satish
 *    Created: Jan 16, 2026
 **/

#pragma once

#include "layernorm.h"
#include "linear.h"
#include "module.h"

class Block : public Module {};

class SelfAttention : public Module {
  size_t m_embeddingSize;
  size_t m_numHeads = 12;

  std::unique_ptr<LayerNorm> m_layernorm;
  // The projection to Q, K, V (in that order).
  std::unique_ptr<Linear> m_proj;
  std::unique_ptr<Linear> m_outProj;

 public:
  SelfAttention(size_t embeddingSize, size_t numHeads);
  Tensor forward(Tensor x);
};
