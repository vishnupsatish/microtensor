/**
 *    Author:  Vishnu Satish
 *    Created: Jan 16, 2026
 **/

#pragma once

#include <memory>

#include "layers.h"
#include "module.h"

class SelfAttention : public Module {
  size_t m_embeddingSize;
  size_t m_numHeads;

  // The projection to Q, K, V (in that order).
  std::unique_ptr<Linear> m_proj;
  std::unique_ptr<Linear> m_outProj;

 public:
  SelfAttention(size_t embeddingSize, size_t numHeads = 12);
  Tensor forward(Tensor x);
};

class TransformerBlock : public Module {
  size_t m_embeddingSize;
  std::unique_ptr<LayerNorm> m_ln1;
  std::unique_ptr<SelfAttention> m_attn;
  std::unique_ptr<LayerNorm> m_ln2;
  std::unique_ptr<MLP> m_mlp;

 public:
  TransformerBlock(size_t embeddingSize);
  Tensor forward(Tensor x);
};
