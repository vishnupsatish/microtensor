/**
 *    Author:  Vishnu Satish
 *    Created: Jan 16, 2026
 **/

#pragma once

#include "layers.h"
#include "module.h"
#include "transformer.h"

// TODO: hyperparameters should be configurable in a cleaner way.
class GPT2 : public Module {
  size_t m_maxSequenceLength;
  int m_vocabSize;
  // Token embeddings
  std::unique_ptr<Embedding> m_token;
  // Positional embeddings
  std::unique_ptr<Embedding> m_pos;
  std::vector<std::unique_ptr<TransformerBlock>> m_transformers;
  std::unique_ptr<LayerNorm> m_ln;

  // Should be weight-tied to m_token but is not...
  std::unique_ptr<Linear> m_revEmb;

 public:
  GPT2(int vocabSize, int maxSequenceLength, int embeddingSize);
  // Returns (batch, sequence length, vocab size) corresponding to logits of
  // predicted next token for each token in the sequence.
  Tensor forward(Tensor x);

  int getVocabSize();
};
