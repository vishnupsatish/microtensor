/**
 *    Author:  Vishnu Satish
 *    Created: Jan 15, 2026
 **/

#include "gpt2.h"

#include <ranges>

#include "module.h"

GPT2::GPT2(int vocabSize, int maxSequenceLength, int embeddingSize)
    // Ugly... need to figure out a better way to take inputs as int to convert
    // to size_t for internal representation.
    : m_maxSequenceLength{static_cast<size_t>(maxSequenceLength)},
      m_token{vocabSize, embeddingSize},
      m_pos{maxSequenceLength, embeddingSize},
      m_ln{Shape{static_cast<size_t>(embeddingSize)}},
      m_revEmb{embeddingSize, vocabSize} {
  m_transformers.reserve(12);
  for (int i = 0; i < 12; ++i) {
    m_transformers.push_back(
        TransformerBlock{static_cast<size_t>(embeddingSize)});
    insertSubModule(&m_transformers[i]);
  }
  insertSubModule(&m_token);
  insertSubModule(&m_pos);
  insertSubModule(&m_ln);
  insertSubModule(&m_revEmb);
}

Tensor GPT2::forward(Tensor x) {
  // x: (batch, sequence length), items are token IDs. Note: sequence length <=
  // m_maxSequenceLength, which denotes the maxmimum sequence length!
  // Add positional and token embeddings.
  // TODO: this should be stored as a class member. Inefficient.
  std::vector<float> posInit;
  posInit.reserve(x.getShape()[1]);
  for (float x : std::views::iota(0U, x.getShape()[1])) {
    posInit.push_back(x);
  }
  Tensor posIdx{Shape{x.getShape()[1]}, posInit};
  auto inp = m_token.forward(x) + m_pos.forward(posIdx);
  for (int i = 0; i < 12; ++i) {
    inp = m_transformers[i].forward(inp);
  }
  inp = m_ln.forward(inp);
  // inp is now in (batch, sequence length, embedding size). We want to create
  // logits for each token in the vocab. Done by multiplying with a tensor of
  // shape (embedding size, vocab size) gives us dot product of "embedding" for
  // predicted next token and embedding for each vocab which are the logits.
  // Usually, this would be weight-tied to token embeddings, but that is
  // annoying to implement given this architecture, but I'll get to it.
  auto logits = m_revEmb.forward(inp);
  return logits;
}
