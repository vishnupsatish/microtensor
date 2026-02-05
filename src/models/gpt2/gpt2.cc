/**
 *    Author:  Vishnu Satish
 *    Created: Jan 15, 2026
 **/

#include "gpt2.h"

#include <memory>
#include <ranges>

#include "layers.h"
#include "module.h"

const int layers = 4;

GPT2::GPT2(int vocabSize, int maxSequenceLength, int embeddingSize)
    // Ugly... need to figure out a better way to take inputs as int to convert
    // to size_t for internal representation.
    : m_maxSequenceLength{static_cast<size_t>(maxSequenceLength)},
      m_vocabSize{vocabSize},
      m_token{std::make_unique<Embedding>(vocabSize, embeddingSize)},
      m_pos{std::make_unique<Embedding>(maxSequenceLength, embeddingSize)},
      m_dropout{std::make_unique<Dropout>(0.1)},
      m_ln{std::make_unique<LayerNorm>(
          Shape{static_cast<size_t>(embeddingSize)})},
      m_revEmb{std::make_unique<Linear>(embeddingSize, vocabSize, false)} {
  m_transformers.reserve(layers);
  for (int i = 0; i < layers; ++i) {
    m_transformers.push_back(
        std::make_unique<TransformerBlock>(static_cast<size_t>(embeddingSize)));
    insertSubModule(m_transformers[i].get());
  }
  insertSubModule(m_token.get());
  insertSubModule(m_pos.get());
  insertSubModule(m_dropout.get());
  insertSubModule(m_ln.get());
  insertSubModule(m_revEmb.get());
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
  auto inp = m_token->forward(x) + m_pos->forward(posIdx);
  inp = m_dropout->forward(inp);
  for (int i = 0; i < layers; ++i) {
    inp = m_transformers[i]->forward(inp);
  }
  inp = m_ln->forward(inp);
  // inp is now in (batch, sequence length, embedding size). We want to create
  // logits for each token in the vocab. Done by multiplying with a tensor of
  // shape (embedding size, vocab size) gives us dot product of "embedding" for
  // predicted next token and embedding for each vocab which are the logits.
  // Usually, this would be weight-tied to token embeddings, but that is
  // annoying to implement given this architecture, but I'll get to it.
  auto logits = m_revEmb->forward(inp);
  return logits;
}

int GPT2::getVocabSize() { return m_vocabSize; }
