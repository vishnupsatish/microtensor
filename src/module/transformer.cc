/**
 *    Author:  Vishnu Satish
 *    Created: Jan 16, 2026
 **/

#include "transformer.h"

#include <cassert>
#include <limits>
#include <memory>

#include "layers.h"

SelfAttention::SelfAttention(size_t embeddingSize, size_t numHeads)
    : m_embeddingSize{embeddingSize},
      m_numHeads{numHeads},
      m_proj{std::make_unique<Linear>(m_embeddingSize, 3 * m_embeddingSize)},
      m_outProj{std::make_unique<Linear>(m_embeddingSize, m_embeddingSize)} {
  insertSubModule(m_proj.get());
  insertSubModule(m_outProj.get());
};

Tensor SelfAttention::forward(Tensor x) {
  // Note: layernorm and residual connection not performed here.

  if (m_embeddingSize % m_numHeads != 0) {
    throw std::runtime_error{
        "Embedding (feature) size must be divisible by the number of heads."};
  }

  auto shape = x.getShape();
  // Unique sequences.
  size_t batch = shape[0];
  // Sequence length. Note: this is determined at runtime and is <= max sequence
  // length.
  size_t seq = shape[1];
  assert(shape[2] == m_embeddingSize);

  std::vector<size_t> transpose{0, 1, 3, 2};
  std::vector<size_t> headPermute{0, 2, 1, 3};

  auto qkv = m_proj->forward(x);
  // Reshape and permute so attention is done to each (sequence, sub-embedding)
  // separately. Each head is treated separately by the matmul.
  auto q = qkv.slice({0, 0, 0}, Shape{batch, seq, m_embeddingSize})
               .reshape({batch, seq, m_numHeads, m_embeddingSize / m_numHeads})
               .permute(headPermute);
  auto k = qkv.slice({0, 0, static_cast<int>(m_embeddingSize)},
                     Shape{batch, seq, m_embeddingSize})
               .reshape({batch, seq, m_numHeads, m_embeddingSize / m_numHeads})
               .permute(headPermute);

  auto v = qkv.slice({0, 0, 2 * static_cast<int>(m_embeddingSize)},
                     Shape{batch, seq, m_embeddingSize})
               .reshape({batch, seq, m_numHeads, m_embeddingSize / m_numHeads})
               .permute(headPermute);

  auto kT = k.permute(transpose);
  // Dot product each query with each key to get logits. Last two dims are seq,
  // seq. Divide by sqrt.
  auto scores = q.matmul(kT) / std::sqrt(m_embeddingSize / m_numHeads);
  // Create an upper-triangular mask (set to -inf).

  Tensor upperTriangularOnes =
      Tensor{{scores.getShape()[2], scores.getShape()[3]},
             std::vector<float>(scores.getShape()[2] * scores.getShape()[3], 1)}
          .triu(1);
  auto masked = scores.maskedFill(upperTriangularOnes,
                                  -std::numeric_limits<float>::infinity());
  auto prob = masked.softmax(-1);
  // Value for each token, scaled by its "score" (dot product of q, k vectors
  // for those tokens).
  auto vals = prob.matmul(v);
  auto pr = vals.permute(headPermute).reshape({batch, seq, m_embeddingSize});

  auto projOut = m_outProj->forward(pr);
  return projOut;
}

////////////////////////////////////////////////////////////////////////////////

TransformerBlock::TransformerBlock(size_t embeddingSize)
    : m_embeddingSize{embeddingSize},
      m_ln1{std::make_unique<LayerNorm>(Shape{m_embeddingSize})},
      m_attn{std::make_unique<SelfAttention>(m_embeddingSize)},
      m_ln2{std::make_unique<LayerNorm>(Shape{m_embeddingSize})},
      m_mlp{std::make_unique<MLP>(m_embeddingSize, 4 * m_embeddingSize)} {
  insertSubModule(m_ln1.get());
  insertSubModule(m_attn.get());
  insertSubModule(m_ln2.get());
  insertSubModule(m_mlp.get());
}

Tensor TransformerBlock::forward(Tensor x) {
  auto attnInput = m_ln1->forward(x);
  auto attnResidual = x + m_attn->forward(attnInput);
  auto mlpInput = m_ln2->forward(attnResidual);
  auto mlpResidual = attnResidual + m_mlp->forward(mlpInput);
  return mlpResidual;
}
