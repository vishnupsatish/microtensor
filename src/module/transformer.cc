/**
 *    Author:  Vishnu Satish
 *    Created: Jan 16, 2026
 **/

#include "transformer.h"

#include <cassert>
#include <limits>
#include <memory>

#include "layernorm.h"

SelfAttention::SelfAttention(size_t embeddingSize, size_t numHeads)
    : m_embeddingSize{embeddingSize},
      m_numHeads{numHeads},
      m_layernorm{std::make_unique<LayerNorm>(Shape{m_embeddingSize})},
      m_proj{std::make_unique<Linear>(m_embeddingSize, 3 * m_embeddingSize)},
      m_outProj{std::make_unique<Linear>(m_embeddingSize, m_embeddingSize)} {
  insertSubModule(m_layernorm.get());
  insertSubModule(m_proj.get());
  insertSubModule(m_outProj.get());
};

Tensor SelfAttention::forward(Tensor x) {
  auto shape = x.getShape();
  // Unique sequences.
  size_t batch = shape[0];
  // Sequence length.
  size_t seq = shape[1];
  assert(shape[2] == m_embeddingSize);

  auto ln = m_layernorm->forward(x);
  auto qkv = m_proj->forward(ln);
  // Reshape and permute so attention is done to each (sequence, sub-embedding)
  // separately. Each head is treated separately by the matmul.
  auto q = qkv.slice({0, 0, 0}, Shape{batch, seq, m_embeddingSize})
               .reshape({batch, seq, m_numHeads, m_embeddingSize / m_numHeads})
               .permute({0, 1, 3, 2});
  auto k = qkv.slice({0, 0, static_cast<int>(m_embeddingSize)},
                     Shape{batch, seq, m_embeddingSize})
               .reshape({batch, seq, m_numHeads, m_embeddingSize / m_numHeads})
               .permute({0, 1, 3, 2});

  auto v = qkv.slice({0, 0, 2 * static_cast<int>(m_embeddingSize)},
                     Shape{batch, seq, m_embeddingSize})
               .reshape({batch, seq, m_numHeads, m_embeddingSize / m_numHeads})
               .permute({0, 1, 3, 2});

  auto kT = k.permute({0, 1, 3, 2});
  // Dot product each query with each key to get logits. Last two dims are seq,
  // seq. Divide by sqrt.
  auto scores = q.matmul(kT) / std::sqrt(m_embeddingSize / m_numHeads);
  // Create an upper-triangular mask (set to -inf).
  Tensor upperTriangularOnes =
      Tensor{{scores.getShape()[2], scores.getShape()[3]}}.triu(1);
  auto masked = scores.maskedFill(upperTriangularOnes,
                                  -std::numeric_limits<float>::infinity());
  auto prob = masked.softmax(-1);
  // Value for each token, scaled by its "score" (dot product of q, k vectors
  // for those tokens).
  auto vals = prob.matmul(v);
  auto pr = vals.permute({0, 1, 3, 2}).reshape({batch, seq, m_embeddingSize});

  auto projOut = m_outProj->forward(pr);
  return x + projOut;
}
