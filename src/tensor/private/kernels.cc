/**
 *    Author:  Vishnu Satish
 *    Created: Jan 11, 2026
 **/

#include "kernels.h"

void getCoordsFromIndex(size_t index, const Shape& shape,
                        std::vector<size_t>& coords) {
  // We iterate backwards to fill the last dimensions first (standard row-major)
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    coords[i] = index % shape[i];
    index /= shape[i];
  }
}

// Performs C = A * B for a single matrix slice
// Assumes pointers are already offset to the correct batch location
void matmulKernel(float* ptr_c, const float* ptr_a, const float* ptr_b,
                  size_t m, size_t k, size_t n, size_t stride_a_m,
                  size_t stride_a_k,  // Strides for A's last 2 dims
                  size_t stride_b_k,
                  size_t stride_b_n,  // Strides for B's last 2 dims
                  size_t stride_c_m,
                  size_t stride_c_n  // Strides for C's last 2 dims
) {
  for (size_t i = 0; i < m; ++i) {    // rows of A
    for (size_t j = 0; j < n; ++j) {  // columns of B

      float sum = 0.0f;

      for (size_t p = 0; p < k; ++p) {
        sum += ptr_a[i * stride_a_m + p * stride_a_k] *
               ptr_b[p * stride_b_k + j * stride_b_n];
      }

      ptr_c[i * stride_c_m + j * stride_c_n] = sum;
    }
  }
}

void matmulBatched(std::shared_ptr<TensorImpl> c, std::shared_ptr<TensorImpl> a,
                   std::shared_ptr<TensorImpl> b) {
  size_t rank = c->getRank();

  size_t batch_rank = rank - 2;
  size_t m = c->m_shape[rank - 2];
  size_t n = c->m_shape[rank - 1];

  size_t k = a->m_shape[rank - 1];

  Shape batch_shape(c->m_shape.begin(), c->m_shape.end() - 2);

  size_t total_batches = 1;
  for (size_t i = 0; i < batch_rank; ++i) total_batches *= batch_shape[i];

    // Each batch in a matmul is completely separate. Therefore, it can safely
    // be parallelized.
#pragma omp parallel
  {
    std::vector<size_t> current_coords(batch_rank, 0);
#pragma omp for
    for (size_t batch = 0; batch < total_batches; ++batch) {
      getCoordsFromIndex(batch, batch_shape, current_coords);

      size_t offset_a = a->m_offset;
      size_t offset_b = b->m_offset;
      size_t offset_c = c->m_offset;

      // Use the current coordinates to find the start of the data
      for (size_t i = 0; i < batch_rank; ++i) {
        offset_a += current_coords[i] * a->m_strides[i];
        offset_b += current_coords[i] * b->m_strides[i];
        offset_c += current_coords[i] * c->m_strides[i];
      }

      float* ptr_a = a->m_data->data() + offset_a;
      float* ptr_b = b->m_data->data() + offset_b;
      float* ptr_c = c->m_data->data() + offset_c;
      matmulKernel(ptr_c, ptr_a, ptr_b, m, k, n, a->m_strides[rank - 2],
                   a->m_strides[rank - 1], b->m_strides[rank - 2],
                   b->m_strides[rank - 1], c->m_strides[rank - 2],
                   c->m_strides[rank - 1]);
    }
  }
}
