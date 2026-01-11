/**
 *    Author:  Vishnu Satish
 *    Created: Jan 11, 2026
 **/

// Implementation details for kernel.h. MUST ONLY be included by that file.

template <typename F>
void elementwiseBinaryKernel(std::shared_ptr<TensorImpl> res,
                             std::shared_ptr<TensorImpl> a,
                             std::shared_ptr<TensorImpl> b, F&& binary_fn) {
  assert(res->m_shape == a->m_shape && res->m_shape == b->m_shape);
  auto out = std::make_shared<TensorImpl>(a->m_shape);
  size_t total_elements = sizeFromShape(a->m_shape);
  std::vector<size_t> coords(a->m_shape.size(), 0);

  for (size_t i = 0; i < total_elements; ++i) {
    size_t offset_a = getPhysicalOffset(coords, a->m_strides, a->m_offset);
    size_t offset_b = getPhysicalOffset(coords, b->m_strides, b->m_offset);
    size_t offset_res =
        getPhysicalOffset(coords, res->m_strides, res->m_offset);
    (*res->m_data)[offset_res] =
        binary_fn((*a->m_data)[offset_a], (*b->m_data)[offset_b]);
    incrementCoords(coords, a->m_shape);
  }
}

template <typename F>
void elementwiseUnaryKernel(std::shared_ptr<TensorImpl> res,
                            std::shared_ptr<TensorImpl> a, F&& unary_fn) {
  assert(res->m_shape == a->m_shape);
  size_t total_elements = sizeFromShape(a->m_shape);
  std::vector<size_t> coords(a->m_shape.size(), 0);

  for (size_t i = 0; i < total_elements; ++i) {
    size_t offset_a = getPhysicalOffset(coords, a->m_strides, a->m_offset);
    size_t offset_res =
        getPhysicalOffset(coords, res->m_strides, res->m_offset);
    (*res->m_data)[offset_res] = unary_fn((*a->m_data)[offset_a]);
    incrementCoords(coords, a->m_shape);
  }
}
