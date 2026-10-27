#pragma once
#ifdef USE_CUDNN

#include "container/tensor/tensor.hpp"

#include <cstddef>
#include <utility>

namespace tensor {

Tensor<float> conv2d_forward_cudnn(const Tensor<float>& x, const Tensor<float>& W,
								   std::pair<size_t, size_t> stride,
								   std::pair<size_t, size_t> pad);

} // namespace tensor

#endif // USE_CUDNN
