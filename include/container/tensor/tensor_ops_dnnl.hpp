#pragma once

#ifdef USE_DNNL

#include <dnnl.hpp>
#include "container/tensor/tensor.hpp"
#include "container/tensor/tensor_functions.hpp"
#include "container/tensor/tensor_utils.hpp"

namespace tensor {

// Helper to check if we can use oneDNN for this operation
template<typename T>
constexpr bool can_use_dnnl() {
    return std::is_same<T, float>::value;  // oneDNN primarily optimized for float
}

// oneDNN-optimized 2D convolution (forward pass)
// Input: x [N, C, H, W]
// Weight: w [OC, C, KH, KW]
// Bias: b [OC] (optional)
// Output: y [N, OC, OH, OW]
template<typename T>
Tensor<T> conv2d_dnnl(
    const Tensor<T>& x,
    const Tensor<T>& w,
    const Tensor<T>& b,
    std::pair<size_t, size_t> stride = {1, 1},
    std::pair<size_t, size_t> padding = {0, 0}
) {
    static_assert(can_use_dnnl<T>(), "oneDNN conv2d only supports float type");

    using namespace dnnl;

    // Get shapes
    auto x_shape = x.get_shape();  // [N, C, H, W]
    auto w_shape = w.get_shape();  // [OC, C, KH, KW]

    if (x_shape.size() != 4 || w_shape.size() != 4) {
        throw std::runtime_error("conv2d_dnnl: input must be 4D tensors");
    }

    size_t N = x_shape[0];   // batch size
    size_t C = x_shape[1];   // input channels
    size_t H = x_shape[2];   // input height
    size_t W = x_shape[3];   // input width

    size_t OC = w_shape[0];  // output channels
    size_t KH = w_shape[2];  // kernel height
    size_t KW = w_shape[3];  // kernel width

    auto [SH, SW] = stride;
    auto [PH, PW] = padding;

    // Calculate output dimensions
    size_t OH = (H + 2 * PH - KH) / SH + 1;
    size_t OW = (W + 2 * PW - KW) / SW + 1;

    // Create oneDNN engine (CPU)
    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    // Define memory dimensions
    memory::dims src_dims = {static_cast<int64_t>(N), static_cast<int64_t>(C),
                              static_cast<int64_t>(H), static_cast<int64_t>(W)};
    memory::dims weights_dims = {static_cast<int64_t>(OC), static_cast<int64_t>(C),
                                  static_cast<int64_t>(KH), static_cast<int64_t>(KW)};
    memory::dims dst_dims = {static_cast<int64_t>(N), static_cast<int64_t>(OC),
                              static_cast<int64_t>(OH), static_cast<int64_t>(OW)};
    memory::dims strides_dims = {static_cast<int64_t>(SH), static_cast<int64_t>(SW)};
    memory::dims padding_dims_l = {static_cast<int64_t>(PH), static_cast<int64_t>(PW)};
    memory::dims padding_dims_r = {static_cast<int64_t>(PH), static_cast<int64_t>(PW)};

    // Create memory descriptors
    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oihw);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);

    // Make input tensors contiguous and KEEP them in scope
    Tensor<T> x_contig = x.contiguous();
    Tensor<T> w_contig = w.contiguous();
    
    // Copy input data to vectors to ensure memory stability
    const auto& x_data = x_contig.raw_data();
    const auto& w_data = w_contig.raw_data();
    std::vector<T> src_vec(x_data.begin(), x_data.end());
    std::vector<T> weights_vec(w_data.begin(), w_data.end());

    // Create memory objects
    auto src_mem = memory(src_md, eng, src_vec.data());
    auto weights_mem = memory(weights_md, eng, weights_vec.data());

    // Allocate output memory
    std::vector<T> dst_data(N * OC * OH * OW);
    auto dst_mem = memory(dst_md, eng, dst_data.data());

    // Create convolution primitive descriptor (oneDNN 3.x API)
    if (!b.empty()) {
        // With bias
        auto b_shape = b.get_shape();
        memory::dims bias_dims = {static_cast<int64_t>(b_shape[0])};
        auto bias_md = memory::desc(bias_dims, memory::data_type::f32, memory::format_tag::x);

        Tensor<T> b_contig = b.contiguous();
        const auto& b_data = b_contig.raw_data();
        std::vector<T> bias_vec(b_data.begin(), b_data.end());
        auto bias_mem = memory(bias_md, eng, bias_vec.data());

        auto conv_pd = convolution_forward::primitive_desc(
            eng,
            prop_kind::forward_inference,
            algorithm::convolution_direct,  // Use direct for simplicity/stability
            src_md, weights_md, bias_md, dst_md,
            strides_dims, padding_dims_l, padding_dims_r
        );

        auto conv_prim = convolution_forward(conv_pd);

        // Execute convolution
        conv_prim.execute(s, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_WEIGHTS, weights_mem},
            {DNNL_ARG_BIAS, bias_mem},
            {DNNL_ARG_DST, dst_mem}
        });
    } else {
        // Without bias
        auto conv_pd = convolution_forward::primitive_desc(
            eng,
            prop_kind::forward_inference,
            algorithm::convolution_direct,
            src_md, weights_md, dst_md,
            strides_dims, padding_dims_l, padding_dims_r
        );

        auto conv_prim = convolution_forward(conv_pd);

        // Execute convolution
        conv_prim.execute(s, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_WEIGHTS, weights_mem},
            {DNNL_ARG_DST, dst_mem}
        });
    }

    // Wait for completion
    s.wait();

    // Return result tensor
    return Tensor<T>({N, OC, OH, OW}, dst_data);
}

} // namespace tensor

#endif // USE_DNNL
