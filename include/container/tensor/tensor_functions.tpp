#pragma once

#include "container/tensor/tensor.hpp"
#include "container/tensor/tensorview.hpp"
#include "container/tensor/tensor_utils.hpp"

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <omp.h>

namespace tensor {


template<typename T>
Tensor<T> broadcast_to(const Tensor<T>& src, const std::vector<size_t>& target_shape) {

	const auto& src_shape = src.get_shape();
	const auto& src_strides = src.get_strides();

	size_t ndim_src = src_shape.size();
	size_t ndim_target = target_shape.size();

	std::vector<size_t> padded_shape = src_shape;
	std::vector<size_t> padded_strides = src_strides;

	if (ndim_src < ndim_target) {
		size_t diff = ndim_target - ndim_src;
		padded_shape.insert(padded_shape.begin(), diff, 1);
		padded_strides.insert(padded_strides.begin(), diff, 0);
	}

	std::vector<size_t> new_strides(ndim_target);
	for (size_t i = 0; i < ndim_target; i++) {
		if (padded_shape[i] == target_shape[i])
			new_strides[i] = padded_strides[i];
		else if (padded_shape[i] == 1) 
			new_strides[i] = 0;
		else
			throw std::runtime_error("broadcast_to: shape mismatch at dim " + std::to_string(i));
	}

	auto view = std::make_shared<TensorView<T>>(
			target_shape,
			src.shared_data(),
			new_strides,
			src.get_offset()
			);

	Tensor<T> result(view);
	return result;
}


template<typename T>
Tensor<T> sum_to(const Tensor<T>& src, const std::vector<size_t>& target_shape) {

	const auto& src_shape = src.get_shape();

	size_t ndim_src = src_shape.size();
	size_t ndim_target = target_shape.size();

	if (ndim_src < ndim_target)
		std::runtime_error("sum_to: target shape has more dimensions than source.");

	std::vector<int> reduce_axes;
	size_t leading = ndim_src - ndim_target;

	for (size_t i = 0; i < leading; i++)
		reduce_axes.push_back(static_cast<int>(i));

	for (size_t i = 0; i < ndim_target; i++) {
		size_t src_dim = src_shape[leading + i];
		size_t tgt_dim = target_shape[i];
		if (tgt_dim == 1 && src_dim != 1) 
			reduce_axes.push_back(static_cast<int>(leading + i));
		else if (tgt_dim != src_dim && tgt_dim != 1)
			throw std::runtime_error("sum_to: shapes are not broadcast-compatible.");
	}
	
	if (reduce_axes.empty()) 
		return src;

	Tensor<T> summed = src.sum(reduce_axes, true);

	if (summed.get_shape() != target_shape)
		summed = summed.reshape(target_shape);

	return summed;

}

template <typename T>
void add_at(Tensor<T>& gx, 
			const std::vector<size_t>& slices, 
			const Tensor<T>& gy) {

	Tensor<T> view;
	for (size_t idx : slices)
		view = gx[idx];

	auto& gx_data = view.raw_data();
	const auto& gy_data = gy.raw_data();
	size_t offset = view.get_offset();

	for (size_t i = 0; i < gy.size(); ++i)
		gx_data[offset + i] += gy_data[i]; 

}

template<typename T>
std::tuple<Tensor<T>, Tensor<T>, std::vector<size_t>> broadcast_binary_operands(
    const Tensor<T>& a, const Tensor<T>& b) {

	std::vector<size_t> shape_a = a.get_shape();
	std::vector<size_t> shape_b = b.get_shape();
	size_t ndim = std::max(shape_a.size(), shape_b.size());

	while (shape_a.size() < ndim) shape_a.insert(shape_a.begin(), 1);
	while (shape_b.size() < ndim) shape_b.insert(shape_b.begin(), 1);

	std::vector<size_t> broadcast_shape(ndim);
	for (size_t i = 0; i < ndim; ++i) {
		if (shape_a[i] == shape_b[i])
			broadcast_shape[i] = shape_a[i];
		else if (shape_a[i] == 1)
			broadcast_shape[i] = shape_b[i];
		else if (shape_b[i] == 1)
			broadcast_shape[i] = shape_a[i];
		else
			throw std::runtime_error("broadcast_binary_operands: shape mismatch");
	}

	Tensor<T> a_bc = broadcast_to(a, broadcast_shape);
	Tensor<T> b_bc = broadcast_to(b, broadcast_shape);

	return {a_bc, b_bc, broadcast_shape};
}


template<typename T>
Tensor<T> stack(const std::vector<Tensor<T>>& tensors) {
	if (tensors.empty())
		throw std::runtime_error("Cannot stack empty tensor list");

	const auto& base_shape = tensors[0].get_shape();
	size_t batch_size = tensors.size();

	std::vector<size_t> new_shape = { batch_size };
	new_shape.insert(new_shape.end(), base_shape.begin(), base_shape.end());
	
	std::vector<T> stacked_data;

	for (const auto& t : tensors) {
		if (t.get_shape() != base_shape)
			throw std::runtime_error("All tensors must have the same shape to stack");

		const auto& data = t.view_data();
		stacked_data.insert(stacked_data.end(), data.begin(), data.end());
	}

	return Tensor<T>(new_shape, stacked_data);
}


template<typename SrcT, typename DstT>
Tensor<DstT> cast_tensor(const Tensor<SrcT>& src) {
	std::vector<DstT> dst(src.size());

	for (size_t i = 0; i < src.size(); i++)
		dst[i] = static_cast<DstT>(src.raw_data()[i]);

	return Tensor<DstT>(src.get_shape(), dst);
}


template<typename T>
Tensor<T> im2col_array(	const Tensor<T> img,
						std::pair<size_t, size_t> kernel_size,
						std::pair<size_t, size_t> stride,
						std::pair<size_t, size_t> pad,
						bool to_matrix) {
	std::vector<size_t> shape = img.get_shape();
	size_t N = shape[0];
	size_t C = shape[1];
	size_t H = shape[2];
	size_t W = shape[3];

	auto [KH, KW] = kernel_size;
	auto [SH, SW] = stride;
	auto [PH, PW] = pad;
	size_t OH = get_conv_outsize(H, KH, SH, PH);
	size_t OW = get_conv_outsize(W, KW, SW, PW);

	// to array
	auto padded = img.pad({{0, 0}, {0, 0}, {PH, PH}, {PW, PW}}, 0);

	Tensor<T> col({N, C, KH, KW, OH, OW});

	#pragma omp parallel for collapse(4)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t j = 0; j < KH; ++j) {
                for (size_t i = 0; i < KW; ++i) {
                    for (size_t y = 0; y < OH; ++y) {
                        for (size_t x = 0; x < OW; ++x)
                            col({n, c, j, i, y, x}) = padded({n, c, j + y * SH, i + x * SW});
                    }
                }
            }
        }
    }

	if (to_matrix) { 
		col = col.transpose({0, 4, 5, 1, 2, 3});
		col = col.contiguous();
		col = col.reshape({N * OH * OW, C * KH * KW});
	}

	return col;

}


template<typename T>
Tensor<T> col2im_array(	const Tensor<T> col,
						std::vector<size_t> img_shape,
						std::pair<size_t, size_t> kernel_size,
						std::pair<size_t, size_t> stride,
						std::pair<size_t, size_t> pad,
						bool to_matrix) {
	size_t N = img_shape[0];
	size_t C = img_shape[1];
	size_t H = img_shape[2];
	size_t W = img_shape[3];

	auto [KH, KW] = kernel_size;
	auto [SH, SW] = stride;
	auto [PH, PW] = pad;
	size_t OH = get_conv_outsize(H, KH, SH, PH);
	size_t OW = get_conv_outsize(W, KW, SW, PW);

	Tensor<T> col_arr = col;


	if (to_matrix) {
		col_arr = col_arr.reshape({N, OH, OW, C, KH, KW});
		col_arr = col_arr.transpose({0, 3, 4, 5, 1, 2});	
		col_arr.contiguous();
	}

	// to array
	Tensor<T> padded_img({N, C, H + 2*PH + SH - 1, W + 2*PW + SW - 1});

	//#pragma omp parallel for collapse(4)
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t j = 0; j < KH; ++j) {
                for (size_t i = 0; i < KW; ++i) {
                    for (size_t y = 0; y < OH; ++y) {
                        for (size_t x = 0; x < OW; ++x) {
                            padded_img({n, c, j + y * SH, i + x * SW}) += col_arr({n, c, j, i, y, x});
						}
                    }
                }
            }
        }
    }

	return padded_img.slice({{0,N}, {0,C}, {PH, PH+H}, {PW, PW+W}});

}





}
