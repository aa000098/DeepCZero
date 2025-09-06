#pragma once

#include "container/tensor/tensor.hpp"


namespace tensor {


template<typename T>
Tensor<T> broadcast_to(const Tensor<T>& src, const std::vector<size_t>& target_shape);


template<typename T>
Tensor<T> sum_to(const Tensor<T>& src, const std::vector<size_t>& target_shape);


template<typename T>
void add_at(Tensor<T>& gx, 
			const std::vector<size_t>& slices,
			const Tensor<T>& gy);

template<typename T>
std::tuple<Tensor<T>, Tensor<T>, std::vector<size_t>> broadcast_binary_operands(
    const Tensor<T>& a, const Tensor<T>& b);


template<typename T>
Tensor<T> arrange(size_t size) {
	std::vector<T> values(size);
	std::iota(values.begin(), values.end(), 0);

	return Tensor<T>({size}, values);
}

template<typename T>
Tensor<T> arrange(size_t start, size_t end, size_t step) {
	if (step == 0) throw std::runtime_error("arrange: step must be non-zero");

	std::vector<T> v;
	if ((step > 0 && start < step) || (step < 0 && start > step)) {
		for (T x = start; (step > 0) ? (x < step) : (x > step); x = static_cast<T>(x + step))
			v.push_back(x);
	}
	return Tensor<T>({v.size()}, v);
}

template<typename T>
Tensor<T> stack(const std::vector<Tensor<T>>& tensors);


template<typename SrcT, typename DstT>
Tensor<DstT> cast_tensor(const Tensor<SrcT>& src);


template<typename T>
Tensor<T> im2col_array(	const Tensor<T> &img,
						std::pair<size_t, size_t> kernel_size,
						std::pair<size_t, size_t> stride,
						std::pair<size_t, size_t> pad,
						bool to_matrix=true);

template<typename T>
Tensor<T> col2im_array(	const Tensor<T> &col,
						std::vector<size_t> img_shape,
						std::pair<size_t, size_t> kernel_size,
						std::pair<size_t, size_t> stride,
						std::pair<size_t, size_t> pad,
						bool to_matrix=true);

template<typename T>
bool is_allclose(const Tensor<T>& a, const Tensor<T>& b, float rtol=1e-4, float atol=1e-6);

}


#include "container/tensor/tensor_functions.tpp"
