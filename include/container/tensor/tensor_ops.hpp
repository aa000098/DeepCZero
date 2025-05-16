#pragma once

#include "container/tensor/tensor.hpp"
#include "container/tensor/tensor_functions.hpp"

namespace tensor {

template<typename T>
Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b) {

	std::vector<size_t> shape_a = a.get_shape();
	std::vector<size_t> shape_b = b.get_shape();
	size_t ndim = std::max(shape_a.size(), shape_b.size());

	while (shape_a.size() < ndim) shape_a.insert(shape_a.begin(), 1);
	while (shape_b.size() < ndim) shape_b.insert(shape_b.begin(), 1);

	std::vector<size_t> broadcast_shape(ndim);
	for (size_t i = 0; i < ndim; i++) {
		if (shape_a[i] == shape_b[i])
			broadcast_shape[i] = shape_a[i];
		else if (shape_a[i] == 1)
			broadcast_shape[i] = shape_b[i];
		else if (shape_b[i] == 1)
			broadcast_shape[i] = shape_a[i];
		else
			throw std::runtime_error("Tensor Sum: Shape mismatch (broadcasting failed)");
	}

	Tensor<T> a_bc = broadcast_to(a, broadcast_shape);
	Tensor<T> b_bc = broadcast_to(b, broadcast_shape);

	// TODO: SIMD optimization needed
    Tensor<T> result(broadcast_shape, T{});
    size_t total = result.size();
    size_t ndim_result = broadcast_shape.size();

    for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
        std::vector<size_t> idx(ndim_result);
        size_t rem = flat_idx;
        for (size_t i = ndim_result; i-- > 0;) {
            idx[i] = rem % broadcast_shape[i];
            rem /= broadcast_shape[i];
        }

        result(idx) = a_bc(idx) + b_bc(idx);
    }

	return result;

}

template<typename T>
Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b) {

	std::vector<size_t> shape_a = a.get_shape();
	std::vector<size_t> shape_b = b.get_shape();
	size_t ndim = std::max(shape_a.size(), shape_b.size());

	// ① 차원 맞춤
	while (shape_a.size() < ndim) shape_a.insert(shape_a.begin(), 1);
	while (shape_b.size() < ndim) shape_b.insert(shape_b.begin(), 1);

	// ② broadcast shape 결정
	std::vector<size_t> broadcast_shape(ndim);
	for (size_t i = 0; i < ndim; ++i) {
		if (shape_a[i] == shape_b[i])
			broadcast_shape[i] = shape_a[i];
		else if (shape_a[i] == 1)
			broadcast_shape[i] = shape_b[i];
		else if (shape_b[i] == 1)
			broadcast_shape[i] = shape_a[i];
		else
			throw std::runtime_error("Tensor Sub: Shape mismatch (broadcasting faield)");
	}

	// ③ 브로드캐스트 적용
	Tensor<T> a_bc = broadcast_to(a, broadcast_shape);
	Tensor<T> b_bc = broadcast_to(b, broadcast_shape);

	// TODO: SIMD optimization needed
    Tensor<T> result(broadcast_shape, T{});
    size_t total = result.size();
    size_t ndim_result = broadcast_shape.size();

    for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
        std::vector<size_t> idx(ndim_result);
        size_t rem = flat_idx;
        for (size_t i = ndim_result; i-- > 0;) {
            idx[i] = rem % broadcast_shape[i];
            rem /= broadcast_shape[i];
        }

        result(idx) = a_bc(idx) - b_bc(idx);
    }

	return result;
}

template<typename T>
Tensor<T> mul(const Tensor<T>& a, const Tensor<T>& b) {

	std::vector<size_t> shape_a = a.get_shape();
	std::vector<size_t> shape_b = b.get_shape();
	size_t ndim = std::max(shape_a.size(), shape_b.size());

	// ① 차원 맞춤
	while (shape_a.size() < ndim) shape_a.insert(shape_a.begin(), 1);
	while (shape_b.size() < ndim) shape_b.insert(shape_b.begin(), 1);

	// ② broadcast shape 결정
	std::vector<size_t> broadcast_shape(ndim);
	for (size_t i = 0; i < ndim; ++i) {
		if (shape_a[i] == shape_b[i])
			broadcast_shape[i] = shape_a[i];
		else if (shape_a[i] == 1)
			broadcast_shape[i] = shape_b[i];
		else if (shape_b[i] == 1)
			broadcast_shape[i] = shape_a[i];
		else
			throw std::runtime_error("Tensor Mul: Shape mismatch (broadcasting faield)");
	}

	// ③ 브로드캐스트 적용
	Tensor<T> a_bc = broadcast_to(a, broadcast_shape);
	Tensor<T> b_bc = broadcast_to(b, broadcast_shape);

	// TODO: SIMD optimization needed
    Tensor<T> result(broadcast_shape, T{});
    size_t total = result.size();
    size_t ndim_result = broadcast_shape.size();

    for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
        std::vector<size_t> idx(ndim_result);
        size_t rem = flat_idx;
        for (size_t i = ndim_result; i-- > 0;) {
            idx[i] = rem % broadcast_shape[i];
            rem /= broadcast_shape[i];
        }

        result(idx) = a_bc(idx) * b_bc(idx);
    }

	return result;

}

template<typename T>
Tensor<T> div(const Tensor<T>& a, const Tensor<T>& b) {

	std::vector<size_t> shape_a = a.get_shape();
	std::vector<size_t> shape_b = b.get_shape();
	size_t ndim = std::max(shape_a.size(), shape_b.size());

	// ① 차원 맞춤
	while (shape_a.size() < ndim) shape_a.insert(shape_a.begin(), 1);
	while (shape_b.size() < ndim) shape_b.insert(shape_b.begin(), 1);

	// ② broadcast shape 결정
	std::vector<size_t> broadcast_shape(ndim);
	for (size_t i = 0; i < ndim; ++i) {
		if (shape_a[i] == shape_b[i])
			broadcast_shape[i] = shape_a[i];
		else if (shape_a[i] == 1)
			broadcast_shape[i] = shape_b[i];
		else if (shape_b[i] == 1)
			broadcast_shape[i] = shape_a[i];
		else
			throw std::runtime_error("Tensor Div: Shape mismatch (broadcasting faield)");
	}

	// ③ 브로드캐스트 적용
	Tensor<T> a_bc = broadcast_to(a, broadcast_shape);
	Tensor<T> b_bc = broadcast_to(b, broadcast_shape);

	// TODO: SIMD optimization needed
    Tensor<T> result(broadcast_shape, T{});
    size_t total = result.size();
    size_t ndim_result = broadcast_shape.size();

    for (size_t flat_idx = 0; flat_idx < total; ++flat_idx) {
        std::vector<size_t> idx(ndim_result);
        size_t rem = flat_idx;
        for (size_t i = ndim_result; i-- > 0;) {
            idx[i] = rem % broadcast_shape[i];
            rem /= broadcast_shape[i];
        }

		if (b_bc(idx) == 0)
			throw std::runtime_error("Tensor Div: Division by zero");	
        result(idx) = a_bc(idx) / b_bc(idx);
    }

	return result;

}

template<typename T>
Tensor<T> neg(const Tensor<T>& a) {
	std::vector<T> result_data(a.raw_data());
	for (auto& val : result_data)
		val = -val;

	return Tensor<T>(a.get_shape(), result_data);
}

// inplace
template<typename T>
void add_inplace(Tensor<T>& a, const Tensor<T>& b) {
	auto& data = a.raw_data();
	const auto& b_data = b.raw_data();
	if (a.get_shape() != b.get_shape())
		throw std::runtime_error("Shape mismatch in add_inplace");
	for (size_t i = 0; i < data.size(); ++i)
		data[i] += b_data[i];
}

template<typename T>
void sub_inplace(Tensor<T>& a, const Tensor<T>& b) {
	auto& data = a.raw_data();
	const auto& b_data = b.raw_data();
	if (a.get_shape() != b.get_shape())
		throw std::runtime_error("Shape mismatch in sub_inplace");
	for (size_t i = 0; i < data.size(); ++i)
		data[i] -= b_data[i];
}

template<typename T>
void mul_inplace(Tensor<T>& a, const Tensor<T>& b) {
	auto& data = a.raw_data();
	const auto& b_data = b.raw_data();
	if (a.get_shape() != b.get_shape())
		throw std::runtime_error("Shape mismatch in mul_inplace");
	for (size_t i = 0; i < data.size(); ++i)
		data[i] *= b_data[i];
}

// Tensor ⊕ Scalar
template<typename T>
void add_scalar_inplace(Tensor<T>& a, T scalar) {
	auto& data = a.raw_data();
	for (auto& val : data)
		val += scalar;
}

// Tensor<T> -= scalar
template<typename T>
void sub_scalar_inplace(Tensor<T>& a, T scalar) {
	auto& data = a.raw_data();
	for (auto& val : data)
		val -= scalar;
}

template<typename T>
void mul_scalar_inplace(Tensor<T>& a, T scalar) {
	auto& data = a.raw_data();
	for (auto& val : data)
		val *= scalar;
}

template<typename T>
void div_scalar_inplace(Tensor<T>& a, T scalar) {
	auto& data = a.raw_data();
	for (auto& val : data)
		val /= scalar;
}


template<typename T>
Tensor<T> pow(const Tensor<T>& x, const T scalar) {
	std::vector<T> result_data(x.raw_data());
	const std::vector<T>& x_data = x.raw_data();
	for (size_t i = 0; i < result_data.size(); i++)
		result_data[i] = std::pow(x_data[i], scalar);

	return Tensor<T>(x.get_shape(), result_data);
}

template<typename T>
Tensor<T> exp(const Tensor<T>& x) {
	std::vector<T> result_data(x.raw_data());
	const std::vector<T>& x_data = x.raw_data();
	for (size_t i = 0; i < result_data.size(); i++)
		result_data[i] = std::exp(x_data[i]);

	return Tensor<T>(x.get_shape(), result_data);
}

template<typename T>
Tensor<T> sin(const Tensor<T>& x) {
	std::vector<T> result_data(x.raw_data());
	const std::vector<T>& x_data = x.raw_data();
	for (size_t i = 0; i < result_data.size(); i++)
		result_data[i] = std::sin(x_data[i]);

	return Tensor<T>(x.get_shape(), result_data);
}

template<typename T>
Tensor<T> cos(const Tensor<T>& x) {
	std::vector<T> result_data(x.raw_data());
	const std::vector<T>& x_data = x.raw_data();
	for (size_t i = 0; i < result_data.size(); i++)
		result_data[i] = std::cos(x_data[i]);

	return Tensor<T>(x.get_shape(), result_data);
}

template<typename T>
Tensor<T> tanh(const Tensor<T>& x) {
	std::vector<T> result_data(x.raw_data());
	const std::vector<T>& x_data = x.raw_data();
	for (size_t i = 0; i < result_data.size(); i++)
		result_data[i] = std::tanh(x_data[i]);

	return Tensor<T>(x.get_shape(), result_data);
}

template<typename T>
Tensor<T> dot(const Tensor<T>& a, const Tensor<T>& b) {
	std::vector<T> result_data(a.raw_data());
	
	const std::vector<T>& x_data = a.raw_data(); 

    Tensor<T> result(result_data);

	return result;
}

}
