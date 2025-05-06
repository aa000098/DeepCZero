#pragma once

#include "container/tensor/tensor.hpp"
#include "container/tensor/tensor_ops.hpp"

namespace tensor {

// Tensor ⊕ Tensor

template<typename T>
Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b) {
	return add(a, b);
}

template<typename T>
Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b) {
	return sub(a, b);
}

template<typename T>
Tensor<T> operator*(const Tensor<T>& a, const Tensor<T>& b) {
	return mul(a, b);
}

template<typename T>
Tensor<T> operator/(const Tensor<T>& a, const Tensor<T>& b) {
	return div(a, b);
}

// Tensor ⊕ Scalar

template<typename T>
Tensor<T> operator+(const Tensor<T>& a, T scalar) {
    Tensor<T> result = a.clone();
    add_scalar_inplace(result, scalar);
    return result;
}

template<typename T>
Tensor<T> operator-(const Tensor<T>& a, T scalar) {
    Tensor<T> result = a.clone();
    sub_scalar_inplace(result, scalar);
    return result;
}

template<typename T>
Tensor<T> operator*(const Tensor<T>& a, T scalar) {
    Tensor<T> result = a.clone();
    mul_scalar_inplace(result, scalar);
    return result;
}

template<typename T>
Tensor<T> operator/(const Tensor<T>& a, T scalar) {
    Tensor<T> result = a.clone();
    div_scalar_inplace(result, scalar);
    return result;
}

// Scalar ⊕ Tensor

template<typename T>
Tensor<T> operator+(T scalar, const Tensor<T>& a) {
    Tensor<T> result = a.clone();
    add_scalar_inplace(result, scalar);
    return result;
}

template<typename T>
Tensor<T> operator-(T scalar, const Tensor<T>& a) {
    Tensor<T> result = a.clone();
    auto& data = result.raw_data();
    for (auto& val : data)
        val = scalar - val;
    return result;
}

template<typename T>
Tensor<T> operator*(T scalar, const Tensor<T>& a) {
    Tensor<T> result = a.clone();
    mul_scalar_inplace(result, scalar);
    return result;
}

// Unary minus: -Tensor
template<typename T>
Tensor<T> operator-(const Tensor<T>& a) {
	return neg(a);	
}

}
