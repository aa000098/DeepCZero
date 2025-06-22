#pragma once 

#include "dataset/transform/transform.hpp"


template<typename T>
Tensor<T> Compose<T>::operator()(const Tensor<T>& input) const {
	Tensor<T> out = input;
	for (auto& t : transforms)
		out = (*t)(out);
	return out;
}


template<typename T>
Tensor<T> Normalize<T>::operator()(const Tensor<T>& input) const {
	return (input - mean) / std;
}

template<typename T>
Tensor<T> Flatten<T>::operator()(const Tensor<T>& input) const {
	std::vector<T> flat_data = input.data();
	return Tensor<T>({flat_data.size()}, flat_data);
}

template<typename SrcT, typename DstT>
Tensor<DstT> AsType<SrcT, DstT>::operator()(const Tensor<SrcT>& input) const {
	return cast_tensor<SrcT, DstT>(input);
}

template<typename T>
Tensor<T> ToFloat<T>::operator()(const Tensor<T>& input) const {
	return AsType<T, float>()(input);
}
