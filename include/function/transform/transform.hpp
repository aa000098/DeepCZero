#pragma once

#include "container/tensor/tensor_all.hpp"

#include <memory>

template<typename T>
class Transform {
public:
	virtual Tensor<T> operator()(const Tensor<T>& input) const = 0;
	virtual ~Transform() = default;
};

template <typename T>
class Compose : public Transform<T> {
private:
	std::vector<std::shared_ptr<Transform<T>>> transforms;

public:
	Compose(const std::vector<std::shared_ptr<Transform<T>>>& tfs) : transforms(tfs) {}

	Tensor<T> operator()(const Tensor<T>& input) const override {
		Tensor<T> out = input;
		for (auto& t : transforms)
			out = (*t)(out);
		return out;
	}
};


template<typename T>
class Normalize : public Transform<T> {
private:
	T mean;
	T std;

public:
	Normalize(T mean, T std) : mean(mean), std(std) {};

	Tensor<T> operator()(const Tensor<T>& input) const override {
		Tensor<T> out = input;
		auto& data = out.raw_data();
		for (auto& x : data)
			x = (x - mean) / std;
		return out;
	}
};


