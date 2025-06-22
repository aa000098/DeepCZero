#pragma once

#include "container/tensor/tensor_all.hpp"

#include <memory>

using namespace tensor;

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
	Compose(const std::initializer_list<std::shared_ptr<Transform<T>>>& tfs) : transforms(tfs) {}

	Tensor<T> operator()(const Tensor<T>& input) const override;
};


template<typename T>
class Normalize : public Transform<T> {
private:
	T mean;
	T std;

public:
	Normalize(T mean = 0, T std = 1) 
		: mean(mean), std(std) {};

	Tensor<T> operator()(const Tensor<T>& input) const override;
};


template<typename T>
class Flatten : public Transform<T> {
public:
	Tensor<T> operator()(const Tensor<T>& input) const override;
};

template<typename SrcT, typename DstT>
class AsType : public Transform<DstT> {
public:
	Tensor<DstT> operator()(const Tensor<SrcT>& input) const override;
};

template<typename T>
class ToFloat : public Transform<T> {
public:
	Tensor<T> operator()(const Tensor<T>& input) const override;
};

#include "dataset/transform/transform.tpp"
