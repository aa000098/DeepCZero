#include "function/slice_functions.hpp"
#include "container/tensor/tensor_all.hpp"
#include "ops/slice_ops.hpp"


Variable function::GetItem::forward(const std::vector<Variable>& xs) {
    const Tensor<>& x_data = xs[0].data();
	
	Tensor<> result = x_data;
	for (size_t idx : slices)
		result = result[idx];

	return Variable(result);
}

std::vector<Variable> function::GetItem::backward(const Variable& gy) {
	const Variable& x = inputs[0];
	return { get_item_grad(gy, slices, x.shape()) };
}


Variable function::GetItemGrad::forward(const std::vector<Variable>& gys) {
    const Tensor<>& gy_data = gys[0].data();
	Tensor<> gx_data(in_shape);

	add_at(gx_data, slices, gy_data);
	return Variable(gx_data);

}

std::vector<Variable> function::GetItemGrad::backward(const Variable& ggy) {
	return { get_item(ggy, slices) };
}
