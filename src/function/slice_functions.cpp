#include "function/slice_functions.hpp"
#include "container/tensor/tensor_all.hpp"
#include "function/ops/slice_ops.hpp"


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


// SliceAxis: slice along specified axis
Variable function::SliceAxis::forward(const std::vector<Variable>& xs) {
	dcz::Device orig_device = xs[0].device();
	const Tensor<> x = !orig_device.is_cpu() ? xs[0].data().cpu() : xs[0].data();
	input_shape = x.get_shape();
	Tensor<> result = x.slice(axis, start, end).contiguous();
	if (!orig_device.is_cpu()) result = result.to(orig_device);
	return Variable(result);
}

std::vector<Variable> function::SliceAxis::backward(const Variable& gy) {
	dcz::Device orig_device = gy.device();
	Tensor<> gy_cpu = !orig_device.is_cpu() ? gy.data().cpu() : gy.data();

	// Scatter gradient back to the full shape (zeros elsewhere)
	Tensor<> gx(input_shape, 0.0f);

	int ax = axis;
	if (ax < 0) ax += static_cast<int>(input_shape.size());

	// Compute iteration bounds
	size_t ndim = input_shape.size();
	size_t outer = 1, inner = 1;
	for (int d = 0; d < ax; d++) outer *= input_shape[d];
	for (size_t d = ax + 1; d < ndim; d++) inner *= input_shape[d];

	size_t slice_len = end - start;
	size_t full_axis_len = input_shape[ax];

	const auto& gy_raw = gy_cpu.raw_data();
	auto& gx_raw = gx.raw_data();

	for (size_t o = 0; o < outer; o++) {
		for (size_t s = 0; s < slice_len; s++) {
			size_t src_offset = (o * slice_len + s) * inner;
			size_t dst_offset = (o * full_axis_len + (start + s)) * inner;
			for (size_t i = 0; i < inner; i++) {
				gx_raw[dst_offset + i] += gy_raw[src_offset + i];
			}
		}
	}

	if (!orig_device.is_cpu()) gx = gx.to(orig_device);
	return { Variable(gx) };
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
