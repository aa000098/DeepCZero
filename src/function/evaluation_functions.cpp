#include "function/evaluation_functions.hpp"
#include "container/tensor/tensor_all.hpp"

namespace function {
	
	Variable Accuracy::forward(const std::vector<Variable>& xs) {
		return this->forward(xs[0], xs[1]);
	}

	Variable Accuracy::forward(const Variable& y, const Variable& t) {
		// argmax/equal use raw_data(), fall back to CPU for device tensors
		const Tensor<> y_data = y.data().is_device() ? y.data().cpu() : y.data();
		const Tensor<> t_data = t.data().is_device() ? t.data().cpu() : t.data();
		auto t_shape = t.shape();

		Tensor<size_t> pred = y_data.argmax(1).reshape(t_shape);
		Tensor<uint8_t> result = pred.equal(cast_tensor<float, size_t>(t_data));
		Tensor<float> result_uint8 = cast_tensor<uint8_t, float>(result);
		float acc = result_uint8.mean();
		return Variable({acc});
	}
}
