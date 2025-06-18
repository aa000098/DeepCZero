#include "function/evaluation_functions.hpp"
#include "container/tensor/tensor_all.hpp"

namespace function {
	
	Variable Accuracy::forward(const std::vector<Variable>& xs) {
		return this->forward(xs[0], xs[1]);
	}

	Variable Accuracy::forward(const Variable& y, const Variable& t) {
		const Tensor<>&y_data = y.data();
		const Tensor<>&t_data = t.data();

		Tensor<size_t> pred = y_data.argmax(1).reshape(t.shape());
		Tensor<uint8_t> result = pred.equal(cast_tensor<float, size_t>(t_data));
		//result.show();
		Tensor<float> result_uint8 = cast_tensor<uint8_t, float>(result);
		float acc = result_uint8.mean();
		return Variable({acc});
	}
}
