#include "function/ops/conv_ops.hpp"
#include "function/function_all.hpp"

#include <memory>


// [Convolution]

Variable conv2d(const Variable &x,
				const Variable &W,
				const Variable &b,
				std::pair<size_t, size_t> stride={1,1},
				std::pair<size_t, size_t> pad={0,0}) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Conv2d>(stride, pad);
	return (*f)({x, W, b});
}
				
Variable deconv2d(const Variable &x,
				const Variable &W,
				const Variable &b,
				std::pair<size_t, size_t> stride={1,1},
				std::pair<size_t, size_t> pad={0,0},
				std::pair<size_t, size_t> outsize={0,0}) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Deconv2d>(stride, pad, outsize);
	return (*f)({x, W, b});
}

Variable conv2dgradw(const Variable &x,
					const Variable &gy,
					const Variable &W,
					std::pair<size_t, size_t> stride={1,1},
					std::pair<size_t, size_t> pad={0,0}) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Conv2dGradW>(W, stride, pad);
	return (*f)({x, gy});
}
				


// [Im2col]

Variable im2col(const Variable& x,
				std::pair<size_t, size_t> kernel_size,
				std::pair<size_t, size_t> stride,
				std::pair<size_t, size_t> pad,
				bool to_matrix) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Im2col>(kernel_size, stride, pad, to_matrix);
	return (*f)(x);
}

Variable col2im(const Variable& x,
				std::vector<size_t> input_shape,
				std::pair<size_t, size_t> kernel_size,
				std::pair<size_t, size_t> stride,
				std::pair<size_t, size_t> pad,
				bool to_matrix) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<Col2im>(input_shape, kernel_size, stride, pad, to_matrix);
	return (*f)(x);
}

