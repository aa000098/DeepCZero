#include "function/ops/slice_ops.hpp"
#include "function/function_all.hpp"


Variable get_item(const Variable& x, 
					const std::vector<size_t> slices) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<GetItem>(slices);
	return (*f)({x});

}

Variable get_item_grad(const Variable& gy,
						const std::vector<size_t> slices,
						const std::vector<size_t> in_shape) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<GetItemGrad>(slices, in_shape);
	return (*f)({gy});

}

Variable slice_axis(const Variable& x, int axis, size_t start, size_t end) {
	using namespace function;
	std::shared_ptr<Function> f = std::make_shared<SliceAxis>(axis, start, end);
	return (*f)({x});
}
