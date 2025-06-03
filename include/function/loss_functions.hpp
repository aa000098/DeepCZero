#include "function/function.hpp"

namespace function {
class MeanSquaredError : public Function {

public:
	Variable forward(const std::vector<Variable>& xs) override;
	std::vector<Variable> backward(const Variable& gy) override;
	~MeanSquaredError() = default;
};

}
