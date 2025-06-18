#pragma once 

#include "function/function.hpp"

namespace function {

	class Accuracy {
	public:

		Variable forward(const std::vector<Variable>& xs);
		Variable forward(const Variable& y, const Variable& t);
		Variable operator()(const Variable& y, const Variable& t) { return forward(y, t); };
	};

}
