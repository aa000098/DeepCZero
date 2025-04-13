#include "ops/ops.hpp"
#include "container/variable.hpp"

int main() {
	Variable x({0.5});

	Variable y = square(exp(square(x)));
	
	y.backward();

	y.show();
	x.show();
}
