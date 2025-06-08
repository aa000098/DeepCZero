#include "deepczero.hpp"

void test_tesnor_random_initialize() {

	Tensor x1 = randn(3, 2);
	Tensor x2 = rand(3, 2);

	x1.show();
	x2.show();
}

int main() {
	test_tesnor_random_initialize();
}
