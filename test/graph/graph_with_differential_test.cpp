#include "deepczero.hpp"

int main() {
	Variable x({2});
	Variable y = x^2;

	y.backward(false, true);

	Variable gx = x.grad();

	x.cleargrad();

	Variable z = (gx^3) + y;
	z.backward();
	x.show();

}
