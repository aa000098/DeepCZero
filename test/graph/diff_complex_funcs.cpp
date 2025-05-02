#include "deepczero.hpp"

Variable sphere(Variable x, Variable y) {
	Variable z = (x^2) + (y^2);
	return z;
}

Variable matyas(Variable x, Variable y) {
	Variable z = (0.26 * ((x^2) + (y^2))) - (0.48 * x * y);
	return z;
}

Variable goldstein(Variable x, Variable y) {
	Variable z = (1 + ((x+y+1)^2) * (19 - (14*x) + (3*(x^2)) - (14*y) + (6*x*y) + 3*(y^2))) * (30 + ((2*x - 3*y)^2) * (18 - 32*x + 12*(x^2) + 48*y - 36*x*y + 27*(y^2)));
	return z;
}

int main() {
// sphere function
	Variable x1({1});
	Variable y1({1});
	Variable z1 = sphere(x1,y1);
	z1.backward();
	z1.show();
	y1.show();
	x1.show();

// matyas function
	Variable x2({1});
	Variable y2({1});
	Variable z2 = matyas(x2,y2);
	z2.backward();
	z2.show();
	y2.show();
	x2.show();

// matyas function
	Variable x3({1});
	Variable y3({1});
	Variable z3 = goldstein(x3,y3);
	z3.backward();
	z3.show();
	y3.show();
	x3.show();
}
