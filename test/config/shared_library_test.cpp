#include "deepczero.hpp"

int main() {
	Variable v1({3.14f});
	Variable v2({3});
	Variable v3 = ((v1*2) + (v2/3)) / 2;
	v3.backward();
	v3.show();
	v2.show();
	v1.show();
}
