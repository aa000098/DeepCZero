#include "deepczero.hpp"

#include <cmath>

double pi = std::acos(-1.0f);

void neural_network_test() {
    std::cout << "▶️ Running Neural Network test...\n";

	Variable x = rand_tensor(	/* rows */	100, 
								/* cols */	1, 
								/* seed */	0);
 	
	Variable y = sin(2 * pi * x) + rand_tensor(100, 1);
	y.show();	

	size_t I = 1;
	size_t H = 10;
	size_t O = 1;

}

int main() {
	neural_network_test();
	return 0;
}
