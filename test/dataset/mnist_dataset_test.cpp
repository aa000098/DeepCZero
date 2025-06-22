#include "deepczero.hpp"

void test_mnist_dataset_show() {
	MNISTDataset dataset(true);
	MNISTDataset testset(false);
	dataset.show(2);
	testset.show(2);
}

int main() {
	test_mnist_dataset_show();
	
}
