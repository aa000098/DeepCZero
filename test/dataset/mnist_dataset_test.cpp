#include "deepczero.hpp"

void test_mnist_dataset_show() {
	MNISTDataset dataset(true);
	dataset.show(0);
}

int main() {
	test_mnist_dataset_show();

}
