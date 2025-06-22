#include "deepczero.hpp"

#include <iostream>

using namespace std;

void test_mnist_dataset_show() {
	
	MNISTDataset dataset(true);
	cout << "dataset.shape(): ";
	for (auto s : dataset.get_data().get_shape())
		cout << s << ", ";
	cout << endl;
	dataset.show(2);

	MNISTDataset testset(false);
	cout << "testset.shape(): ";
	for (auto s : testset.get_data().get_shape())
		cout << s << ", ";
	cout << endl;
	testset.show(2);

	cout << "dataset.get_data().get_shape()" << endl;
	for (auto s : dataset.get_data(0).get_shape())
		cout << s << ", ";
	cout << endl;

}

int main() {
	test_mnist_dataset_show();
	
}
