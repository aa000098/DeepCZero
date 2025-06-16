#include "deepczero.hpp"

#include <cassert>

void test_spiral_dataset() {
    size_t num_data = 100;
    size_t num_class = 3;
    size_t input_dim = 2;

    // Train dataset
    SpiralDataset train_ds(num_data, num_class, true);
    Tensor<> train_x = train_ds.get_data();
    Tensor<> train_t = train_ds.get_label();

    assert(train_x.get_shape()[0] == num_data * num_class);
    assert(train_x.get_shape()[1] == input_dim);
    assert(train_t.get_shape()[0] == num_data * num_class);

    std::cout << "[✅] Train dataset shape = (" 
              << train_x.get_shape()[0] << ", " << train_x.get_shape()[1] << ")\n";

    // Test dataset
    SpiralDataset test_ds(num_data, num_class, false);
    Tensor<> test_x = test_ds.get_data();
    Tensor<> test_t = test_ds.get_label();

    assert(test_x.get_shape()[0] == num_data * num_class);
    assert(test_x.get_shape()[1] == input_dim);
    assert(test_t.get_shape()[0] == num_data * num_class);

    // Check label range
    for (size_t i = 0; i < test_t.get_shape()[0]; ++i) {
        float label = test_t.raw_data()[i];
        assert(label >= 0 && label < static_cast<float>(num_class));
    }
	std::cout << test_ds.size() << std::endl;

    std::cout << "[✅] Test dataset label range OK\n";
    std::cout << "[🎉] SpiralDataset test passed!\n";
}

int main() {
	test_spiral_dataset();
}
