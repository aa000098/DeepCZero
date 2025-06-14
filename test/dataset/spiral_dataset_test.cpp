#include "deepczero.hpp"

#include <cassert>

void test_spiral_dataset() {
    size_t num_data = 100;
    size_t num_class = 3;
    size_t input_dim = 2;

    // Train dataset
    SpiralDataset train_ds(num_data, num_class, true);
    Variable train_x = train_ds.get_data();
    Variable train_t = train_ds.get_target();
	train_t.show();

    assert(train_x.shape()[0] == num_data * num_class);
    assert(train_x.shape()[1] == input_dim);
    assert(train_t.shape()[0] == num_data * num_class);

    std::cout << "[✅] Train dataset shape = (" 
              << train_x.shape()[0] << ", " << train_x.shape()[1] << ")\n";

    // Test dataset
    SpiralDataset test_ds(num_data, num_class, false);
    Variable test_x = test_ds.get_data();
    Variable test_t = test_ds.get_target();

    assert(test_x.shape()[0] == num_data * num_class);
    assert(test_x.shape()[1] == input_dim);
    assert(test_t.shape()[0] == num_data * num_class);

    // Check label range
    for (size_t i = 0; i < test_t.shape()[0]; ++i) {
        float label = test_t.data().raw_data()[i];
        assert(label >= 0 && label < static_cast<float>(num_class));
    }

    std::cout << "[✅] Test dataset label range OK\n";
    std::cout << "[🎉] SpiralDataset test passed!\n";
}

int main() {
	test_spiral_dataset();
}
