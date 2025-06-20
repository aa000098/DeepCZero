#include "deepczero.hpp"

#include <cassert>
#include <iostream>

void test_dataloader_spiral_dataset() {
    size_t num_data = 100;
    size_t num_class = 3;
    size_t batch_size = 10;

    SpiralDataset train_set(num_data, num_class, true);
    SpiralDataset test_set(num_data, num_class, false);

    DataLoader train_loader(train_set, batch_size, true);
    DataLoader test_loader(test_set, batch_size, false);

    size_t total_seen = 0;
	size_t max_epoch = 1;

	for (size_t i = 0; i < max_epoch; i++) {
	    while (train_loader.has_next()) {
        	auto [x_batch, y_batch] = train_loader.next();

        	// 체크: 배치 shape
        	auto x_shape = x_batch.get_shape();  // [B, D]
        	auto y_shape = y_batch.get_shape();  // [B] or [B, 1]

        	assert(x_shape[0] <= batch_size);
        	assert(x_shape[1] == 2);  // Spiral 데이터의 feature dim = 2

        	assert(y_shape[0] == x_shape[0]);  // 라벨 수 = 입력 수

        	total_seen += x_shape[0];
		}
    }

    // 전체 배치 개수 누적 = 전체 데이터 수
    assert(total_seen == num_data * num_class);
    std::cout << "[PASS] DataLoader test with SpiralDataset (" << total_seen << " samples)\n";
}

int main() {
	test_dataloader_spiral_dataset();
}
