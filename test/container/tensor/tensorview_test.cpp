#include "container/tensor/tensorview.hpp"
#include "container/tensor/tensorND.hpp"
#include "container/tensor/tensor.hpp"

#include <memory>
#include <cassert>

using namespace tensor;

void test_tensor_view() {
    // 텐서 생성 및 초기 상태 확인
    TensorND<float> tensorND({4, 3, 2});
    Tensor<float> tensor(std::make_shared<TensorND<float>>(tensorND));

    auto shape = tensor.get_shape();
    assert((shape == std::vector<size_t>{4, 3, 2}));
    auto strides = tensor.get_strides();
    assert((strides == std::vector<size_t>{6, 2, 1}));

    // View 생성: 1D view with shape [2], stride [1]
    TensorView<float> view({2}, tensor.shared_data(), {1});

    assert(view.ndim() == 1);
    assert(view.size() == 2);
    assert((view.get_shape() == std::vector<size_t>{2}));
    assert((view.get_strides() == std::vector<size_t>{1}));

    // View에 값 쓰기
    view({1}) = 2;

    // View와 tensor가 데이터 공유하는지 확인
    assert(tensor.raw_data()[1] == 2);
    assert(view({1}) == 2);

    // tensor의 다른 위치에 값 할당
    tensor({0, 0, 0}) = 4;
    tensor({1, 0, 1}) = 3;
    tensor({2, 1, 1}) = 5;

    assert(tensor({0, 0, 0}) == 4);
    assert(tensor({1, 0, 1}) == 3);
    assert(tensor({2, 1, 1}) == 5);

    // View는 tensor와 같은 메모리 보므로 값이 반영돼 있어야 함
    assert(view({1}) == 2);

    // TensorView show 대신 수동 체크
    std::cout << "[PASS] tensor.view(), indexing, assignment, sharing test\n";

	tensor.show();
    // Indexing test: tensor[1] is shape [3, 2]
    Tensor<float> slice1 = tensor[1];
	slice1.show();
    assert((slice1.get_shape() == std::vector<size_t>{3, 2}));

    Tensor<float> slice2 = tensor[2][1];
	slice2.show();
    assert((slice2.get_shape() == std::vector<size_t>{2}));
    assert(slice2({1}) == 5);

    std::cout << "[PASS] tensor[i] and tensor[i][j] indexing test\n";
}

void test_is_contiguous() {
	std::vector<float> data(12);
	for (size_t i = 0; i < data.size(); ++i) data[i] = static_cast<float>(i);

	Tensor<float> a({1, 3, 4}, data);
	Tensor<float> moved = a.transpose({1, 0, 2});
	assert((moved.get_shape() == std::vector<size_t>{3, 1, 4}));
	assert(moved.is_contiguous());
	Tensor<float> moved_c = moved.contiguous();
	assert(moved_c.shared_data() == a.shared_data());
	assert(moved_c({2, 0, 3}) == 11.0f);

	Tensor<float> b({4, 3}, data);
	Tensor<float> bt = b.transpose({1, 0});
	assert(!bt.is_contiguous());

	Tensor<float> row = b[0];
	assert(!row.is_contiguous());
	assert(row.contiguous().raw_data().size() == 3);

	Tensor<float> col = bt.slice(1, 0, 1);
	assert((col.get_shape() == std::vector<size_t>{3, 1}));
	assert(!col.is_contiguous());
	Tensor<float> col_c = col.clone();
	assert(col_c.raw_data().size() == 3);
	assert(col_c({2, 0}) == 2.0f);

	std::cout << "[PASS] is_contiguous ignores size-1 strides and rejects partial views\n";
}

int main() {
	test_tensor_view();
	test_is_contiguous();
}
