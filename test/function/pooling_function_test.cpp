#include "deepczero.hpp"

#include <cassert>
#include <iostream>

void test_pooling_forward_backward() {
    using namespace function;

    // 입력 텐서: N=1, C=1, H=W=4
    // 값: 1..16 (행우선 증가)
    Tensor<> x({1, 1, 4, 4});
    {
        float v = 1.0f;
        for (size_t h = 0; h < 4; ++h) {
            for (size_t w = 0; w < 4; ++w) {
                x({0, 0, h, w}) = v++;
            }
        }
    }
    Variable vx(x);

    // 풀링 파라미터: kernel=2x2, stride=2x2, pad=0
    std::pair<size_t,size_t> kernel = {2, 2};
    std::pair<size_t,size_t> stride = {2, 2};
    std::pair<size_t,size_t> pad    = {0, 0};

    // 1) Forward 검증: 기대 출력은 [[6,8],[14,16]]
    Variable vy = pooling(vx, kernel, stride, pad);
    const Tensor<>& y = vy.data();

    assert(y.get_shape() == std::vector<size_t>({1, 1, 2, 2}));
	std::cout << "[y shape test passed]" << std::endl;
    {
        // ravel 순서로 기대값 나열 (NCHW = 1x1x2x2)
        Tensor<> y_expected({1,1,2,2}, {
            6.0f, 8.0f,
            14.0f, 16.0f
        });
        assert(is_allclose(y, y_expected) && "Pooling forward output mismatch");
    }
	std::cout << "[y value test passed]" << std::endl;

    // 2) Backward 검증 #1: gy = 1 (모두 1)
    // 비겹침 윈도우에서 최대값 위치로만 1이 흘러감
	vy.backward();
    const Tensor<>& gx1 = vx.grad().data();

    assert(gx1.get_shape() == std::vector<size_t>({1, 1, 4, 4}) && "gx shape must be [1,1,4,4]");
	std::cout << "[gx shape test passed]" << std::endl;

    // 기대 gx: 2x2 블록의 우하단(윈도우 내 argmax=3 위치)에만 1
    // 전역 좌표로는 (1,1), (1,3), (3,1), (3,3)
    {
        Tensor<> zero({1,1,4,4});
        // 비교를 위해 zero에 기대 위치만 1로 세팅
        zero({0,0,1,1}) = 1.0f;
        zero({0,0,1,3}) = 1.0f;
        zero({0,0,3,1}) = 1.0f;
        zero({0,0,3,3}) = 1.0f;

        // gx1 == zero 확인
        Tensor<> diff = gx1 - zero;
        // 모두 0이어야 함
        //Tensor<> diff_abs = diff.abs();
        // 합이 0인지 간단 체크
        float sum = 0.0f;
        Tensor<> flat = diff.ravel();
        for (size_t i = 0; i < flat.size(); ++i) sum += flat({i}) > 0 ? static_cast<float>(flat({i})) : -1 * static_cast<float>(flat({i}));
        assert(std::fabs(sum) < 1e-6f && "Pooling backward (ones) mismatch");
    }
	std::cout << "[gx value test passed]" << std::endl;


    // 3) (선택) Pooling2DWithIndexes 검증: same output as forward
    // Pooling 내부에 저장된 indexes/input_shape에 접근 가능하다는 가정 하에서만 사용.
    // 접근이 불가하면 이 블록은 주석 처리해도 무방.
    {
		Tensor<> col = im2col_array(x, kernel, stride, pad, false);
		auto col_shape = col.get_shape();
		size_t N = col_shape[0]; 
		size_t C = col_shape[1]; 
		size_t KH = col_shape[2]; 
		size_t KW = col_shape[3]; 
		size_t OH = col_shape[4]; 
		size_t OW = col_shape[5]; 
		col = col.reshape({N, C, KH*KW, OH, OW});
		Tensor<size_t> indexes = col.argmax(2);

		const Variable& gy = vy.grad(); 
		Variable ggy = pooling2d_with_indexes(gy, indexes, vx.shape(), kernel, stride, pad);

        const Tensor<>& y2 = ggy.data();

        Tensor<> diff = y2 - y;
        float sum = 0.0f;
        Tensor<> flat = diff.ravel();
        for (size_t i = 0; i < flat.size(); ++i) sum += flat({i}) > 0 ? static_cast<float>(flat({i})) : -1 * static_cast<float>(flat({i}));
        assert(std::fabs(sum) < 1e-6f && "Pooling2DWithIndexes forward mismatch");
    }

    // 모두 통과
    std::cout << "[OK] test_pooling_forward_backward passed.\n";
}

int main() {
    test_pooling_forward_backward();
    return 0;
}
