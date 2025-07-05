#include "deepczero.hpp"

#include <iostream>
#include <cassert>

void test_get_conv_outsize() {
	int H = 4, W = 4;
	int KH = 3, KW = 3;
	int SH = 1, SW = 1;
	int PH = 1, PW = 1;

	int OH = get_conv_outsize(H, KH, SH, PH);
	int OW = get_conv_outsize(W, KW, SW, PW);


    // 예상 출력: OH = 4, OW = 4
    assert(OH == 4);
    assert(OW == 4);

    std::cout << "✅ test_get_conv_outsize passed! Output: (" << OH << ", " << OW << ")\n";
}


int main() {
	test_get_conv_outsize();
}
