#include "deepczero.hpp"

using layer::Linear;

class TwoLayerNet : public Model {

public:
	TwoLayerNet(size_t hidden_size, size_t out_size) {
		std::shared_ptr<Layer> l1 = std::make_shared<Linear>(hidden_size);
		std::shared_ptr<Layer> l2 = std::make_shared<Linear>(out_size);
		register_sublayers("l1", l1);
		register_sublayers("l2", l2);
	}

	Variable forward(const std::vector<Variable>& xs) override {
		Variable x = xs[0];

		std::shared_ptr<Layer> l1 = get_sublayer("l1");
		std::shared_ptr<Layer> l2 = get_sublayer("l2");

		Variable y = sigmoid((*l1)(x));
		y = (*l2)(y);
		return y;
	}
};

void test_twolayernet_forward_and_plot() {
	Variable x(randn(5, 10), "x");
	TwoLayerNet model(100, 10);
	model.plot({x});

}

int main() {
	test_twolayernet_forward_and_plot();
}
