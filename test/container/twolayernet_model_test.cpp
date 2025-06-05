#include "deepczero.hpp"

#include <cmath>

double pi = std::acos(-1.0f);

using layer::Linear;
using layer::MLP;

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

void test_twolayernet_learning() {
	Variable x(rand(10, 1), "x");
	Variable y = sin(2 * pi * x) + rand(10, 1);

	float lr = 0.2;
	size_t max_iter = 1000;
	size_t hidden_size = 10;

	TwoLayerNet model(hidden_size, 1);

	Variable y_pred;
	Variable loss;

	for (size_t i = 0; i < max_iter; i++) {
		y_pred = model(x);
		loss = mean_squared_error(y, y_pred);

		model.cleargrads();
		loss.backward();

		for (auto& p : model.get_params()) 
			p.data() -= lr * p.grad().data();

		if (i % 1000 == 0)
			loss.show();

	}
}

void test_mlp_declaration() {
	MLP model1({10, 1}); 
	MLP model2({10, 20, 30, 40, 1});
}

int main() {
	test_twolayernet_forward_and_plot();
	test_twolayernet_learning();
}
