#include "container/variable_all.hpp"
#include "container/tensor/tensor_all.hpp"
#include "function/function.hpp"
#include "graph/graph.hpp"
#include "config/config.hpp"

#include <unordered_set>
#include <string>
#include <iostream>

void Variable::backward(bool retain_grad, bool create_graph) {
	if (!impl->grad)
		impl->grad = std::make_unique<Variable>(Tensor<>(impl->data.get_shape(), 1));
	impl->grad->set_name("gy");
	auto creator = impl->creator.get();
	if (!creator) return;
	Graph graph(creator);
	std::vector<Function*> topo_order = graph.get_topo_order();

	for (auto& f : topo_order) {
		std::vector<std::shared_ptr<VariableImpl<>>> inputs = f->get_inputs();
		std::shared_ptr<VariableImpl<>> output = f->get_output();
		Variable* gy = output->grad.get();

		{
			dcz::UsingConfig is_higher_order_diff(create_graph);
			std::vector<Variable> gxs = f->backward(*gy);
			for (size_t i = 0; i < gxs.size(); ++i) {
				std::shared_ptr<VariableImpl<>> input = inputs[i];
				const Variable& gx = gxs[i];
				if (!input->grad)
					input->grad = std::make_unique<Variable>(gx);
				else 
					(*input->grad) = (*input->grad) + gx;
			}
			if (!retain_grad) output->grad.reset();
		}
	}
}

void Variable::clear_graph() {
    std::unordered_set<std::uintptr_t> visited;
    clear_graph(visited);
}

void Variable::clear_graph(std::unordered_set<std::uintptr_t>& visited) {
    std::uintptr_t vid = impl->id();
    if (visited.count(vid)) return;     
	visited.insert(vid);

    if (impl->grad) {
        impl->grad->clear_graph(visited);
        impl->grad.reset();
    }

    if (impl->creator) {
        std::shared_ptr<Function> f = impl->creator;

        for (auto& input_impl : f->get_inputs()) {
			if (!input_impl) continue;
            Variable input(input_impl);
            input.clear_graph(visited);
        }
    }

    impl->creator.reset(); 
}

Variable Variable::gather_rows(const std::vector<size_t> &indices) const {

    const Tensor<>& x = this->data();
    const std::vector<size_t>& shape = x.get_shape();
    size_t ndim = shape.size();

    if (ndim == 0)
        throw std::runtime_error("get_item: scalar Variable cannot be indexed.");

    // 슬라이싱 대상: 첫 번째 차원
    size_t subblock_size = 1;
    for (size_t i = 1; i < ndim; ++i)
        subblock_size *= shape[i];

    std::vector<float> new_data;
    new_data.reserve(indices.size() * subblock_size);

    for (size_t i : indices) {
        if (i >= shape[0])
            throw std::out_of_range("get_item: index out of bounds");

        for (size_t j = 0; j < subblock_size; ++j) {
            // 전체 플랫 인덱스 계산: row offset + local offset
            size_t flat_idx = i * subblock_size + j;
            new_data.push_back(x.raw_data()[flat_idx]);
        }
    }

    // 새 shape: 첫 번째 차원만 변경, 나머지는 유지
    std::vector<size_t> new_shape = shape;
    new_shape[0] = indices.size();

    return Variable(Tensor<>(new_shape, new_data));

}

Variable Variable::gather_rows(const Tensor<size_t>& indices) const {
	return this->gather_rows(indices.raw_data());
}

void Variable::debug_refs() {
    std::cout << "[Variable] name: " << impl->name 
              << ", use_count: " << impl.use_count()
              << ", creator: " << (impl->creator ? impl->creator->name() : "null") 
              << std::endl;
}

void Variable::show() const {
	std::cout << "Variable {\n";

	const auto& data = impl->data;
	auto shape = data.get_shape();
	std::cout << "  data: ";
	if (data.empty()) 
		std::cout << "(no data)\n";
	else {
		if (shape.size() == 1) data.show();
		else {
			std::cout << "\n";
			data.show();
		}
	}
	std::cout << "  name: " << (impl->name.empty() ? "(unnamed)" : impl->name) << std::endl;
	
	std::cout << "  grad: ";
	if (impl->grad) {
		auto gshape = impl->grad->data().get_shape();
		if (gshape.size() == 1) impl->grad->data().show();
		else {
			std::cout << "\n";
			impl->grad->data().show();
		}
	} else
		std::cout << "(no grad)\n";
	
	std::cout << "}\n";
}
