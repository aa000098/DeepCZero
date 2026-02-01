#include "container/tensor/tensor_all.hpp"
#include "container/container_all.hpp"
#include "function/function.hpp"
#include "graph/graph.hpp"
#include "config/config.hpp"

#include <unordered_set>
#include <string>
#include <iostream>

void Variable::backward(bool retain_grad, bool create_graph, bool debug) {
	if (!impl->grad)
		impl->grad = std::make_unique<Variable>(Tensor<>(impl->data.get_shape(), 1));
	impl->grad->set_name("gy");
	auto creator = impl->creator.get();
	if (!creator) return;
	Graph graph(creator);
	std::vector<Function*> topo_order = graph.get_topo_order();

	if (debug) {
		std::cout << "[DEBUG] Starting backward pass with " << topo_order.size() << " functions" << std::endl;
	}

	for (size_t func_idx = 0; func_idx < topo_order.size(); ++func_idx) {
		auto& f = topo_order[func_idx];

		if (debug) {
			std::cout << "[DEBUG] Processing function " << func_idx << "/" << topo_order.size()
			          << " - " << f->name() << std::endl;
		}

		std::vector<std::shared_ptr<VariableImpl<>>> inputs = f->get_inputs();
		std::shared_ptr<VariableImpl<>> output = f->get_output();

		if (!output) {
			if (debug) std::cout << "[DEBUG] Warning: output is null for " << f->name() << std::endl;
			continue;
		}

		if (!output->grad) {
			if (debug) std::cout << "[DEBUG] Warning: output->grad is null for " << f->name() << std::endl;
			continue;
		}

		Variable* gy = output->grad.get();

		{
			dcz::UsingConfig is_higher_order_diff("enable_backprop", create_graph);
			if (debug) std::cout << "[DEBUG] Calling backward on " << f->name() << std::endl;
			std::vector<Variable> gxs = f->backward(*gy);
			if (debug) std::cout << "[DEBUG] Backward returned " << gxs.size() << " gradients" << std::endl;

			for (size_t i = 0; i < gxs.size(); ++i) {
				std::shared_ptr<VariableImpl<>> input = inputs[i];
				const Variable& gx = gxs[i];

				if (debug) std::cout << "[DEBUG]   Input " << i << " - gradient accumulation" << std::endl;
				// Skip empty gradients (e.g., bias gradient when no bias exists)
				if (gx.empty()) {
					if (debug) std::cout << "[DEBUG]     Skipping empty gradient" << std::endl;
					continue;
				}

				if (!input->grad) {
					if (debug) std::cout << "[DEBUG]     Creating new gradient" << std::endl;
					if (create_graph) {
						// For higher-order derivatives, keep computation graph
						input->grad = std::make_unique<Variable>(gx);
					} else {
						// CRITICAL FIX: Use clone() for deep copy (now properly handles TensorView)
						input->grad = std::make_unique<Variable>(gx.data().clone());
					}
				} else {
					if (debug) std::cout << "[DEBUG]     Adding to existing gradient" << std::endl;
					Variable new_grad = (*input->grad) + gx;
					if (create_graph) {
						// For higher-order derivatives, preserve the computation graph
						input->grad = std::make_unique<Variable>(new_grad);
					} else {
						// CRITICAL FIX: Use clone() for deep copy
						input->grad = std::make_unique<Variable>(new_grad.data().clone());
					}
				}

				if (debug) std::cout << "[DEBUG]     Gradient accumulation done" << std::endl;
			}
			if (!retain_grad) output->grad.reset();
		}
	}

	if (debug) {
		std::cout << "[DEBUG] Backward completed successfully!" << std::endl;
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

void Variable::unchain_backward() {
	std::unordered_set<std::uintptr_t> visited;
	unchain_backward(visited);
}

void Variable::unchain_backward(std::unordered_set<std::uintptr_t>& visited) {
	if (!impl || !impl->creator) {
		return;
	}

	// Check if already visited to avoid infinite recursion
	std::uintptr_t var_id = id();
	if (visited.find(var_id) != visited.end()) {
		return;
	}
	visited.insert(var_id);

	// Get the creator function
	auto creator = impl->creator.get();

	// Recursively unchain all input variables
	for (auto& input : creator->get_inputs()) {
		Variable input_var(input);
		input_var.unchain_backward(visited);
	}

	// Finally, unchain this variable's creator
	unchain();
}
