#include "graph/graph.hpp"

#include <queue>
#include <algorithm>
#include <typeinfo>
#include <iostream>

void Graph::build_from(Function* output) {
	graph.clear();
	in_degree.clear();
	topo_order.clear();

	std::unordered_set<Function*> visited;
	dfs(output, visited);
}

void Graph::dfs(Function* f, std::unordered_set<Function*>& visited) {
	if (visited.count(f)) return;
	visited.insert(f);

	for (auto& input_var : f->get_inputs()) {
		Function* prev_func = input_var->creator.get();
		if (prev_func) {
			graph[prev_func].push_back(f);
			in_degree[f]++;
			dfs(prev_func, visited);
		}
	}
}


std::vector<Function*> Graph::get_topo_order() {
	std::vector<Function*> order;
	std::queue<Function*> queue;

	for (const auto& [node, _] : graph) {
		if (in_degree[node] == 0) {
			queue.push(node);
		}
	}

	while(!queue.empty()) {
		Function* f = queue.front();
		queue.pop();
		order.push_back(f);

		for (Function* next: graph[f]) {
			in_degree[next]--;
			if (in_degree[next] == 0) {
				queue.push(next);
			}
		}
	}
	
	std::reverse(order.begin(), order.end());
	return order;
}


void Graph::print_graph() {
    std::cout << "=== Computation Graph ===" << std::endl;

    for (const auto& pair : graph) {
        Function* from = pair.first;
        const std::vector<Function*>& to_list = pair.second;

		int deg = in_degree.count(from) ? in_degree[from] : 0;

        std::cout << "[ " << from << " | " << typeid(*from).name() << " | in-degree: " << deg << " ] -> { ";
        for (Function* to : to_list) {
            std::cout << to << " | " << typeid(*to).name() << " , ";
        }
        std::cout << "}" << std::endl;
    }

    std::cout << "=========================" << std::endl;
}
