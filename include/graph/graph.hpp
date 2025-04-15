#pragma once

#include "container/variable.hpp" 
#include "function/function.hpp"  

#include <unordered_set>
#include <unordered_map>


class Graph {
private:
	std::unordered_map<Function*, std::vector<Function*>> graph;
	std::unordered_map<Function*, int> in_degree;
	std::vector<Function*> topo_order;

public:
	void build_from(Function* output);
	std::vector<Function*> get_topo_order();

private:
	void dfs(Function* f, std::unordered_set<Function*>& visited);

public:
	void print_graph();
};
