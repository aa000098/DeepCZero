#include "graph/utils/utils.hpp"
#include "graph/graph.hpp"

#include <fstream>
#include <cstdlib>
#include <iostream>
#include <filesystem>

std::string shape_to_string(const std::vector<size_t>& shape) {
	std::ostringstream oss;
	oss << "(";
	for (size_t i = 0; i < shape.size(); i++) {
		oss << shape[i];
		if (i != shape.size() - 1)
			oss << ", ";
	}
	if (shape.size() == 1)
		oss << ",";
	oss << ")";
	return oss.str();
}

std::string _dot_var(Variable v, bool verbose) {
	std::stringstream oss;

	std::string label = v.name();
		
	if (verbose && !v.empty()) {
		if (!label.empty())
			label += ": ";

		label += shape_to_string(v.shape()) + " " + v.dtype_string();
	}

	std::uintptr_t id = v.id();

	oss << id << " [label=\"" << label << "\", color=orange, style=filled]\n";
	return oss.str();
}

std::string _dot_func(function::Function* f) {
	std::stringstream oss;
	
	std::string label = f->name();

	std::uintptr_t id = f->id();
	oss << id << " [label=\"" << label << "\", color=lightblue, style=filled, shape=box]\n";
	for (auto x : f->get_inputs()) 
		oss << x->id() << " -> " << f->id() << "\n";
//	for (auto y : f->get_output())
	auto y = f->get_output();
	oss << f->id() << " -> " << y->id() << "\n";

	return oss.str();
}

std::string get_dot_graph(Variable output, bool verbose) {
	std::stringstream oss;

	function::Function* out_func = output.get_creator().get();
	Graph graph(out_func);
	std::vector<function::Function*> topo_order = graph.get_topo_order();

	oss << _dot_var(output, verbose);
	for (auto& f : topo_order) {
		oss << _dot_func(f);
		std::vector<std::shared_ptr<VariableImpl<>>> inputs = f->get_inputs(); 
		std::shared_ptr<VariableImpl<>> output = f->get_output();

		for (auto& x : inputs) 
			oss << _dot_var(Variable(x), verbose);
	}

	return oss.str();
}


void plot_dot_graph(
		Variable output, 
		bool verbose, 
		std::string to_file) {
	const std::string home = std::getenv("HOME");
	const std::string dir = home + "/.deepczero";
	const std::string dot_path = dir + "/" + to_file + ".dot";
	const std::string img_path = dir + "/" + to_file + ".png";

	std::filesystem::create_directories(dir);

	std::ofstream ofs(dot_path);
	if (!ofs.is_open()) {
		std::cerr << "Failed to open file: " << dot_path << std::endl;
		return;
	}

	ofs << "digraph computational_graph {\n";
//	ofs << "  rankdir=LR;\n";
	ofs << get_dot_graph(output, verbose);
	ofs << "}\n";
	ofs.close();
	
	std::string cmd = "dot -Tpng " + dot_path + " -o " + img_path;
	int ret = std::system(cmd.c_str());
	if(ret != 0) {
		std::cerr << "Graphviz exec failed: " << cmd << std::endl;
		std::cerr << "You can install Graphviz using one of the following methods:\n";
		std::cerr << "  - Ubuntu/Debian: sudo apt install graphviz\n";
	}
	else 
		std::cout << "Graphviz generated success: " << img_path << std::endl;
}

void trace_variable_refs(const Variable v, std::unordered_set<std::uintptr_t>* visited) {
    std::unordered_set<std::uintptr_t> default_set;

    if (!visited) visited = &default_set;

    auto vid = v.get_impl()->id();
    if (visited->count(vid)) return;
    visited->insert(vid);

    std::cout << "Variable: " << (v.get_impl()->name.empty() ? "(unnamed)" : v.get_impl()->name)
              << ", id: " << vid
              << ", use_count: " << v.get_impl().use_count() << std::endl;

    auto f = v.get_creator();
    if (f) {
        std::cout << "  Creator Function: " << f->name()
                  << ", id: " << f->id()
                  << ", shared_ptr use_count: " << f.use_count() << std::endl;

        auto inputs = f->get_inputs();
        for (size_t i = 0; i < inputs.size(); ++i) {
            auto inp = inputs[i];
            std::cout << "    Input[" << i << "] id: " << inp->id()
                      << ", use_count: " << inp.use_count() << std::endl;

            Variable vi(inp);
            trace_variable_refs(vi, visited);  // 재귀 호출
        }

        auto out = f->get_output();
        if (out) {
            std::cout << "    Output id: " << out->id()
                      << ", use_count: " << out.use_count() << std::endl;
        }
    }
}
