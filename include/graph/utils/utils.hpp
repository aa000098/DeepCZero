#pragma once

#include "container/variable.hpp"
#include "function/function.hpp"

#include <string>
#include <sstream>
#include <set>

using function::Function;

std::string _dot_var(Variable v, bool verbose = false);
std::string _dot_func(Function* f);
std::string get_dot_graph(Variable output, bool verbose = true);
void plot_dot_graph(Variable output, bool verbose = true, std::string to_file="test_graph");
void trace_variable_refs(const Variable v, std::unordered_set<std::uintptr_t>* visited = nullptr);
std::string demangle(const char* name);
std::string remove_namespace(const std::string& name);
