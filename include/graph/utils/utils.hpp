#pragma once

#include "container/variable.hpp"
#include "function/function.hpp"

#include <string>
#include <sstream>

std::string _dot_var(Variable v, bool verbose = false);
std::string _dot_func(Function* f);
std::string get_dot_graph(Variable output, bool verbose = true);
void plot_dot_graph(Variable output, bool verbose = true, std::string to_file="test_graph");
