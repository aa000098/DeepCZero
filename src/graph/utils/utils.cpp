#include "graph/utils/utils.hpp"


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

	std::uintptr_t id = reinterpret_cast<std::uintptr_t>(v.id());

	oss << "  \"" << id << "\" [label=\"" << label << "\", color=orange, style=filled]";
	return oss.str();
}


