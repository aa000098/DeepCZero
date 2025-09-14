#pragma once

#include <cassert>
#include <cstddef>
#include <utility>

static std::pair<size_t, size_t> to_pair(std::initializer_list<size_t> il) {
	assert(il.size()==2);
	auto it=il.begin();
	return std::pair<size_t, size_t>{*it, *(it+1)};
}
