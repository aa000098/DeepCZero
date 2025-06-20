#pragma once

template <typename T>
class Iterator {
private:
	T* container;
	size_t index;

public:
	Iterator(T* container, size_t index)
		: container(container), index(index) {};

	Iterator& operator++() {
		index++;
		return *this;
	}

	auto operator*() {
		return (*container)[index];
	}

	bool operator!=(const Iterator& other) const {
		return index != other.index;
	}
};
