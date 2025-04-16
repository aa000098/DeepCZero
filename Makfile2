# compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Iinclude -Wall -Wextra -O2

# directory
SRC_DIR = src
TEST_DIR = test
BUILD_DIR = build
BIN_DIR = bin

# src and objs
SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
TESTS := $(shell find $(TEST_DIR) -name '*.cpp')

OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))
TEST_BINS = $(patsubst $(TEST_DIR)/%.cpp, $(BIN_DIR)/%, $(TESTS))

# default rules
all: $(TEST_BINS)

$(BIN_DIR)/%: $(TEST_DIR)/%.cpp $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

# test auto exec
test: $(TEST_BINS)
	@echo "Running all test..."
	@for bin in $(TEST_BINS); do \
		echo "Running $$bin"; \
		./$$bin || exit 1; \
	done

clean:
	rm -rf $(BIN_DIR) $(BUILD_DIR)
