# Limit parallel jobs to avoid OOM
MAKEFLAGS += -j2

# compiler
CXX = g++

# Optional: Enable Intel MKL for optimized BLAS operations
# To disable: make USE_MKL=0
USE_MKL ?= 1

# directory
SRC_DIR = src
TEST_DIR = test
BUILD_DIR = build
BIN_DIR = bin

THIRD_PARTY_DIR = third_party
CNPY_DIR = $(THIRD_PARTY_DIR)/cnpy
STB_DIR = $(THIRD_PARTY_DIR)/stb

# base flags
CXXFLAGS = -std=c++17 -Iinclude -Wall -Wextra -O2 -fPIC -MMD -MP -fopenmp -I$(CNPY_DIR) -I$(STB_DIR)
LDFLAGS = -L$(BIN_DIR) -ldeepczero -lcurl -lz -fopenmp -lzip

# Include Intel MKL configuration if enabled
ifeq ($(USE_MKL), 1)
    -include Makefile.mkl
endif

# src and objs
SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
TESTS := $(shell find $(TEST_DIR) -name '*.cpp' -not -path '$(TEST_DIR)/benchmark/*')
BENCHMARKS := $(shell find $(TEST_DIR)/benchmark -name '*.cpp' 2>/dev/null)

OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))
TEST_OBJS := $(patsubst $(TEST_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(TESTS))

TEST_BINS = $(patsubst $(TEST_DIR)/%.cpp, $(BIN_DIR)/%, $(TESTS))
BENCH_BINS = $(patsubst $(TEST_DIR)/%.cpp, $(BIN_DIR)/%, $(BENCHMARKS))
DEPS := $(OBJS:.o=.d)

LIB_NAME = libdeepczero.so
LIB_TARGET = $(BIN_DIR)/$(LIB_NAME)

# third party
CNPY_SRC = $(CNPY_DIR)/cnpy.cpp
CNPY_OBJ = $(BUILD_DIR)/cnpy.o
OBJS += $(CNPY_OBJ)


# default rules
all: $(LIB_TARGET)

$(LIB_TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) -shared -o $@ $^ $(LDFLAGS)

$(BIN_DIR)/%: $(BUILD_DIR)/%.o $(LIB_TARGET)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CNPY_OBJ): $(CNPY_SRC)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@


# test auto exec
test: $(TEST_BINS)
	@echo "Running all tests..."
	@for bin in $(TEST_BINS); do \
		echo "Running $$bin"; \
		./$$bin || exit 1; \
	done

# benchmark
bench: $(BENCH_BINS)
	@echo "Running all benchmarks..."
	@for bin in $(BENCH_BINS); do \
		echo "Running $$bin"; \
		./$$bin || exit 1; \
	done

# memory leak check
memory: $(TEST_BINS)
	@echo "Running all tests with valgrind..."
	@for bin in $(TEST_BINS); do \
		echo "Running valgrind $$bin"; \
		valgrind --leak-check=full --error-exitcode=1 ./$$bin || exit 1; \
	done

clean:
	rm -rf $(BIN_DIR) $(BUILD_DIR)

-include $(DEPS)
