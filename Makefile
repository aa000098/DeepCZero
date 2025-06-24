# compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Iinclude -Wall -Wextra -O2 -fPIC -MMD -MP -fopenmp
LDFLAGS = -L$(BIN_DIR) -ldeepczero -lcurl -lz -fopenmp
  
# directory
SRC_DIR = src
TEST_DIR = test
BUILD_DIR = build
BIN_DIR = bin

# src and objs
SRCS := $(shell find $(SRC_DIR) -name '*.cpp')
TESTS := $(shell find $(TEST_DIR) -name '*.cpp')

OBJS := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))
TEST_OBJS := $(patsubst $(TEST_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(TESTS))

TEST_BINS = $(patsubst $(TEST_DIR)/%.cpp, $(BIN_DIR)/%, $(TESTS))
DEPS := $(OBJS:.o=.d)

LIB_NAME = libdeepczero.so
LIB_TARGET = $(BIN_DIR)/$(LIB_NAME)



# default rules
all: $(LIB_TARGET)

$(LIB_TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) -shared -o $@ $^ $(LDFLAGS)

$(BIN_DIR)/%: $(BUILD_DIR)/%.o $(LIB_TARGET)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# test auto exec
test: $(TEST_BINS)
	@echo "Running all test..."
	@for bin in $(TEST_BINS); do \
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
