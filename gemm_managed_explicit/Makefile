# Compiler and Flags
NVCC := nvcc
NVCC_FLAGS := -O3 -std=c++11 -Wno-deprecated-gpu-targets
LIBS := -lcublas -lcudart

# Find all .cu files in the directory
SRCS := $(wildcard *.cu)

# Generate executable names by removing the .cu extension
BINS := $(SRCS:.cu=)

# Default target builds everything
all: $(BINS)

# Rule to compile each .cu file into its own executable
%: %.cu
	@echo "Building $@..."
	$(NVCC) $(NVCC_FLAGS) $< -o $@ $(LIBS)

# Clean up all executables and CSV logs
clean:
	rm -f $(BINS) *.csv

.PHONY: all clean
