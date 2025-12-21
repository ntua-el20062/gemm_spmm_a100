# Compiler and flags
NVCC        = nvcc
NVCC_FLAGS  = -O3 -std=c++14
NVCC_FLAGS += -gencode arch=compute_80,code=sm_80
LIBS        = -lcusparse

# Executable targets
TARGETS = spmm_csr spmm_dense_tile_overlap

all: $(TARGETS)

spmm_csr: spmm_csr.cu
	$(NVCC) $(NVCC_FLAGS) $< $(LIBS) -o $@

spmm_dense_tile_overlap: spmm_dense_tile_overlap.cu
	$(NVCC) $(NVCC_FLAGS) $< $(LIBS) -o $@

clean:
	rm -f $(TARGETS)

