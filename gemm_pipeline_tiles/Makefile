NVCC       := nvcc
NVCC_FLAGS := -O3 -std=c++14
NVCC_FLAGS += -gencode arch=compute_80,code=sm_80

TARGETS    := gemm_pipeline_tiling

all: $(TARGETS)

gemm_pipeline_tiling: gemm_pipeline_tiling.cu utils.h
	$(NVCC) $(NVCC_FLAGS) $< -lcublas -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean

