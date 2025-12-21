NVCC       := nvcc
NVCC_FLAGS := -O3 -std=c++14
NVCC_FLAGS += -gencode arch=compute_80,code=sm_80

TARGETS    := gemm_pipeline_full_overlap gemm_pipeline_double_buffering gemm_pipeline_initial_approach

all: $(TARGETS)

gemm_pipeline_initial_approach: gemm_pipeline_initial_approach.cu utils.h
	$(NVCC) $(NVCC_FLAGS) $< -lcublas -o $@

gemm_pipeline_double_buffering: gemm_pipeline_double_buffering.cu utils.h
	$(NVCC) $(NVCC_FLAGS) $< -lcublas -o $@

gemm_pipeline_full_overlap: gemm_pipeline_full_overlap.cu utils.h
	$(NVCC) $(NVCC_FLAGS) $< -lcublas -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean

