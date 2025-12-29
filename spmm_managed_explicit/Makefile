# Compiler and flags
NVCC        = nvcc
NVCC_FLAGS  = -O3 -std=c++14
LIBS        = -lcusparse
NVCC_FLAGS += -arch=sm_80
NVCC_FLAGS += -lineinfo

TARGETS = spmm_managed_A spmm_explicit_A spmm_managed_C spmm_managed_A_mem_advise spmm_managed_C_mem_advise

all: $(TARGETS)

spmm_managed_A_mem_advise: spmm_managed_A_mem_advise.cu
	$(NVCC) $(NVCC_FLAGS) $< $(LIBS) -o $@

spmm_managed_C_mem_advise: spmm_managed_C_mem_advise.cu
	$(NVCC) $(NVCC_FLAGS) $< $(LIBS) -o $@

spmm_managed_A: spmm_managed_A.cu
	$(NVCC) $(NVCC_FLAGS) $< $(LIBS) -o $@

spmm_managed_C: spmm_managed_C.cu
	$(NVCC) $(NVCC_FLAGS) $< $(LIBS) -o $@

spmm_explicit_A: spmm_explicit_A.cu
	$(NVCC) $(NVCC_FLAGS) $< $(LIBS) -o $@

clean:
	rm -f $(TARGETS)

