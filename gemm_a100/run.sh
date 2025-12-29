#./cublas_gemm_example_managed 10000 10000 10000 25 > results_%_managed.txt
#./cublas_gemm_example_managed 20000 20000 20000 25 >> results_%_managed.txt
#python3 plot_spikes.py
#./cublas_gemm_example_malloc 10000 10000 10000 25 > results_%_malloc.txt
#./cublas_gemm_example_malloc 20000 20000 20000 25 >> results_%_malloc.txt
#python3 plot_spikes.py

#nsys profile --trace=cuda,osrt --cuda-memory-usage=true -o um_profile_malloc ./cublas_gemm_example_malloc 10000 10000 10000 25 > results_%_malloc.txt
#nsys profile --trace=cuda,osrt --cuda-memory-usage=true -o um_profile_managed ./cublas_gemm_example_managed 10000 10000 10000 25 > results_%_managed.txt
#python3 plot_mem_malloc.py
#python3 plot_mem_managed.py
#python3 plot_spikes_managed.py
#python3 plot_spikes_malloc.py

#nsys profile --trace=cuda,osrt --cuda-memory-usage=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true -o um_profile_managed_A ./cublas_gemm_example_managed_A 10000 10000 10000 25 > results_%_managed_A.txt;
python3 plot_mem_managed_A.py;
#nsys profile --trace=cuda,osrt --cuda-memory-usage=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true -o um_profile_managed_C ./cublas_gemm_example_managed_C 10000 10000 10000 25 > results_%_managed_C.txt;
python3 plot_mem_managed_C.py;
#nsys profile --trace=cuda,osrt --cuda-memory-usage=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true -o um_profile_malloc_A ./cublas_gemm_example_malloc_A 10000 10000 10000 25 > results_%_malloc_A.txt;
python3 plot_mem_malloc_A.py;
#nsys profile --trace=cuda,osrt --cuda-memory-usage=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true -o um_profile_malloc_C ./cublas_gemm_example_malloc_C 10000 10000 10000 25 > results_%_malloc_C.txt;
python3 plot_mem_malloc_C.py;
