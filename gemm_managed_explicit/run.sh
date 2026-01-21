#./cublas_gemm_example_explicit_A 10000 10000 10000 25 > results_%_explicit_A_10k.txt;
#python3 plot_mem_explicit_A.py;

#./cublas_gemm_example_explicit_C 10000 10000 10000 25 > results_%_explicit_C_10k.txt;
#python3 plot_mem_explicit_C.py;

#nsys profile --trace=cuda,osrt --cuda-memory-usage=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true -o um_profile_managed_A ./cublas_gemm_example_managed_A 10000 10000 10000 25 #> results_%_managed_A_10k.txt;
#python3 plot_mem_managed_A.py;

#nsys profile --trace=cuda,osrt --cuda-memory-usage=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true -o um_profile_managed_C ./cublas_gemm_example_managed_C 10000 10000 10000 25 #> results_%_managed_C_10k.txt;
#python3 plot_mem_managed_C.py;

#nsys profile --trace=cuda,osrt --cuda-memory-usage=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true -o um_profile_managed_C_mem_advise ./cublas_gemm_example_managed_C_mem_advise 10000 10000 10000 25 #> results_%_managed_C_mem_advise_10k.txt;
#python3 plot_mem_managed_C_mem_advise.py;

#nsys profile --trace=cuda,osrt --cuda-memory-usage=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true -o um_profile_managed_A_mem_advise ./cublas_gemm_example_managed_A_mem_advise 10000 10000 10000 25 #> results_%_managed_A_mem_advise_10k.txt;
#python3 plot_mem_managed_A_mem_advise.py;

#./cublas_gemm_example_managed_A_mem_advise_readmostly 10000 10000 10000 25 > results_%_managed_A_mem_advise_10k_read_mostly.txt;
python3 plot_mem_managed_A_mem_advise_read_mostly.py;

#./cublas_gemm_example_managed_C_mem_advise_readmostly 10000 10000 10000 25 > results_%_managed_C_mem_advise_10k_read_mostly.txt;
python3 plot_mem_managed_C_mem_advise_read_mostly.py;

