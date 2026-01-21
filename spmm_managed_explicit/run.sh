#nsys profile --trace=cuda,osrt --cuda-memory-usage=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true -o um_profile_managed_A_mem_advise ./spmm_managed_A_mem_advise ../../suitesparse_mats/ldoor/ldoor.mtx 128 > results_%_managed_A_mem_advise_ldoor.txt;
#python3 plot_mem_managed_A_mem_advise.py;

#nsys profile --trace=cuda,osrt --cuda-memory-usage=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true -o um_profile_managed_C_mem_advise ./spmm_managed_C_mem_advise ../../suitesparse_mats/ldoor/ldoor.mtx 128 > results_%_managed_C_mem_advise_ldoor.txt;
#python3 plot_mem_managed_C_mem_advise.py;

nsys profile --trace=cuda,osrt --cuda-memory-usage=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true -o um_profile_managed_A ./spmm_managed_A suitesparse/ldoor/ldoor.mtx 128 

nsys profile --trace=cuda,osrt --cuda-memory-usage=true --cuda-um-gpu-page-faults=true --cuda-um-cpu-page-faults=true -o um_profile_managed_C ./spmm_managed_C suitesparse/ldoor/ldoor.mtx 128 
#python3 plot_mem_managed_C.py;

#./spmm_managed_A_mem_advise ../spmm/suitesparse/ldoor/ldoor.mtx 128 > results_%_managed_A_mem_advise_ldoor.txt;
#python3 plot_mem_managed_A_mem_advise.py;

#./spmm_managed_C_mem_advise ../spmm/suitesparse/ldoor/ldoor.mtx 128 > results_%_managed_C_mem_advise_ldoor.txt;
#python3 plot_mem_managed_C_mem_advise.py;

#./spmm_managed_A ../spmm/suitesparse/ldoor/ldoor.mtx 128 > results_%_managed_A_ldoor.txt;
#python3 plot_mem_managed_A.py;

#./spmm_managed_C ../spmm/suitesparse/ldoor/ldoor.mtx 128 > results_%_managed_C_ldoor.txt;
#python3 plot_mem_managed_C.py;

#./spmm_explicit_A ../spmm/suitesparse/ldoor/ldoor.mtx 128 > results_%_explicit_A_ldoor.txt;
#python3 plot_mem_explicit_A.py;
