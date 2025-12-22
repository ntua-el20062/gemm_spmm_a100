#!/bin/bash
#echo -e "------------FULL OVERLAP TILED----------------------\n"
#for N in 10000 20000 25000 30000 50000 80000 100000 105000 106000 107000 110000; do
#       ./gemm_pipeline_tiling $N
#       echo -e "---------------------------------------------------\n\n\n"
#done


for N in 10000 20000 25000 30000 50000 80000 100000 105000 106000 107000 110000; do
    nsys profile -o ./nsys_reports/nsys_report_gemm_tiled_${N} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_tiling $N
    echo -e "--------------------------------------------------------\n\n\n"
done
