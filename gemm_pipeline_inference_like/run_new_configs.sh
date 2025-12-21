#!/bin/bash

for N in 10000 20000 25000 30000; do
  for batch in 5 10 20; do
    for steps in 20 40 60 80 100 150; do
       echo -e "------------INITIAL APPROACH----------------------\n"
       nsys profile -o ./nsys_reports/nsys_report_ia_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_initial_approach $N $batch $steps 0
       echo -e "--------------------------------------------------\n\n"

       echo -e "------------DOUBLE BUFFERING----------------------\n"
       nsys profile -o ./nsys_reports/nsys_report_db_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_double_buffering $N $batch $steps 0
       echo -e "--------------------------------------------------\n\n"

       echo -e "------------FULL OVERLAP----------------------\n"
       nsys profile -o ./nsys_reports/nsys_report_fo_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_full_overlap $N $batch $steps 0
       echo -e "--------------------------------------------------\n"
       echo -e "==================================================\n\n\n"
    done
  done
done

