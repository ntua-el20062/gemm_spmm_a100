N=15000; batch=5; steps=15
echo "----------------Initial approach-----------------"
nsys profile -o ./nsys_reports/nsys_report_ia_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_initial_approach $N $batch $steps
echo "-----------------double Buffering-----------------"
nsys profile -o ./nsys_reports/nsys_report_db_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_double_buffering $N $batch $steps
echo "-----------------full overlap-----------------"
nsys profile -o ./nsys_reports/nsys_report_fo_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_full_overlap $N $batch $steps
echo -e "=================================================\n\n\n"

N=10000; batch=10; steps=100
echo "----------------Initial approach-----------------"
nsys profile -o ./nsys_reports/nsys_report_ia_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_initial_approach $N $batch $steps
echo "-----------------double Buffering-----------------"
nsys profile -o ./nsys_reports/nsys_report_db_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_double_buffering $N $batch $steps
echo "-----------------full overlap-----------------"
nsys profile -o ./nsys_reports/nsys_report_fo_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_full_overlap $N $batch $steps
echo -e "=================================================\n\n\n"


N=10000; batch=10; steps=200
echo "----------------Initial approach-----------------"
nsys profile -o ./nsys_reports/nsys_report_ia_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_initial_approach $N $batch $steps
echo "-----------------double Buffering-----------------"
nsys profile -o ./nsys_reports/nsys_report_db_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_double_buffering $N $batch $steps
echo "-----------------full overlap-----------------"
nsys profile -o ./nsys_reports/nsys_report_fo_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_full_overlap $N $batch $steps
echo -e "=================================================\n\n\n"


N=10000; batch=20; steps=120
echo "----------------Initial approach-----------------"
nsys profile -o ./nsys_reports/nsys_report_ia_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_initial_approach $N $batch $steps
echo "-----------------double Buffering-----------------"
nsys profile -o ./nsys_reports/nsys_report_db_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_double_buffering $N $batch $steps
echo "-----------------full overlap-----------------"
nsys profile -o ./nsys_reports/nsys_report_fo_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_full_overlap $N $batch $steps
echo -e "=================================================\n\n\n"


N=10000; batch=20; steps=200
echo "----------------Initial approach-----------------"
nsys profile -o ./nsys_reports/nsys_report_ia_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_initial_approach $N $batch $steps
echo "-----------------double Buffering-----------------"
nsys profile -o ./nsys_reports/nsys_report_db_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_double_buffering  $N $batch $steps
echo "-----------------full overlap-----------------"
nsys profile -o ./nsys_reports/nsys_report_fo_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_full_overlap $N $batch $steps
echo -e "=================================================\n\n\n"


N=10000; batch=50; steps=600
echo "----------------Initial approach-----------------"
nsys profile -o ./nsys_reports/nsys_report_ia_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_initial_approach $N $batch $steps
echo "-----------------double Buffering-----------------"
nsys profile -o ./nsys_reports/nsys_report_db_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_double_buffering  $N $batch $steps
echo "-----------------full overlap-----------------"
nsys profile -o ./nsys_reports/nsys_report_fo_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_full_overlap $N $batch $steps
echo -e "=================================================\n\n\n"



N=10000; batch=30; steps=150
echo "----------------Initial approach-----------------"
nsys profile -o ./nsys_reports/nsys_report_ia_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_initial_approach $N $batch $steps
echo "-----------------double Buffering-----------------"
nsys profile -o ./nsys_reports/nsys_report_db_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_double_buffering  $N $batch $steps
echo "-----------------full overlap-----------------"
nsys profile -o ./nsys_reports/nsys_report_fo_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_full_overlap $N $batch $steps
echo -e "=================================================\n\n\n"


N=10000; batch=30; steps=300
echo "----------------Initial approach-----------------"
nsys profile -o ./nsys_reports/nsys_report_ia_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_initial_approach $N $batch $steps
echo "-----------------double Buffering-----------------"
nsys profile -o ./nsys_reports/nsys_report_db_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_double_buffering  $N $batch $steps
echo "-----------------full overlap-----------------"
nsys profile -o ./nsys_reports/nsys_report_fo_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_full_overlap $N $batch $steps
echo -e "=================================================\n\n\n"



N=10000; batch=60; steps=360
echo "----------------Initial approach-----------------"
nsys profile -o ./nsys_reports/nsys_report_ia_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_initial_approach $N $batch $steps
echo "-----------------double Buffering-----------------"
nsys profile -o ./nsys_reports/nsys_report_db_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_double_buffering  $N $batch $steps
echo "-----------------full overlap-----------------"
nsys profile -o ./nsys_reports/nsys_report_fo_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_full_overlap $N $batch $steps
echo -e "=================================================\n\n\n"


N=20000; batch=10; steps=100
echo "----------------Initial approach-----------------"
nsys profile -o ./nsys_reports/nsys_report_ia_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_initial_approach $N $batch $steps
echo "-----------------double Buffering-----------------"
nsys profile -o ./nsys_reports/nsys_report_db_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_double_buffering  $N $batch $steps
echo "-----------------full overlap-----------------"
nsys profile -o ./nsys_reports/nsys_report_fo_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_full_overlap $N $batch $steps
echo -e "=================================================\n\n\n"


N=20000; batch=10; steps=300
echo "----------------Initial approach-----------------"
nsys profile -o ./nsys_reports/nsys_report_ia_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_initial_approach $N $batch $steps
echo "-----------------double Buffering-----------------"
nsys profile -o ./nsys_reports/nsys_report_db_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_double_buffering  $N $batch $steps
echo "-----------------full overlap-----------------"
nsys profile -o ./nsys_reports/nsys_report_fo_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_full_overlap $N $batch $steps
echo -e "=================================================\n\n\n"


N=20000; batch=15; steps=150
echo "----------------Initial approach-----------------"
nsys profile -o ./nsys_reports/nsys_report_ia_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_initial_approach $N $batch $steps
echo "-----------------double Buffering-----------------"
nsys profile -o ./nsys_reports/nsys_report_db_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_double_buffering  $N $batch $steps
echo "-----------------full overlap-----------------"
nsys profile -o ./nsys_reports/nsys_report_fo_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_full_overlap $N $batch $steps
echo -e "=================================================\n\n\n"


N=20000; batch=20; steps=200
echo "----------------Initial approach-----------------"
nsys profile -o ./nsys_reports/nsys_report_ia_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_initial_approach $N $batch $steps
echo "-----------------double Buffering-----------------"
nsys profile -o ./nsys_reports/nsys_report_db_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_double_buffering  $N $batch $steps
echo "-----------------full overlap-----------------"
nsys profile -o ./nsys_reports/nsys_report_fo_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_full_overlap $N $batch $steps
echo -e "=================================================\n\n\n"

N=25000; batch=5; steps=25
echo "----------------Initial approach-----------------"
nsys profile -o ./nsys_reports/nsys_report_ia_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_initial_approach $N $batch $steps
echo "-----------------double Buffering-----------------"
nsys profile -o ./nsys_reports/nsys_report_db_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_double_buffering  $N $batch $steps
echo "-----------------full overlap-----------------"
nsys profile -o ./nsys_reports/nsys_report_fo_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_full_overlap $N $batch $steps
echo -e "=================================================\n\n\n"


N=25000; batch=20; steps=200
echo "----------------Initial approach-----------------"
nsys profile -o ./nsys_reports/nsys_report_ia_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_initial_approach $N $batch $steps
echo "-----------------double Buffering-----------------"
nsys profile -o ./nsys_reports/nsys_report_db_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_double_buffering  $N $batch $steps
echo "-----------------full overlap-----------------"
nsys profile -o ./nsys_reports/nsys_report_fo_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_full_overlap $N $batch $steps
echo -e "=================================================\n\n\n"

N=30000; batch=15; steps=150
echo "----------------Initial approach-----------------"
nsys profile -o ./nsys_reports/nsys_report_ia_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_initial_approach $N $batch $steps
echo "-----------------double Buffering-----------------"
nsys profile -o ./nsys_reports/nsys_report_db_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_double_buffering  $N $batch $steps
echo "-----------------full overlap-----------------"
nsys profile -o ./nsys_reports/nsys_report_fo_${N}_${batch}_${steps} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./gemm_pipeline_full_overlap $N $batch $steps
echo -e "=================================================\n\n\n"

