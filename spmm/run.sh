#!/usr/bin/env bash
set -u -o pipefail

MATS=(
   #"../../suitesparse_mats/big_matrices/rgg_n_2_20_s0/rgg_n_2_20_s0.mtx"
   "suitesparse/ecology1/ecology1.mtx"
   "suitesparse/rgg_n_2_21_s0/rgg_n_2_21_s0.mtx"
   #"../../suitesparse_mats/big_matrices/Queen_4147/Queen_4147.mtx"
   #"../../suitesparse_mats/small_matrices/Goodwin_095/Goodwin_095.mtx"
   #"../../suitesparse_mats/small_matrices/GL7d14/GL7d14.mtx"
   "suitesparse/c-49/c-49.mtx"
   #"../../suitesparse_mats/small_matrices/c-49/c-49.mtx"
   #"../../suitesparse_mats/small_matrices/ch5-5-b3/ch5-5-b3.mtx"
   #"../../suitesparse_mats/small_matrices/ex19/ex19.mtx"
   "suitesparse/majorbasis/majorbasis.mtx"
   "suitesparse/scircuit/scircuit.mtx"
   "suitesparse/sparsine/sparsine.mtx"
   #"../../suitesparse_mats/small_matrices/torsion1/torsion1.mtx"
   #"../../suitesparse_mats/small_matrices/usps_norm_5NN/usps_norm_5NN.mtx"
)
NRUNS=5

EXE_STREAM="./spmm_dense_tile_overlap"
EXE_CSR="./spmm_csr"

mkdir -p logs

# K tile_rows tile_K precision buff_size
STREAM_CFGS=(
#  "2048 64 double"
#  "2048 256 double"
#  "4096 64 double"
#  "4096 256 double"
#  "8192 64 double"
#  "8192 256 double"
#  "16834 64 double"
#  "16834 256 double"
""
)

CSR_CFGS=(
  "2048"
  "4096"
  "8192"
  "16384"
)

run_and_log() {
  local exe="$1"
  local mat="$2"
  local cfg="$3"
  local label="$4"

  local mat_name
  mat_name=$(basename "$mat" .mtx)
  local cfg_tag="${cfg// /_}"
  local log="logs/${label}_${mat_name}_${cfg_tag}.log"

  echo "------------------------------------------------------"
  echo "Running $label on matrix: $mat_name"
  echo "Config: [$cfg]   NRUNS=$NRUNS"
  echo "Log: $log"
  : > "$log"

  local failed=0

  for ((i=1; i<=NRUNS; ++i)); do
    echo "  Run $i / $NRUNS..."
    $exe "$mat" $cfg | tee -a "$log"
    local status=${PIPESTATUS[0]}
    if (( status != 0 )); then
      echo "  Run $i failed with status $status (likely OOM or similar)."
      echo "  Skipping remaining runs for this config."
      failed=1
      break
    fi
    echo "" >> "$log"
    echo "=========================================================="
  done

  if (( failed != 0 )); then
    echo "Skipping average computation for $label on $mat_name, cfg=[$cfg] due to failure."
    echo ""
    return
  fi

  echo "Computing averages..."

  if [[ "$label" == "spmm_2" ]]; then
    # Expected line format:
    # GPU SpMM=116.168 ms, H2D=65.9103 ms, D2H=229.712 ms, End2End=3696.71 ms

    local mean_spmm mean_h2d mean_d2h mean_e2e

    mean_spmm=$(
      awk 'match($0, /GPU SpMM=([0-9.]+)/, a) { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }' "$log"
    )

    mean_h2d=$(
      awk 'match($0, /H2D=([0-9.]+)/, a) { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }' "$log"
    )

    mean_d2h=$(
      awk 'match($0, /D2H=([0-9.]+)/, a) { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }' "$log"
    )

    mean_e2e=$(
      awk 'match($0, /End2End=([0-9.]+)/, a) { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }' "$log"
    )

    echo ""
    echo "=== Averages for $label on $mat_name, cfg=[$cfg] over $NRUNS runs ==="
    echo "  GPU SpMM = ${mean_spmm:-NA} ms"
    echo "  H2D      = ${mean_h2d:-NA} ms"
    echo "  D2H      = ${mean_d2h:-NA} ms"
    echo "  End2End  = ${mean_e2e:-NA} ms"
    echo ""

  elif [[ "$label" == "overlap" ]]; then
    # spmm_stream_overlap output format:
    # End2End=XXXX ms
    # t_cpu_alloc=... ms
    # t_gpu_alloc=... ms
    # t_h2d_ms=... ms
    # t_d2h_ms=... ms
    # t_spmm_ms=... ms

    local mean_e2e mean_cpu_alloc mean_gpu_alloc mean_h2d mean_d2h mean_spmm

    mean_e2e=$(
      awk 'match($0, /End2End[^0-9]*([0-9.]+)/, a) { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }' "$log"
    )

    mean_cpu_alloc=$(
      awk 'match($0, /t_cpu_alloc[^0-9]*([0-9.]+)/, a) { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }' "$log"
    )

    mean_gpu_alloc=$(
      awk 'match($0, /t_gpu_alloc[^0-9]*([0-9.]+)/, a) { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }' "$log"
    )

    mean_h2d=$(
      awk 'match($0, /t_h2d_ms[^0-9]*([0-9.]+)/, a) { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }' "$log"
    )

    mean_d2h=$(
      awk 'match($0, /t_d2h_ms[^0-9]*([0-9.]+)/, a) { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }' "$log"
    )

    mean_spmm=$(
      awk 'match($0, /t_spmm_ms[^0-9]*([0-9.]+)/, a) { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }' "$log"
    )
    mean_compute=$(
      awk 'match($0, /t_pure_computation[^0-9]*([0-9.]+)/, a)   { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }'       "$log"
    )


    echo ""
    echo "=== Averages for $label on $mat_name, cfg=[$cfg] over $NRUNS runs ==="
    echo "  End2End      = ${mean_e2e:-NA} ms"
    echo "  t_cpu_alloc  = ${mean_cpu_alloc:-NA} ms"
    echo "  t_gpu_alloc  = ${mean_gpu_alloc:-NA} ms"
    echo "  t_h2d_ms     = ${mean_h2d:-NA} ms"
    echo "  t_d2h_ms     = ${mean_d2h:-NA} ms"
    echo "  t_spmm_ms    = ${mean_spmm:-NA} ms"
    echo "  t_pure_computation    = ${mean_compute:-NA} ms"
    echo ""

  else  
    local mean_e2e mean_csr mean_gpu_alloc mean_cpu_alloc mean_h2d mean_spmm mean_d2h mean_dealloc

    mean_e2e=$(
      awk '
        match($0, /(End2End|End-to-end)[^0-9]*([0-9.]+)/, a) { s += a[2]; n++ }
        END { if (n>0) printf "%.3f", s/n; }' "$log"
    )

    mean_csr=$(
      awk 'match($0, /t_csr[^0-9]*([0-9.]+)/, a)      { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }'      "$log"
    )

    mean_gpu_alloc=$(
      awk 'match($0, /t_gpu_alloc[^0-9]*([0-9.]+)/, a) { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }'       "$log"
    )

    mean_cpu_alloc=$(
      awk 'match($0, /t_cpu_alloc[^0-9]*([0-9.]+)/, a) { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }'       "$log"
    )

    mean_h2d=$(
      awk 'match($0, /t_h2d_ms[^0-9]*([0-9.]+)/, a)    { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }'       "$log"
    )

    mean_spmm=$(
      awk 'match($0, /t_spmm_ms[^0-9]*([0-9.]+)/, a)   { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }'       "$log"
    )

    mean_compute=$(
      awk 'match($0, /t_pure_computation[^0-9]*([0-9.]+)/, a)   { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }'       "$log"
    )

    mean_d2h=$(
      awk 'match($0, /t_d2h_ms[^0-9]*([0-9.]+)/, a)    { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }'       "$log"
    )

    mean_dealloc=$(
      awk 'match($0, /t_dealloc[^0-9]*([0-9.]+)/, a)   { s += a[1]; n++ }
           END { if (n>0) printf "%.3f", s/n; }'       "$log"
    )

    echo ""
    echo "=== Averages for $label on $mat_name, cfg=[$cfg] over $NRUNS runs ==="
    echo "  End2End     = ${mean_e2e:-NA} ms"
    echo "  t_csr       = ${mean_csr:-NA} ms"
    echo "  t_gpu_alloc = ${mean_gpu_alloc:-NA} ms"
    echo "  t_cpu_alloc = ${mean_cpu_alloc:-NA} ms"
    echo "  t_h2d_ms    = ${mean_h2d:-NA} ms"
    echo "  t_spmm_ms   = ${mean_spmm:-NA} ms"
    echo "  t_pure_computation   = ${mean_compute:-NA} ms"
    echo "  t_d2h_ms    = ${mean_d2h:-NA} ms"
    echo "  t_dealloc   = ${mean_dealloc:-NA} ms"
    echo ""
  fi
}

# ========= Run everything for each matrix ========= #
#for mat in "${MATS[@]}"; do
#  mat_name=$(basename "$mat" .mtx)
#  results_file="results_best_tile_K/results_${mat_name}.txt"

#  {
#    echo "============================================================="
#    echo "               MATRIX: ${mat_name}"
#    echo "============================================================="

    #for cfg in "${STREAM_CFGS[@]}"; do
    #  run_and_log "$EXE_STREAM" "$mat" "$cfg" "overlap"
    #done

    #for cfg in "${CSR_CFGS[@]}"; do
    #  run_and_log "$EXE_CSR" "$mat" "$cfg" "csr"
    #done
#  } > "$results_file"
#done

for mat in "${MATS[@]}"; do
  mat_name=$(basename "$mat" .mtx)
  nsys profile -o ./results_best_tile_K/nsys_reports/nsys_report_${mat_name} -f true -t cuda,cublas --cuda-memory-usage=true --stats=true -w true ./spmm_dense_tile_overlap $mat > ./results_best_tile_K/results_${mat_name}.txt
done
