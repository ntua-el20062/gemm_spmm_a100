#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cctype>
#include <sys/time.h>
#include <cstring>
#include <type_traits>

#include <cuda_runtime.h>
#include <cusparse_v2.h>

// ---------- error macros ----------
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

#define CUSPARSE_CHECK(call)                                                  \
    do {                                                                      \
        cusparseStatus_t st = (call);                                         \
        if (st != CUSPARSE_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuSPARSE error %s:%d: %d\n",                     \
                    __FILE__, __LINE__, (int)st);                             \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// simple wall-clock timer
double wtime()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// ---------------- templated CSR struct + loader ----------------
template<typename T>
struct CSR {
    int m = 0;
    int n = 0;
    int nnz = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<T>   vals;
};

template<typename T>
CSR<T> read_matrix_market_to_csr(const std::string &filename)
{
    std::ifstream fin(filename);
    if (!fin) {
        std::cerr << "Cannot open file " << filename << "\n";
        std::exit(EXIT_FAILURE);
    }

    std::string line;
    if (!std::getline(fin, line)) {
        std::cerr << "Empty Matrix Market file\n";
        std::exit(EXIT_FAILURE);
    }
    if (line.rfind("%%MatrixMarket", 0) != 0) {
        std::cerr << "File does not start with MatrixMarket header\n";
        std::exit(EXIT_FAILURE);
    }

    bool symmetric = false;
    {
        std::string lower = line;
        for (char &c : lower) c = std::tolower(c);
        if (lower.find("symmetric") != std::string::npos)
            symmetric = true;
    }

    // skip comments
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        if (line[0] == '%') continue;
        break;
    }

    CSR<T> csr;
    int m, n, nnz_header;
    {
        std::stringstream ss(line);
        ss >> m >> n >> nnz_header;
        csr.m = m;
        csr.n = n;
    }

    // read triplets as double from file, cast to T
    std::vector<int> I, J;
    std::vector<T>   V;
    I.reserve(nnz_header * (symmetric ? 2 : 1));
    J.reserve(nnz_header * (symmetric ? 2 : 1));
    V.reserve(nnz_header * (symmetric ? 2 : 1));

    int i, j;
    double val_d;
    while (fin >> i >> j >> val_d) {
        i--; j--;
        I.push_back(i);
        J.push_back(j);
        V.push_back(static_cast<T>(val_d));
        if (symmetric && i != j) {
            I.push_back(j);
            J.push_back(i);
            V.push_back(static_cast<T>(val_d));
        }
    }

    int nnz = (int)I.size();
    csr.nnz = nnz;
    csr.row_ptr.assign(m + 1, 0);
    csr.col_idx.resize(nnz);
    csr.vals.resize(nnz);

    for (int e = 0; e < nnz; ++e)
        csr.row_ptr[I[e] + 1]++;
    for (int r = 0; r < m; ++r)
        csr.row_ptr[r + 1] += csr.row_ptr[r];

    std::vector<int> offset = csr.row_ptr;
    for (int e = 0; e < nnz; ++e) {
        int row = I[e];
        int dest = offset[row]++;
        csr.col_idx[dest] = J[e];
        csr.vals[dest]    = V[e];
    }

    /*
    std::cout << "Loaded CSR<" << (std::is_same<T,float>::value ? "float" : "double")
              << ">: m=" << csr.m << " n=" << csr.n
              << " nnz=" << csr.nnz
              << " symmetric=" << (symmetric ? "yes" : "no") << "\n";
  */
	      return csr;
}

// -------- helpers to map T -> cudaDataType --------
inline cudaDataType getCudaType(float)  { return CUDA_R_32F; }
inline cudaDataType getCudaType(double) { return CUDA_R_64F; }

template<typename T>
cudaDataType getCudaType()
{
    return getCudaType(T{});
}

// Simple timing struct for summary
struct SpmmTimings {
    double t_end_ms;      // end-to-end (from start of GPU/tiling phase)
    double t_setup_ms;    // allocs, handle, bufferSize
    double t_tileprep_ms; // CSR tile build (host)
    double t_h2d_ms;      // H2D of X tiles + CSR tiles
    double t_spmm_ms;     // total GPU SpMM time (events)
    double t_d2h_ms;      // D2H of C tiles
    double t_misc_ms;     // correction so sum == end
};

// ---------------- main 3D-tiled overlap kernel (templated) ----------------
template<typename T>
SpmmTimings run_spmm_stream_3d_overlap(const CSR<T> &csr, int K,
                                       int tile_rows, int tile_K,
                                       bool use_tf32_math)
{
    /*std::cout << "\n=== SpMM 3D-tiled (A rows, X/C cols, "
              << (std::is_same<T,float>::value ? "float" : "double")
              << ", overlap H2D+compute"
              << (use_tf32_math && std::is_same<T,float>::value ? ", TF32 math" : "")
              << ") ===\n";
    */
    int m = csr.m;
    int n = csr.n;

    int num_row_tiles = (m + tile_rows - 1) / tile_rows;
    int num_k_tiles   = (K + tile_K   - 1) / tile_K;
    /*
    std::cout << "Row tiles: " << num_row_tiles
              << " (tile_rows=" << tile_rows << ")\n";
    std::cout << "K tiles:   " << num_k_tiles
              << " (tile_K="   << tile_K   << ")\n";
    */
    // Full dense X and C on host (column-major: n x K, m x K)
    std::vector<T> h_X((size_t)n * K, (T)1);
    std::vector<T> h_C((size_t)m * K, (T)0);

    // Precompute max nnz per row tile
    int max_nnz_tile = 0;
    for (int t = 0; t < num_row_tiles; ++t) {
        int r0 = t * tile_rows;
        int r1 = std::min(m, r0 + tile_rows);
        int p0 = csr.row_ptr[r0];
        int p1 = csr.row_ptr[r1];
        max_nnz_tile = std::max(max_nnz_tile, p1 - p0);
    }

    SpmmTimings Tm{};
    double t0 = wtime();

    // --- setup ---
    double t_setup_start = wtime();

    // device CSR tile buffers (double buffered)
    int *d_row_ptr_tile[2] = {nullptr, nullptr};
    int *d_col_idx_tile[2] = {nullptr, nullptr};
    T   *d_vals_tile[2]    = {nullptr, nullptr};

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&d_row_ptr_tile[b],
                              (tile_rows + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col_idx_tile[b],
                              max_nnz_tile * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_vals_tile[b],
                              max_nnz_tile * sizeof(T)));
    }

    // pinned host CSR tile buffers (double buffered)
    int *h_row_ptr_tile[2] = {nullptr, nullptr};
    int *h_col_idx_tile[2] = {nullptr, nullptr};
    T   *h_vals_tile[2]    = {nullptr, nullptr};

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMallocHost(&h_row_ptr_tile[b],
                                  (tile_rows + 1) * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_col_idx_tile[b],
                                  max_nnz_tile * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_vals_tile[b],
                                  max_nnz_tile * sizeof(T)));
    }

    // device dense tiles for X and C
    int K_tile_max = std::min(tile_K, K);
    int m_tile_max = std::min(tile_rows, m);

    T *d_X_tile = nullptr; // n x K_tile_max
    T *d_C_tile = nullptr; // tile_rows x K_tile_max

    CUDA_CHECK(cudaMalloc(&d_X_tile,
                          (size_t)n * K_tile_max * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_C_tile,
                          (size_t)tile_rows * K_tile_max * sizeof(T)));

    // cuSPARSE + streams
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    // Optional TF32 math mode (only if cuSPARSE supports it)
#if defined(CUSPARSE_MATH_TF32) && defined(CUSPARSE_DEFAULT_MATH)
    if (use_tf32_math && std::is_same<T,float>::value) {
        CUSPARSE_CHECK(cusparseSetMathMode(handle, CUSPARSE_MATH_TF32));
    } else {
        CUSPARSE_CHECK(cusparseSetMathMode(handle, CUSPARSE_DEFAULT_MATH));
    }
#else
    if (use_tf32_math && std::is_same<T,float>::value) {
        std::cerr << "Warning: TF32 requested but this cuSPARSE "
                     "does not support CUSPARSE_MATH_TF32; "
                     "falling back to plain FP32.\n";
    }
    (void)use_tf32_math;
#endif

    cudaStream_t stream_h2d, stream_compute;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_h2d, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUSPARSE_CHECK(cusparseSetStream(handle, stream_compute));

    // SpMM buffer size query with worst-case tile sizes
    cusparseSpMatDescr_t matA_tmp;
    cusparseDnMatDescr_t matX_tmp, matC_tmp;

    cudaDataType dt = getCudaType<T>();

    CUSPARSE_CHECK(cusparseCreateCsr(
        &matA_tmp,
        m_tile_max, n, max_nnz_tile,
        d_row_ptr_tile[0],
        d_col_idx_tile[0],
        d_vals_tile[0],
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        dt));

    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matX_tmp,
        n, K_tile_max, n,
        d_X_tile,
        dt,
        CUSPARSE_ORDER_COL));

    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matC_tmp,
        m_tile_max, K_tile_max, tile_rows,
        d_C_tile,
        dt,
        CUSPARSE_ORDER_COL));

    T alpha = (T)1;
    T beta  = (T)0;

    size_t bufferSize = 0;
    void *dBuffer = nullptr;

    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA_tmp, matX_tmp, &beta, matC_tmp,
        dt,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));

    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    CUSPARSE_CHECK(cusparseDestroySpMat(matA_tmp));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matX_tmp));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matC_tmp));

    // CUDA events for GPU SpMM timing
    cudaEvent_t ev_spmm_start, ev_spmm_stop;
    CUDA_CHECK(cudaEventCreate(&ev_spmm_start));
    CUDA_CHECK(cudaEventCreate(&ev_spmm_stop));

    // events to signal CSR H2D is done per buffer
    cudaEvent_t ev_h2d_done[2];
    CUDA_CHECK(cudaEventCreate(&ev_h2d_done[0]));
    CUDA_CHECK(cudaEventCreate(&ev_h2d_done[1]));

    Tm.t_setup_ms = (wtime() - t_setup_start) * 1e3;

    // --- main 3D tile loops (overlap CSR H2D with compute) ---
    double tileprep_ms_total = 0.0;
    double h2d_ms_total      = 0.0;
    double spmm_ms_total_gpu = 0.0;
    double d2h_ms_total      = 0.0;

    for (int tk = 0; tk < num_k_tiles; ++tk) {
        int k0 = tk * tile_K;
        int this_K = std::min(tile_K, K - k0);

        // H2D X tile (async)
        double t_h2d_X_start = wtime();
        for (int kk = 0; kk < this_K; ++kk) {
            const T *h_src = h_X.data() + (size_t)(k0 + kk) * n;
            T       *d_dst = d_X_tile   + (size_t)kk * n;
            CUDA_CHECK(cudaMemcpyAsync(d_dst, h_src,
                                       (size_t)n * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       stream_h2d));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_h2d)); // ensure X tile ready
        h2d_ms_total += (wtime() - t_h2d_X_start) * 1e3;

        int cur = 0;

        // preload first CSR tile
        if (num_row_tiles > 0) {
            int tr = 0;
            int r0 = tr * tile_rows;
            int r1 = std::min(m, r0 + tile_rows);
            int m_tile = r1 - r0;
            int p0 = csr.row_ptr[r0];
            int p1 = csr.row_ptr[r1];
            int nnz_tile = p1 - p0;

            double t_tileprep_start = wtime();
            h_row_ptr_tile[cur][0] = 0;
            for (int i = 0; i < m_tile; ++i) {
                int gr = r0 + i;
                int row_nnz = csr.row_ptr[gr + 1] - csr.row_ptr[gr];
                h_row_ptr_tile[cur][i + 1] = h_row_ptr_tile[cur][i] + row_nnz;
            }
            std::memcpy(h_col_idx_tile[cur],
                        csr.col_idx.data() + p0,
                        nnz_tile * sizeof(int));
            std::memcpy(h_vals_tile[cur],
                        csr.vals.data() + p0,
                        nnz_tile * sizeof(T));
            tileprep_ms_total += (wtime() - t_tileprep_start) * 1e3;

            double t_h2d_tile_start = wtime();
            CUDA_CHECK(cudaMemcpyAsync(d_row_ptr_tile[cur],
                                       h_row_ptr_tile[cur],
                                       (m_tile + 1) * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream_h2d));
            CUDA_CHECK(cudaMemcpyAsync(d_col_idx_tile[cur],
                                       h_col_idx_tile[cur],
                                       nnz_tile * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream_h2d));
            CUDA_CHECK(cudaMemcpyAsync(d_vals_tile[cur],
                                       h_vals_tile[cur],
                                       nnz_tile * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       stream_h2d));
            CUDA_CHECK(cudaEventRecord(ev_h2d_done[cur], stream_h2d));
            CUDA_CHECK(cudaStreamSynchronize(stream_h2d)); // for timing
            h2d_ms_total += (wtime() - t_h2d_tile_start) * 1e3;
        }

        for (int tr = 0; tr < num_row_tiles; ++tr) {
            int r0 = tr * tile_rows;
            int r1 = std::min(m, r0 + tile_rows);
            int m_tile = r1 - r0;
            int p0 = csr.row_ptr[r0];
            int p1 = csr.row_ptr[r1];
            int nnz_tile = p1 - p0;

            int next = 1 - cur;

            // preload next CSR tile while computing current
            if (tr + 1 < num_row_tiles) {
                int r0n = (tr + 1) * tile_rows;
                int r1n = std::min(m, r0n + tile_rows);
                int m_tile_n = r1n - r0n;
                int p0n = csr.row_ptr[r0n];
                int p1n = csr.row_ptr[r1n];
                int nnz_tile_n = p1n - p0n;

                double t_tileprep_start = wtime();
                h_row_ptr_tile[next][0] = 0;
                for (int i = 0; i < m_tile_n; ++i) {
                    int gr = r0n + i;
                    int row_nnz = csr.row_ptr[gr + 1] - csr.row_ptr[gr];
                    h_row_ptr_tile[next][i + 1] =
                        h_row_ptr_tile[next][i] + row_nnz;
                }
                std::memcpy(h_col_idx_tile[next],
                            csr.col_idx.data() + p0n,
                            nnz_tile_n * sizeof(int));
                std::memcpy(h_vals_tile[next],
                            csr.vals.data() + p0n,
                            nnz_tile_n * sizeof(T));
                tileprep_ms_total += (wtime() - t_tileprep_start) * 1e3;

                double t_h2d_tile_start = wtime();
                CUDA_CHECK(cudaMemcpyAsync(d_row_ptr_tile[next],
                                           h_row_ptr_tile[next],
                                           (m_tile_n + 1) * sizeof(int),
                                           cudaMemcpyHostToDevice,
                                           stream_h2d));
                CUDA_CHECK(cudaMemcpyAsync(d_col_idx_tile[next],
                                           h_col_idx_tile[next],
                                           nnz_tile_n * sizeof(int),
                                           cudaMemcpyHostToDevice,
                                           stream_h2d));
                CUDA_CHECK(cudaMemcpyAsync(d_vals_tile[next],
                                           h_vals_tile[next],
                                           nnz_tile_n * sizeof(T),
                                           cudaMemcpyHostToDevice,
                                           stream_h2d));
                CUDA_CHECK(cudaEventRecord(ev_h2d_done[next], stream_h2d));
                h2d_ms_total += (wtime() - t_h2d_tile_start) * 1e3;
            }

            // wait for current CSR tile to be ready
            CUDA_CHECK(cudaStreamWaitEvent(stream_compute,
                                           ev_h2d_done[cur], 0));

            // zero C tile
            CUDA_CHECK(cudaMemsetAsync(d_C_tile, 0,
                                       (size_t)m_tile * this_K * sizeof(T),
                                       stream_compute));

            // descriptors for this tile
            cusparseSpMatDescr_t matA_tile;
            cusparseDnMatDescr_t matX_tile, matC_tile;

            CUSPARSE_CHECK(cusparseCreateCsr(
                &matA_tile,
                m_tile, n, nnz_tile,
                d_row_ptr_tile[cur],
                d_col_idx_tile[cur],
                d_vals_tile[cur],
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                dt));

            CUSPARSE_CHECK(cusparseCreateDnMat(
                &matX_tile,
                n, this_K, n,
                d_X_tile,
                dt,
                CUSPARSE_ORDER_COL));

            CUSPARSE_CHECK(cusparseCreateDnMat(
                &matC_tile,
                m_tile, this_K, tile_rows,
                d_C_tile,
                dt,
                CUSPARSE_ORDER_COL));

            // SpMM
            CUDA_CHECK(cudaEventRecord(ev_spmm_start, stream_compute));
            CUSPARSE_CHECK(cusparseSpMM(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA_tile, matX_tile, &beta, matC_tile,
                dt,
                CUSPARSE_SPMM_ALG_DEFAULT,
                dBuffer));
            CUDA_CHECK(cudaEventRecord(ev_spmm_stop, stream_compute));
            CUDA_CHECK(cudaEventSynchronize(ev_spmm_stop));

            float spmm_ms_gpu = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&spmm_ms_gpu,
                                            ev_spmm_start, ev_spmm_stop));
            spmm_ms_total_gpu += spmm_ms_gpu;

            // D2H C tile to host  (FIXED: correct cudaMemcpyAsync signature)
            double t_d2h_tile_start = wtime();
            for (int kk = 0; kk < this_K; ++kk) {
                const T *d_src = d_C_tile + (size_t)kk * tile_rows;
                T       *h_dst =
                    h_C.data() + (size_t)(k0 + kk) * m + r0;
                CUDA_CHECK(cudaMemcpyAsync(h_dst,
                                           d_src,
                                           (size_t)m_tile * sizeof(T),
                                           cudaMemcpyDeviceToHost,
                                           stream_compute));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream_compute));
            d2h_ms_total += (wtime() - t_d2h_tile_start) * 1e3;

            CUSPARSE_CHECK(cusparseDestroySpMat(matA_tile));
            CUSPARSE_CHECK(cusparseDestroyDnMat(matX_tile));
            CUSPARSE_CHECK(cusparseDestroyDnMat(matC_tile));

            cur = next;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double t_end_ms = (wtime() - t0) * 1e3;

    // cleanup
    CUDA_CHECK(cudaEventDestroy(ev_spmm_start));
    CUDA_CHECK(cudaEventDestroy(ev_spmm_stop));
    CUDA_CHECK(cudaEventDestroy(ev_h2d_done[0]));
    CUDA_CHECK(cudaEventDestroy(ev_h2d_done[1]));

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFree(d_row_ptr_tile[b]));
        CUDA_CHECK(cudaFree(d_col_idx_tile[b]));
        CUDA_CHECK(cudaFree(d_vals_tile[b]));
        CUDA_CHECK(cudaFreeHost(h_row_ptr_tile[b]));
        CUDA_CHECK(cudaFreeHost(h_col_idx_tile[b]));
        CUDA_CHECK(cudaFreeHost(h_vals_tile[b]));
    }

    CUDA_CHECK(cudaFree(d_X_tile));
    CUDA_CHECK(cudaFree(d_C_tile));
    CUDA_CHECK(cudaFree(dBuffer));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream_h2d));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));

    // accounting
    Tm.t_end_ms      = t_end_ms;
    Tm.t_tileprep_ms = tileprep_ms_total;
    Tm.t_h2d_ms      = h2d_ms_total;
    Tm.t_spmm_ms     = spmm_ms_total_gpu;
    Tm.t_d2h_ms      = d2h_ms_total;

    double accounted =
        Tm.t_setup_ms +
        Tm.t_tileprep_ms +
        Tm.t_h2d_ms +
        Tm.t_d2h_ms;

    Tm.t_misc_ms = Tm.t_end_ms - accounted;
    if (Tm.t_misc_ms < 0.0) Tm.t_misc_ms = 0.0;

    /*std::cout << "\n=== Detailed timing (host ms; GPU SpMM ms separate) ===\n";
    std::cout << "Setup (alloc+cuSPARSE):      " << Tm.t_setup_ms     << " ms\n";
    std::cout << "Tile host prep (CSR build):  " << Tm.t_tileprep_ms  << " ms\n";
    std::cout << "H2D (X tiles + CSR tiles):   " << Tm.t_h2d_ms       << " ms\n";
    std::cout << "D2H (C tiles):               " << Tm.t_d2h_ms       << " ms\n";
    std::cout << "Misc / unaccounted (host):   " << Tm.t_misc_ms      << " ms\n";
    std::cout << "End-to-end (measured):       " << Tm.t_end_ms       << " ms\n";
    std::cout << "GPU SpMM time (events sum):  " << Tm.t_spmm_ms      << " ms\n";
    */
    return Tm;
}

// ---------------- precision selection & main ----------------

enum class PrecMode {
    FP32,
    FP32_TF32,
    FP64
};

PrecMode parse_precision(const std::string &s)
{
    std::string lower = s;
    for (char &c : lower) c = std::tolower(c);

    if (lower == "fp32" || lower == "float32" || lower == "float")
        return PrecMode::FP32;
    if (lower == "tf32")
        return PrecMode::FP32_TF32;
    if (lower == "fp64" || lower == "double" || lower == "float64")
        return PrecMode::FP64;

    std::cerr << "Unknown precision '" << s
              << "'. Use: fp32 | tf32 | fp64 | double | float64\n";
    std::exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
	
    //printf("===================================================================\n");
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " matrix.mtx [K=32] [tile_rows=10000] [tile_K=64] [precision]\n"
                  << "  precision: fp32 | tf32 | fp64 | double | float64\n";
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];
    int K         = (argc >= 3) ? std::atoi(argv[2]) : 32;
    int tile_rows = (argc >= 4) ? std::atoi(argv[3]) : 10000;
    int tile_K    = (argc >= 5) ? std::atoi(argv[4]) : 64;
    std::string prec_str = (argc >= 6) ? argv[5] : "fp32";

    double t0 = wtime();
    PrecMode mode = parse_precision(prec_str);
    //std::cout << "Requested precision: " << prec_str << "\n";

    if (mode == PrecMode::FP64) {
        using T = double;
        CSR<T> csr = read_matrix_market_to_csr<T>(filename);
        SpmmTimings t = run_spmm_stream_3d_overlap<T>(csr, K, tile_rows, tile_K,
                                                      /*use_tf32_math=*/false);
        
	t0 = 1e3*(wtime() - t0);
	std::cout << "\nSPMM_2\n";
        std::cout << "GPU SpMM="  << t.t_spmm_ms
                  << " ms, H2D=" << t.t_h2d_ms
                  << " ms, D2H=" << t.t_d2h_ms
                  << " ms, End2End=" << t0 << " ms\n";
    } else {
        using T = float;
        CSR<T> csr = read_matrix_market_to_csr<T>(filename);
        bool use_tf32_math = (mode == PrecMode::FP32_TF32);
        SpmmTimings t = run_spmm_stream_3d_overlap<T>(csr, K, tile_rows, tile_K,
                                                      use_tf32_math);
        t0 = 1e3*(wtime() - t0);

        std::cout << "\nSPMM_2:"
                  << (use_tf32_math ? "TF32 math" : "FP32")
                  << ", overlap) ===\n";
        std::cout << "GPU SpMM="  << t.t_spmm_ms
                  << " ms, H2D=" << t.t_h2d_ms
                  << " ms, D2H=" << t.t_d2h_ms
                  << " ms, End2End=" << t0 << " ms\n";
    }

    return 0;
}

