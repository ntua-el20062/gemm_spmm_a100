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

// ---------------- CSR struct + loader ----------------
struct CSR {
    int m = 0;
    int n = 0;
    int nnz = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> vals;
};

CSR read_matrix_market_to_csr(const std::string &filename)
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

    CSR csr;
    int m, n, nnz_header;
    {
        std::stringstream ss(line);
        ss >> m >> n >> nnz_header;
        csr.m = m;
        csr.n = n;
    }

    // read triplets
    std::vector<int> I, J;
    std::vector<double> V;
    I.reserve(nnz_header * (symmetric ? 2 : 1));
    J.reserve(nnz_header * (symmetric ? 2 : 1));
    V.reserve(nnz_header * (symmetric ? 2 : 1));

    int i, j;
    double val;
    while (fin >> i >> j >> val) {
        i--; j--;
        I.push_back(i);
        J.push_back(j);
        V.push_back(val);
        if (symmetric && i != j) {
            I.push_back(j);
            J.push_back(i);
            V.push_back(val);
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

    std::cout << "Loaded CSR: m=" << csr.m << " n=" << csr.n
              << " nnz=" << csr.nnz
              << " symmetric=" << (symmetric ? "yes" : "no") << "\n";
    return csr;
}

// Simple timing struct for summary
struct SpmmTimings {
    double t_end_ms;   // end-to-end
    double t_setup_ms; // setup (allocs, handle, bufferSize)
    double t_tileprep_ms; // CSR tile build on host
    double t_h2d_ms;   // H2D (X tiles + CSR tiles)
    double t_spmm_ms;  // sum of host+sync time for all SpMM calls
    double t_d2h_ms;   // D2H (C tiles)
    double t_misc_ms;  // difference so sum == t_end_ms
};

SpmmTimings run_spmm_stream_3d(const CSR &csr, int K, int tile_rows, int tile_K)
{
    std::cout << "\n=== SpMM 3D-tiled (A rows, X/C cols; only tile C on device) ===\n";

    int m = csr.m;
    int n = csr.n;
    int nnz = csr.nnz;

    int num_row_tiles = (m + tile_rows - 1) / tile_rows;
    int num_k_tiles   = (K + tile_K   - 1) / tile_K;

    std::cout << "Row tiles: " << num_row_tiles
              << " (tile_rows=" << tile_rows << ")\n";
    std::cout << "K tiles:   " << num_k_tiles
              << " (tile_K="   << tile_K   << ")\n";

    // Full dense X and C reside on the host only
    std::vector<double> h_X((size_t)n * K, 1.0);
    std::vector<double> h_C((size_t)m * K, 0.0);

    // Precompute max nnz per row tile
    int max_nnz_tile = 0;
    for (int t = 0; t < num_row_tiles; ++t) {
        int r0 = t * tile_rows;
        int r1 = std::min(m, r0 + tile_rows);
        int p0 = csr.row_ptr[r0];
        int p1 = csr.row_ptr[r1];
        max_nnz_tile = std::max(max_nnz_tile, p1 - p0);
    }

    // Timing
    double t0 = wtime();
    SpmmTimings T{};
    double t_setup_start = wtime();

    // Allocate device CSR tile buffers (max tile)
    int *d_row_ptr_tile = nullptr;
    int *d_col_idx_tile = nullptr;
    double *d_vals_tile = nullptr;

    CUDA_CHECK(cudaMalloc(&d_row_ptr_tile, (tile_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx_tile, max_nnz_tile * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals_tile,    max_nnz_tile * sizeof(double)));

    // Pinned host tile buffers for CSR
    int    *h_row_ptr_tile = nullptr;
    int    *h_col_idx_tile = nullptr;
    double *h_vals_tile    = nullptr;

    CUDA_CHECK(cudaMallocHost(&h_row_ptr_tile, (tile_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_col_idx_tile, max_nnz_tile * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_vals_tile,    max_nnz_tile * sizeof(double)));

    // Device dense tiles for X and C
    int K_tile_max = std::min(tile_K, K);
    int m_tile_max = std::min(tile_rows, m);

    double *d_X_tile = nullptr; // n x K_tile_max
    double *d_C_tile = nullptr; // tile_rows x K_tile_max

    CUDA_CHECK(cudaMalloc(&d_X_tile,
                          (size_t)n * K_tile_max * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C_tile,
                          (size_t)tile_rows * K_tile_max * sizeof(double)));

    // cuSPARSE setup
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    // SpMM buffer size query with worst-case tile sizes
    cusparseSpMatDescr_t matA_tmp;
    cusparseDnMatDescr_t matX_tmp, matC_tmp;

    CUSPARSE_CHECK(cusparseCreateCsr(
        &matA_tmp,
        m_tile_max, n, max_nnz_tile,
        d_row_ptr_tile,
        d_col_idx_tile,
        d_vals_tile,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matX_tmp,
        n, K_tile_max, n,
        d_X_tile,
        CUDA_R_64F,
        CUSPARSE_ORDER_COL));

    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matC_tmp,
        m_tile_max, K_tile_max, tile_rows,
        d_C_tile,
        CUDA_R_64F,
        CUSPARSE_ORDER_COL));

    double alpha = 1.0;
    double beta  = 0.0;
    size_t bufferSize = 0;
    void *dBuffer = nullptr;

    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA_tmp, matX_tmp, &beta, matC_tmp,
        CUDA_R_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));

    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    CUSPARSE_CHECK(cusparseDestroySpMat(matA_tmp));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matX_tmp));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matC_tmp));

    T.t_setup_ms = (wtime() - t_setup_start) * 1e3;

    // ----------------- MAIN 3D TILE LOOPS -----------------
    double tileprep_ms_total = 0.0;
    double h2d_ms_total      = 0.0;
    double spmm_ms_total     = 0.0;
    double d2h_ms_total      = 0.0;

    // Loop over K tiles
    for (int tk = 0; tk < num_k_tiles; ++tk) {
        int k0 = tk * tile_K;
        int this_K = std::min(tile_K, K - k0);

        // --- H2D for X tile (copy columns [k0, k0+this_K) ) ---
        // h_X is column-major (n rows, K cols): column j at h_X + j*n
        double t_h2d_X_start = wtime();
        for (int kk = 0; kk < this_K; ++kk) {
            const double *h_src = h_X.data() + (size_t)(k0 + kk) * n;
            double       *d_dst = d_X_tile   + (size_t)kk * n;
            CUDA_CHECK(cudaMemcpy(d_dst, h_src,
                                  (size_t)n * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }
        h2d_ms_total += (wtime() - t_h2d_X_start) * 1e3;

        // Loop over row tiles
        for (int tr = 0; tr < num_row_tiles; ++tr) {
            int r0 = tr * tile_rows;
            int r1 = std::min(m, r0 + tile_rows);
            int m_tile = r1 - r0;

            // --- build CSR tile on host (pinned buffers) ---
            double t_tileprep_start = wtime();

            int p0 = csr.row_ptr[r0];
            int p1 = csr.row_ptr[r1];
            int nnz_tile = p1 - p0;

            h_row_ptr_tile[0] = 0;
            for (int i = 0; i < m_tile; ++i) {
                int gr = r0 + i;
                int row_nnz = csr.row_ptr[gr + 1] - csr.row_ptr[gr];
                h_row_ptr_tile[i + 1] = h_row_ptr_tile[i] + row_nnz;
            }

            std::memcpy(h_col_idx_tile,
                        csr.col_idx.data() + p0,
                        nnz_tile * sizeof(int));
            std::memcpy(h_vals_tile,
                        csr.vals.data() + p0,
                        nnz_tile * sizeof(double));

            tileprep_ms_total += (wtime() - t_tileprep_start) * 1e3;

            // --- H2D CSR tile ---
            double t_h2d_tile_start = wtime();
            CUDA_CHECK(cudaMemcpy(d_row_ptr_tile,
                                  h_row_ptr_tile,
                                  (m_tile + 1) * sizeof(int),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_col_idx_tile,
                                  h_col_idx_tile,
                                  nnz_tile * sizeof(int),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_vals_tile,
                                  h_vals_tile,
                                  nnz_tile * sizeof(double),
                                  cudaMemcpyHostToDevice));
            h2d_ms_total += (wtime() - t_h2d_tile_start) * 1e3;

            // --- create descriptors for this tile ---
            cusparseSpMatDescr_t matA_tile;
            cusparseDnMatDescr_t matX_tile, matC_tile;

            CUSPARSE_CHECK(cusparseCreateCsr(
                &matA_tile,
                m_tile, n, nnz_tile,
                d_row_ptr_tile,
                d_col_idx_tile,
                d_vals_tile,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_R_64F));

            // X tile: n x this_K, leading dim n
            CUSPARSE_CHECK(cusparseCreateDnMat(
                &matX_tile,
                n, this_K, n,
                d_X_tile,
                CUDA_R_64F,
                CUSPARSE_ORDER_COL));

            // C tile: m_tile x this_K, leading dim = tile_rows
            // (we allocate tile_rows x K_tile_max, but only use first m_tile rows)
            CUSPARSE_CHECK(cusparseCreateDnMat(
                &matC_tile,
                m_tile, this_K, tile_rows,
                d_C_tile,
                CUDA_R_64F,
                CUSPARSE_ORDER_COL));

            // --- SpMM for this (row, K) tile ---
            double t_spmm_start = wtime();
            CUSPARSE_CHECK(cusparseSpMM(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA_tile, matX_tile, &beta, matC_tile,
                CUDA_R_64F,
                CUSPARSE_SPMM_ALG_DEFAULT,
                dBuffer));
            // default stream, so this is sync w.r.t host after returning
            spmm_ms_total += (wtime() - t_spmm_start) * 1e3;

            // --- D2H C tile into full host C ---
            double t_d2h_tile_start = wtime();
            for (int kk = 0; kk < this_K; ++kk) {
                const double *d_src = d_C_tile + (size_t)kk * tile_rows;
                double       *h_dst =
                    h_C.data() + (size_t)(k0 + kk) * m + r0;
                CUDA_CHECK(cudaMemcpy(h_dst,
                                      d_src,
                                      (size_t)m_tile * sizeof(double),
                                      cudaMemcpyDeviceToHost));
            }
            d2h_ms_total += (wtime() - t_d2h_tile_start) * 1e3;

            CUSPARSE_CHECK(cusparseDestroySpMat(matA_tile));
            CUSPARSE_CHECK(cusparseDestroyDnMat(matX_tile));
            CUSPARSE_CHECK(cusparseDestroyDnMat(matC_tile));
        } // end row tiles
    } // end K tiles

    CUDA_CHECK(cudaDeviceSynchronize());
    double t_end_ms = (wtime() - t0) * 1e3;

    // free resources
    CUDA_CHECK(cudaFree(d_row_ptr_tile));
    CUDA_CHECK(cudaFree(d_col_idx_tile));
    CUDA_CHECK(cudaFree(d_vals_tile));

    CUDA_CHECK(cudaFreeHost(h_row_ptr_tile));
    CUDA_CHECK(cudaFreeHost(h_col_idx_tile));
    CUDA_CHECK(cudaFreeHost(h_vals_tile));

    CUDA_CHECK(cudaFree(d_X_tile));
    CUDA_CHECK(cudaFree(d_C_tile));
    CUDA_CHECK(cudaFree(dBuffer));

    CUSPARSE_CHECK(cusparseDestroy(handle));

    // accounting
    T.t_end_ms     = t_end_ms;
    T.t_tileprep_ms = tileprep_ms_total;
    T.t_h2d_ms     = h2d_ms_total;
    T.t_spmm_ms    = spmm_ms_total;
    T.t_d2h_ms     = d2h_ms_total;

    double accounted = T.t_setup_ms +
                       T.t_tileprep_ms +
                       T.t_h2d_ms +
                       T.t_spmm_ms +
                       T.t_d2h_ms;

    T.t_misc_ms = T.t_end_ms - accounted;
    if (T.t_misc_ms < 0.0) T.t_misc_ms = 0.0;

    // print breakdown
    std::cout << "\n=== Detailed timing (host, ms) ===\n";
    std::cout << "Setup (alloc+cuSPARSE):     " << T.t_setup_ms     << " ms\n";
    std::cout << "Tile host prep (CSR build): " << T.t_tileprep_ms  << " ms\n";
    std::cout << "H2D (X tiles + CSR tiles):  " << T.t_h2d_ms       << " ms\n";
    std::cout << "SpMM (all tiles, host+sync):" << T.t_spmm_ms      << " ms\n";
    std::cout << "D2H (C tiles):              " << T.t_d2h_ms       << " ms\n";
    std::cout << "Misc / unaccounted:         " << T.t_misc_ms      << " ms\n";

    return T;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " matrix.mtx [K=32] [tile_rows=10000] [tile_K=64]\n";
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];
    int K         = (argc >= 3) ? std::atoi(argv[2]) : 32;
    int tile_rows = (argc >= 4) ? std::atoi(argv[3]) : 10000;
    int tile_K    = (argc >= 5) ? std::atoi(argv[4]) : 64;

    double t0 = wtime();
    CSR csr = read_matrix_market_to_csr(filename);
    SpmmTimings t = run_spmm_stream_3d(csr, K, tile_rows, tile_K);

    t0 = 1000*(wtime() - t0);
    std::cout << "\n=== Summary (3D-tiled) ===\n";
    std::cout << "SpMM="     << t.t_spmm_ms
              << " ms, H2D=" << t.t_h2d_ms
              << " ms, D2H=" << t.t_d2h_ms
              << " ms, End2End=" << t0 << " ms\n";

    return 0;
}

