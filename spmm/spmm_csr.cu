#include <cstring>
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

// ---------------- timing struct ----------------
struct SpmmTimings {
    double t_gpu_alloc  = 0.0;  // all cudaMalloc (incl. dBuffer)
    double t_cpu_alloc  = 0.0;  // host allocations (e.g., h_C)
    double t_cusparse   = 0.0;  // handle + descriptors + bufferSize query
    double t_dealloc    = 0.0;  // destroy/free everything (CPU wall time)
    double t_d2h_ms     = 0.0;  // D2H time for C (CUDA events)
    double t_h2d_ms     = 0.0;  // H2D time for CSR (CUDA events)
    double t_csr        = 0.0;  // read + build CSR (CPU)
    double t_spmm_ms    = 0.0;  // sum of SpMM kernel times (CUDA events)
    double t_total_ms   = 0.0;  // end-to-end (main timing)
};

// ---------------- device fill kernel ----------------
template<typename T>
__global__ void fill_matrix_kernel(T* __restrict__ A, size_t n_elem, T value)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elem)
        A[idx] = value;
}

template<typename T>
void fill_matrix_device(T* dA, size_t rows, size_t cols, T value)
{
    size_t n_elem = (size_t)rows * (size_t)cols;
    size_t block  = 256;
    size_t grid   = (n_elem + block - 1) / block;

    fill_matrix_kernel<<<grid, block>>>(dA, n_elem, value);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------- timing util ----------------
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

    return csr;
}

// ---------------- SpMM full GPU with timings ----------------
void run_spmm_full_gpu(const CSR &csr, int K, int num_iters, SpmmTimings &Tm)
{
    std::cout << "\n=== SpMM full GPU (CSR resident) ===\n";

    int m   = csr.m;
    int n   = csr.n;
    int nnz = csr.nnz;

    std::cout << "Matrix: m=" << m << ", n=" << n << ", nnz=" << nnz
              << ", K=" << K << ", iters=" << num_iters << "\n";

    int    *d_row_ptr = nullptr, *d_col_idx = nullptr;
    double *d_vals    = nullptr, *d_X = nullptr, *d_C = nullptr;

    // Simple sanity: avoid obviously insane K (won't catch everything, but helps)
    if (K <= 0) {
        std::cerr << "Error: K must be > 0\n";
        std::exit(EXIT_FAILURE);
    }

    // ----------- phase 1: GPU allocations -----------
    double t0 = wtime();
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (m + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals,    nnz * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_X,       (size_t)n * K * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_C,       (size_t)m * K * sizeof(double)));
    double t1 = wtime();
    Tm.t_gpu_alloc += 1000.0 * (t1 - t0);

    // ----------- phase 2: cuSPARSE handle + descriptors + workspace query -----------
    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matX, matC;
    size_t bufferSize = 0;
    void *dBuffer = nullptr;

    t0 = wtime();
    CUSPARSE_CHECK(cusparseCreate(&handle));

    CUSPARSE_CHECK(cusparseCreateCsr(
        &matA,
        m, n, nnz,
        d_row_ptr,
        d_col_idx,
        d_vals,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matX,
        n, K, n, d_X,          // ld = n (column-major)
        CUDA_R_64F,
        CUSPARSE_ORDER_COL));

    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matC,
        m, K, m, d_C,          // ld = m (column-major)
        CUDA_R_64F,
        CUSPARSE_ORDER_COL));

    double alpha = 1.0, beta = 0.0;

    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matX, &beta, matC,
        CUDA_R_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));
    t1 = wtime();
    Tm.t_cusparse += 1000.0 * (t1 - t0);

    // workspace alloc counts as GPU alloc
    t0 = wtime();
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));
    t1 = wtime();
    Tm.t_gpu_alloc += 1000.0 * (t1 - t0);

    // ----------- initialize X directly on the GPU -----------
    fill_matrix_device<double>(d_X, n, K, 1.0);
    CUDA_CHECK(cudaDeviceSynchronize());


    // ----------- CPU alloc for output C (PINNED) -----------
    t0 = wtime();
    double* h_C = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_C, (size_t)m * (size_t)K * sizeof(double)));

    // optional: initialize to zero
    std::fill(h_C, h_C + (size_t)m * (size_t)K, 0.0);

    t1 = wtime();
    Tm.t_cpu_alloc += 1000.0 * (t1 - t0);

    // ----------- prepare events for timing -----------
    cudaEvent_t ev_h2d_start, ev_h2d_stop;
    cudaEvent_t ev_spmm_start, ev_spmm_stop;
    cudaEvent_t ev_d2h_start, ev_d2h_stop;

    CUDA_CHECK(cudaEventCreate(&ev_h2d_start));
    CUDA_CHECK(cudaEventCreate(&ev_h2d_stop));
    CUDA_CHECK(cudaEventCreate(&ev_spmm_start));
    CUDA_CHECK(cudaEventCreate(&ev_spmm_stop));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_start));
    CUDA_CHECK(cudaEventCreate(&ev_d2h_stop));

    // ----------- H2D CSR once (timed) -----------
    // Create pinned staging for CSR
    int    *h_row_ptr_p = nullptr;
    int    *h_col_idx_p = nullptr;
    double *h_vals_p    = nullptr;

    CUDA_CHECK(cudaMallocHost(&h_row_ptr_p, (m + 1) * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_col_idx_p, nnz * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&h_vals_p,    nnz * sizeof(double)));

    std::memcpy(h_row_ptr_p, csr.row_ptr.data(), (m + 1) * sizeof(int));
    std::memcpy(h_col_idx_p, csr.col_idx.data(), nnz * sizeof(int));
    std::memcpy(h_vals_p,    csr.vals.data(),    nnz * sizeof(double));
    
    
    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(ev_h2d_start));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, csr.row_ptr.data(), (m + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, csr.col_idx.data(), nnz * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, csr.vals.data(), nnz * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev_h2d_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_h2d_stop));
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, ev_h2d_start, ev_h2d_stop));
    Tm.t_h2d_ms += (h2d_ms);

    // ----------- iterations: SpMM only (timed) -----------
    Tm.t_spmm_ms = 0.0;
    for (int it = 0; it < num_iters; ++it) {
        float spmm_ms = 0.0f;
        CUDA_CHECK(cudaEventRecord(ev_spmm_start));
        CUSPARSE_CHECK(cusparseSpMM(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matX, &beta, matC,
            CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer));
        CUDA_CHECK(cudaEventRecord(ev_spmm_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_spmm_stop));
        CUDA_CHECK(cudaEventElapsedTime(&spmm_ms, ev_spmm_start, ev_spmm_stop));
        Tm.t_spmm_ms += spmm_ms;
    }

    // ----------- D2H C once (timed) -----------
    float d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventRecord(ev_d2h_start));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, (size_t)m * K * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_d2h_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_d2h_stop));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, ev_d2h_start, ev_d2h_stop));
    Tm.t_d2h_ms = d2h_ms;

    // ----------- destroy timing events & dealloc -----------

    double t_dealloc_start = wtime();

    CUDA_CHECK(cudaEventDestroy(ev_h2d_start));
    CUDA_CHECK(cudaEventDestroy(ev_h2d_stop));
    CUDA_CHECK(cudaEventDestroy(ev_spmm_start));
    CUDA_CHECK(cudaEventDestroy(ev_spmm_stop));
    CUDA_CHECK(cudaEventDestroy(ev_d2h_start));
    CUDA_CHECK(cudaEventDestroy(ev_d2h_stop));

    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matX));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matC));
    CUSPARSE_CHECK(cusparseDestroy(handle));

    CUDA_CHECK(cudaFree(dBuffer));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_C));

    double t_dealloc_end = wtime();
    Tm.t_dealloc += 1000.0 * (t_dealloc_end - t_dealloc_start);

    // Print quick per-iter stats
    std::cout << "Total SpMM time over " << num_iters
              << " iters = " << Tm.t_spmm_ms << " ms\n";
    std::cout << "Average SpMM time per iter = "
              << (Tm.t_spmm_ms / num_iters) << " ms\n";
}

// ---------------- main ----------------
int main(int argc, char **argv)
{
    printf("===================================================================\n");
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " matrix.mtx [K=32] [iters=10]\n";
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];
    int K     = (argc >= 3) ? std::atoi(argv[2]) : 32;
    int iters = (argc >= 4) ? std::atoi(argv[3]) : 5;

    if (iters <= 0) {
        std::cerr << "Error: iters must be > 0\n";
        return EXIT_FAILURE;
    }

    std::cout << "Input matrix: " << filename
              << ", K=" << K << ", iters=" << iters << "\n";

    SpmmTimings Tm;

    double t_total_start = wtime();

    // --- CSR read + build timing ---
    double t_csr_start = wtime();
    CSR csr = read_matrix_market_to_csr(filename);
    double t_csr_end = wtime();
    Tm.t_csr = 1000.0 * (t_csr_end - t_csr_start);

    // --- GPU SpMM ---
    run_spmm_full_gpu(csr, K, iters, Tm);

    double t_total_end = wtime();
    Tm.t_total_ms = 1000.0 * (t_total_end - t_total_start);

    std::cout << "\n==== SUMMARY ====\n";
    std::cout << "End-to-end          = " << Tm.t_total_ms  << " ms\n";
    std::cout << "t_csr           = " << Tm.t_csr       << " ms\n";
    std::cout << "t_gpu_alloc     = " << Tm.t_gpu_alloc << " ms\n";
    std::cout << "t_cpu_alloc     = " << Tm.t_cpu_alloc << " ms\n";
    std::cout << "t_h2d_ms     = " << Tm.t_h2d_ms    << " ms\n";
    std::cout << "t_spmm_ms    = " << Tm.t_spmm_ms   << " ms\n";
    std::cout << "t_d2h_ms       = " << Tm.t_d2h_ms    << " ms\n";

    return 0;
}

