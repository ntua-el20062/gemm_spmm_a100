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
#include <cstdio>
#include <unistd.h>
#include <math.h>
#include <limits>

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


double wtime()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

//CSR struct
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

    //skip comments
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

    return csr;
}

inline cudaDataType getCudaType(float)  { return CUDA_R_32F; }
inline cudaDataType getCudaType(double) { return CUDA_R_64F; }

template<typename T>
cudaDataType getCudaType()
{
    return getCudaType(T{});
}

//timing struct
struct SpmmTimings {
    double t_gpu_alloc = 0.0;   // device allocations + cuSPARSE buffer, streams, events
    double t_cpu_alloc = 0.0;   // host allocations/initialization in this function
    double t_spmm_ms   = 0.0;   // sum of all SpMM calls (event-based)
    double t_h2d_ms    = 0.0;   // sum of all H2D memcpy groups (event-based)
    double t_d2h_ms    = 0.0;   // sum of all D2H memcpy groups (event-based)
    double t_compute    = 0.0;
};


template<typename T>
bool check_A_times_I_equals_A(const CSR<T>& A, const T* C,int ldc,double tol) {
    int m = A.m;
    int n = A.n;
    //some matrices have duplicates: build dense A with accumulated duplicates and col-major to match C
    std::vector<double> A_dense((size_t)m * n, 0.0);

    for (int i = 0; i < m; ++i) {
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
            int j = A.col_idx[k];
            double val = (double)A.vals[k];
            A_dense[(size_t)i + (size_t)j * (size_t)m] += val;
        }
    }

    bool ok = true;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            double a = A_dense[(size_t)i + (size_t)j * (size_t)m];
            double c = (double)C[(size_t)i + (size_t)j * (size_t)ldc];
            double diff = std::fabs(a - c);

            if (diff > tol) {
                std::cout << "Value mismatch at (" << i << "," << j
                          << "): C=" << c << " vs A=" << a
                          << " (diff=" << diff << ")\n";
                ok = false;
            }
        }
    }
    return ok;
}


template<typename T>
CSR<T> dense_to_csr(const T* C, int m, int n, int ld, double tol)
{
    CSR<T> out;
    out.m = m;
    out.n = n;

    out.row_ptr.assign(m + 1, 0);
    std::vector<int> I;
    std::vector<int> J;
    std::vector<T>   V;

    I.reserve(m * 4); // arbitrary guess; will grow if needed
    J.reserve(m * 4);
    V.reserve(m * 4);

    // scan row by row
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            // column-major: C(i,j) = C[i + j*ld]
            T val = C[(size_t)i + (size_t)j * (size_t)ld];
            if (std::fabs((double)val) > tol) {
                I.push_back(i);
                J.push_back(j);
                V.push_back(val);
            }
        }
    }

    int nnz = (int)I.size();
    out.nnz = nnz;
    out.col_idx.resize(nnz);
    out.vals.resize(nnz);

    // build row_ptr (like in read_matrix_market_to_csr)
    for (int k = 0; k < nnz; ++k) {
        out.row_ptr[I[k] + 1]++;
    }
    for (int r = 0; r < m; ++r) {
        out.row_ptr[r + 1] += out.row_ptr[r];
    }

    std::vector<int> offset = out.row_ptr;
    for (int k = 0; k < nnz; ++k) {
        int row  = I[k];
        int dest = offset[row]++;
        out.col_idx[dest] = J[k];
        out.vals[dest]    = V[k];
    }

    return out;
}

template<typename T>
bool csr_equal(const CSR<T>& A, const CSR<T>& B, double tol)
{
    if (A.m != B.m || A.n != B.n || A.nnz != B.nnz) {
        std::cerr << "CSR mismatch: dims or nnz differ\n";
        return false;
    }

    if (A.row_ptr.size() != B.row_ptr.size() ||
        A.col_idx.size() != B.col_idx.size() ||
        A.vals.size()    != B.vals.size()) {
        std::cerr << "CSR mismatch: internal array sizes differ\n";
        return false;
    }

    for (size_t i = 0; i < A.row_ptr.size(); ++i) {
        if (A.row_ptr[i] != B.row_ptr[i]) {
            std::cerr << "CSR mismatch: row_ptr[" << i << "] "
                      << A.row_ptr[i] << " vs " << B.row_ptr[i] << "\n";
            return false;
        }
    }

    for (size_t k = 0; k < A.col_idx.size(); ++k) {
        if (A.col_idx[k] != B.col_idx[k]) {
            std::cerr << "CSR mismatch: col_idx[" << k << "] "
                      << A.col_idx[k] << " vs " << B.col_idx[k] << "\n";
            return false;
        }
    }

    for (size_t k = 0; k < A.vals.size(); ++k) {
        double av = (double)A.vals[k];
        double bv = (double)B.vals[k];
        double diff = std::fabs(av - bv);
        if (diff > tol) {
            std::cerr << "CSR mismatch: vals[" << k << "] "
                      << av << " vs " << bv << " (diff=" << diff << ")\n";
            return false;
        }
    }

    return true;
}

void print_memory() {
    //GPU
    size_t free_bytes;
    size_t total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    // Convert bytes to GB for easier reading
    const float GB = 1024.0f * 1024.0f * 1024.0f;
    float free_gb = (float)free_bytes / GB;
    float total_gb = (float)total_bytes / GB;
    float used_gb = total_gb - free_gb;
    // Print the results
    printf("--- GPU Memory Usage ---\n");
    printf("Total GPU Memory: %.2f GB\n", total_gb);
    printf("Free GPU Memory:  %.2f GB\n", free_gb);
    printf("Used GPU Memory:  %.2f GB\n", used_gb);
    printf("------------------------\n\n");
    //CPU
    long rss_pages, virt_pages;
    FILE* f = fopen("/proc/self/statm", "r");
    fscanf(f, "%ld %ld", &virt_pages, &rss_pages);
    fclose(f);
    long page_size_kb = sysconf(_SC_PAGESIZE) / 1024;
    printf("Process RSS:  %.2f GB\n", rss_pages * page_size_kb / 1024.0 / 1024.0);
}


template<typename T>
SpmmTimings run_spmm(const CSR<T> &csr, int K, int tile_K, int buff_size, bool use_tf32_math) {
    SpmmTimings Tm{};

    int m   = csr.m;
    int n   = csr.n;
    int nnz = csr.nnz;

    if (m == 0 || n == 0 || K == 0) return Tm;

    double t_cpu_start = wtime();
    T *h_X = nullptr;
    T *h_C = nullptr;

    print_memory();

    CUDA_CHECK(cudaMallocHost(&h_X, sizeof(T) * n * K));
    CUDA_CHECK(cudaMallocHost(&h_C, sizeof(T) * m * K));

    // X = I_n (for A*I == A)
    if(csr.m < 4e3) {
	for (int j = 0; j < K; ++j) {
              for (int i = 0; i < n; ++i) {
                h_X[(size_t)i + (size_t)j * (size_t)n] = (i == j) ? (T)1 : (T)0;
              }
         }
    } else {
       for(int i=0; i < n*K; i++) {
    	    h_X[i] = (T)1;
       }
    }

    //print_memory();

    Tm.t_cpu_alloc = 1e3 * (wtime() - t_cpu_start);

    double t_gpu_start = wtime();

    int *d_row_ptr = nullptr;
    int *d_col_idx = nullptr;
    T   *d_vals    = nullptr;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, sizeof(int) * (m + 1)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, sizeof(int) * nnz));
    CUDA_CHECK(cudaMalloc(&d_vals,    sizeof(T)   * nnz));

    double t1 = wtime();
    CUDA_CHECK(cudaMemcpy(d_row_ptr, csr.row_ptr.data(),  sizeof(int) * (m + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, csr.col_idx.data(),
                          sizeof(int) * nnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, csr.vals.data(),
                          sizeof(T) * nnz, cudaMemcpyHostToDevice));
    Tm.t_h2d_ms += 1e3 * (wtime() - t1);

    int K_tile_max = tile_K;
    int num_tiles  = (K + tile_K - 1) / tile_K;

    //cyclic buffers
    const int NUM_BUF = buff_size;
    int buf_count = std::min(NUM_BUF, num_tiles); 

    std::vector<T*> d_X_tile(buf_count, nullptr);
    std::vector<T*> d_C_tile(buf_count, nullptr);

    for (int b = 0; b < buf_count; ++b) {
        CUDA_CHECK(cudaMalloc(&d_X_tile[b], sizeof(T) * n * K_tile_max));
        CUDA_CHECK(cudaMalloc(&d_C_tile[b], sizeof(T) * m * K_tile_max));
    }

    // ---------------- cuSPARSE + streams ----------------
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    cudaStream_t stream_h2d, stream_compute, stream_d2h;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_h2d,     cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_d2h,     cudaStreamNonBlocking));

    CUSPARSE_CHECK(cusparseSetStream(handle, stream_compute));

    cudaDataType dt = getCudaType<T>();

    cusparseSpMatDescr_t matA;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA,
                                     m, n, nnz,
                                     d_row_ptr, d_col_idx, d_vals,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, dt));

    // Dummy dense descriptors for buffer size query
    cusparseDnMatDescr_t matX_tmp, matC_tmp;
    CUSPARSE_CHECK(cusparseCreateDnMat(&matX_tmp, n, K_tile_max, n,
                                       nullptr, dt, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&matC_tmp, m, K_tile_max, m,
                                       nullptr, dt, CUSPARSE_ORDER_COL));

    T alpha = (T)1, beta = (T)0;
    size_t bufferSize = 0;
    void *dBuffer = nullptr;

    CUSPARSE_CHECK(cusparseSpMM_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, matX_tmp,
                                           &beta, matC_tmp,
                                           dt, CUSPARSE_SPMM_ALG_DEFAULT,
                                           &bufferSize));
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    CUSPARSE_CHECK(cusparseDestroyDnMat(matX_tmp));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matC_tmp));

    Tm.t_gpu_alloc = 1e3 * (wtime() - t_gpu_start);

    std::vector<cudaEvent_t> h2d_start, h2d_stop;
    std::vector<cudaEvent_t> spmm_start, spmm_stop;
    std::vector<cudaEvent_t> d2h_start, d2h_stop;

    std::vector<cudaEvent_t> ev_buf_free(buf_count);     //buffer reusable after D2H
    std::vector<cudaEvent_t> ev_h2d_done(buf_count);     //H2D finished
    std::vector<cudaEvent_t> ev_compute_done(buf_count); //SpMM finished

    for (int b = 0; b < buf_count; ++b) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_buf_free[b],     cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_h2d_done[b],     cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_compute_done[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(ev_buf_free[b], 0));  //initially all free
    }

    //event for t_computations time
    cudaEvent_t ev_loop_start, ev_loop_end;
    CUDA_CHECK(cudaEventCreate(&ev_loop_start));
    CUDA_CHECK(cudaEventCreate(&ev_loop_end));

    //preload tile 0
    if (num_tiles > 0) {
        int t0      = 0;
        int k0      = 0;
        int this_K0 = std::min(tile_K, K - k0);
        int buf0    = 0 % buf_count;

        CUDA_CHECK(cudaStreamWaitEvent(stream_h2d, ev_buf_free[buf0], 0));
        cudaEvent_t ev_s, ev_e;
        CUDA_CHECK(cudaEventCreate(&ev_s));
        CUDA_CHECK(cudaEventCreate(&ev_e));

        CUDA_CHECK(cudaEventRecord(ev_s, stream_h2d));

        CUDA_CHECK(cudaMemcpyAsync( d_X_tile[buf0], h_X + (size_t)k0 * n,sizeof(T) * n * this_K0, cudaMemcpyHostToDevice,stream_h2d));

        CUDA_CHECK(cudaEventRecord(ev_e, stream_h2d));
        h2d_start.push_back(ev_s);
        h2d_stop.push_back(ev_e);

        CUDA_CHECK(cudaEventRecord(ev_h2d_done[buf0], stream_h2d));
    }

    double t_comp = wtime();
    //main loop
    for (int t = 0; t < num_tiles; ++t) {

        int k0     = t * tile_K;
        int this_K = std::min(tile_K, K - k0);
        int buf    = t % buf_count;

        //d2h(t-1)
        if (t > 0) {
            int prev      = t - 1;
            int prev_buf  = prev % buf_count;
            int prev_k0   = prev * tile_K;
            int prev_K    = std::min(tile_K, K - prev_k0);

            CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, ev_compute_done[prev_buf], 0));

            cudaEvent_t ev_s, ev_e;
            CUDA_CHECK(cudaEventCreate(&ev_s));
            CUDA_CHECK(cudaEventCreate(&ev_e));
            CUDA_CHECK(cudaEventRecord(ev_s, stream_d2h));

            CUDA_CHECK(cudaMemcpyAsync(h_C + (size_t)prev_k0 * m,d_C_tile[prev_buf],sizeof(T) * m * prev_K, cudaMemcpyDeviceToHost,stream_d2h));

            CUDA_CHECK(cudaEventRecord(ev_e, stream_d2h));
            d2h_start.push_back(ev_s);
            d2h_stop.push_back(ev_e);

            CUDA_CHECK(cudaEventRecord(ev_buf_free[prev_buf], stream_d2h));
        }

        //spmm(t)
        {
            CUDA_CHECK(cudaStreamWaitEvent(stream_compute, ev_h2d_done[buf], 0));

            cudaEvent_t ev_s, ev_e;
            CUDA_CHECK(cudaEventCreate(&ev_s));
            CUDA_CHECK(cudaEventCreate(&ev_e));
            CUDA_CHECK(cudaEventRecord(ev_s, stream_compute));

            CUDA_CHECK(cudaMemsetAsync(d_C_tile[buf], 0,sizeof(T) * m * this_K, stream_compute));

            cusparseDnMatDescr_t matX, matC;
            CUSPARSE_CHECK(cusparseCreateDnMat(
                &matX, n, this_K, n,
                d_X_tile[buf], dt, CUSPARSE_ORDER_COL));
            CUSPARSE_CHECK(cusparseCreateDnMat(
                &matC, m, this_K, m,
                d_C_tile[buf], dt, CUSPARSE_ORDER_COL));

            CUSPARSE_CHECK(cusparseSpMM(
                handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matA, matX,
                &beta, matC,
                dt, CUSPARSE_SPMM_ALG_DEFAULT,
                dBuffer));

            CUDA_CHECK(cudaEventRecord(ev_e, stream_compute));
            spmm_start.push_back(ev_s);
            spmm_stop.push_back(ev_e);

            CUDA_CHECK(cudaEventRecord(ev_compute_done[buf], stream_compute));

            CUSPARSE_CHECK(cusparseDestroyDnMat(matX));
            CUSPARSE_CHECK(cusparseDestroyDnMat(matC));
        }

        //h2d(t+1)
        if (t + 1 < num_tiles) {
            int next      = t + 1;
            int next_buf  = next % buf_count;
            int next_k0   = next * tile_K;
            int next_K    = std::min(tile_K, K - next_k0);

            CUDA_CHECK(cudaStreamWaitEvent(stream_h2d, ev_buf_free[next_buf], 0));

            cudaEvent_t ev_s, ev_e;
            CUDA_CHECK(cudaEventCreate(&ev_s));
            CUDA_CHECK(cudaEventCreate(&ev_e));
            CUDA_CHECK(cudaEventRecord(ev_s, stream_h2d));

            CUDA_CHECK(cudaMemcpyAsync(d_X_tile[next_buf], h_X + (size_t)next_k0 * n, sizeof(T) * n * next_K, cudaMemcpyHostToDevice,stream_h2d));

            CUDA_CHECK(cudaEventRecord(ev_e, stream_h2d));

            h2d_start.push_back(ev_s);
            h2d_stop.push_back(ev_e);

            CUDA_CHECK(cudaEventRecord(ev_h2d_done[next_buf], stream_h2d));
        }
    }
    

    //d2h for last tile
    if (num_tiles > 0) {
        int last      = num_tiles - 1;
        int last_buf  = last % buf_count;
        int last_k0   = last * tile_K;
        int last_K    = std::min(tile_K, K - last_k0);

        CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, ev_compute_done[last_buf], 0));

        cudaEvent_t ev_s, ev_e;
        CUDA_CHECK(cudaEventCreate(&ev_s));
        CUDA_CHECK(cudaEventCreate(&ev_e));
        CUDA_CHECK(cudaEventRecord(ev_s, stream_d2h));

        CUDA_CHECK(cudaMemcpyAsync(h_C + (size_t)last_k0 * m,d_C_tile[last_buf], sizeof(T) * m * last_K, cudaMemcpyDeviceToHost, stream_d2h));

        CUDA_CHECK(cudaEventRecord(ev_e, stream_d2h));
        d2h_start.push_back(ev_s);
        d2h_stop.push_back(ev_e);

        CUDA_CHECK(cudaEventRecord(ev_buf_free[last_buf], stream_d2h));

    }

    CUDA_CHECK(cudaStreamSynchronize(stream_h2d));
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_d2h));

    Tm.t_compute += 1e3*(wtime() - t_comp);

    float ms;
    for (size_t i = 0; i < h2d_start.size(); ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms, h2d_start[i], h2d_stop[i]));
        Tm.t_h2d_ms += ms;
    }
    for (size_t i = 0; i < spmm_start.size(); ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms, spmm_start[i], spmm_stop[i]));
        Tm.t_spmm_ms += ms;
    }
    for (size_t i = 0; i < d2h_start.size(); ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms, d2h_start[i], d2h_stop[i]));
        Tm.t_d2h_ms += ms;
    }

/*    if (!h2d_start.empty()) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, h2d_start.front(), h2d_stop.front()));
        Tm.t_compute += ms;
    }

    if (!d2h_start.empty()) {
           float ms = 0.0f;
           CUDA_CHECK(cudaEventElapsedTime(&ms, d2h_start.back(),d2h_stop.back()));
           Tm.t_compute += ms;
    }
*/

    if(csr.m < 4e3) {
        double tol = std::is_same<T,float>::value ? 1e-4 : 1e-10;
        bool same  = check_A_times_I_equals_A(csr, h_C, m, tol);

        if (same) {
            std::cout << "CSR self-check: PASSED (A*I == A numerically)\n";
        } else {
            std::cout << "CSR self-check: FAILED (numerical mismatch)\n";
        }
    }

    for (auto &e : h2d_start) CUDA_CHECK(cudaEventDestroy(e));
    for (auto &e : h2d_stop)  CUDA_CHECK(cudaEventDestroy(e));
    for (auto &e : spmm_start) CUDA_CHECK(cudaEventDestroy(e));
    for (auto &e : spmm_stop)  CUDA_CHECK(cudaEventDestroy(e));
    for (auto &e : d2h_start) CUDA_CHECK(cudaEventDestroy(e));
    for (auto &e : d2h_stop)  CUDA_CHECK(cudaEventDestroy(e));

    for (int b = 0; b < buf_count; ++b) {
        CUDA_CHECK(cudaEventDestroy(ev_buf_free[b]));
        CUDA_CHECK(cudaEventDestroy(ev_h2d_done[b]));
        CUDA_CHECK(cudaEventDestroy(ev_compute_done[b]));
    }

    for (int b = 0; b < buf_count; ++b) {
        CUDA_CHECK(cudaFree(d_X_tile[b]));
        CUDA_CHECK(cudaFree(d_C_tile[b]));
    }

    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(dBuffer));

    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroy(handle));

    CUDA_CHECK(cudaStreamDestroy(stream_h2d));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaStreamDestroy(stream_d2h));

    CUDA_CHECK(cudaFreeHost(h_X));
    CUDA_CHECK(cudaFreeHost(h_C));

    return Tm;
}


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


template <typename T>
int compute_tile_K(const CSR<T>& csr, size_t free_bytes, int K) {
    const size_t m   = static_cast<size_t>(csr.m);
    const size_t n   = static_cast<size_t>(csr.n);
    const size_t nnz = static_cast<size_t>(csr.nnz);

    // CSR on device: rowptr + colidx + vals
    const size_t sparse_A_bytes =
        (m + 1) * sizeof(int) +          // row_ptr
        nnz * (sizeof(int) + sizeof(T)); // col_idx + vals

    const size_t bytes_per_K = 2 * (m + n) * sizeof(T);

    const double safety_fraction = 0.95;  // use 90% of free memory
    size_t usable_free = static_cast<size_t>(free_bytes * safety_fraction);

    if (usable_free <= sparse_A_bytes || bytes_per_K == 0) {
        return std::min(K, 64);  // small but non-zero
    }

    const size_t available_for_tiles = usable_free - sparse_A_bytes;
    size_t max_tile_K = available_for_tiles / bytes_per_K;

    if (max_tile_K == 0) {
        max_tile_K = 1;
    }

    if (K > 0) {
        max_tile_K = std::min(max_tile_K, static_cast<size_t>(K));
    }

    if (max_tile_K > static_cast<size_t>(std::numeric_limits<int>::max())) {
        max_tile_K = static_cast<size_t>(std::numeric_limits<int>::max());
    }

    // Debug info (optional)
    /*std::cout << "compute_tile_K: free_bytes=" << free_bytes
              << ", sparse_A_bytes=" << sparse_A_bytes
              << ", bytes_per_K=" << bytes_per_K
              << ", max_tile_K=" << max_tile_K << "\n";
    */
    return static_cast<int>(max_tile_K);
}


int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " matrix.mtx [K=32] [tile_K=64] [precision] [buf_count]\n"
                  << "  precision: fp32 | tf32 | fp64 | double | float64\n";
        return EXIT_FAILURE;
    }

    using T = double;
    std::string filename = argv[1];
    CSR<T> csr = read_matrix_market_to_csr<T>(filename);

    int K         = (argc >= 3) ? std::atoi(argv[2]) : csr.n;
    std::string prec_str = (argc >= 4) ? argv[3] : "double";
    int buf_count = (argc >= 5) ? std::atoi(argv[4]) : 2;

    PrecMode mode = parse_precision(prec_str);

    size_t free_bytes;
    size_t total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    double t0 = wtime();

    if (mode == PrecMode::FP64) {
        //CSR<T> csr = read_matrix_market_to_csr<T>(filename);
	int tile_K = compute_tile_K<T>(csr, free_bytes, K);

	std::cout << "tile_k= " << tile_K << std::endl;
	SpmmTimings t = run_spmm<T>(csr, K, tile_K , buf_count,
                                    /*use_tf32_math=*/false);
        t0 = 1000.0 * (wtime() - t0);
        std::cout << "\nSPMM OVERLAP STREAM (FP64, CSR resident, K-tiled)\n";
        std::cout << "End2End="      << t0            << " ms\n";
        std::cout << "t_cpu_alloc="  << t.t_cpu_alloc << " ms\n";
        std::cout << "t_gpu_alloc="  << t.t_gpu_alloc << " ms\n";
        std::cout << "t_h2d_ms="     << t.t_h2d_ms    << " ms\n";
        std::cout << "t_d2h_ms="     << t.t_d2h_ms    << " ms\n";
        std::cout << "t_spmm_ms="    << t.t_spmm_ms   << " ms\n";
        std::cout << "t_pure_computation_and_transfers="    << t.t_compute   << " ms\n";

    } else {
        //CSR<T> csr = read_matrix_market_to_csr<T>(filename);
        bool use_tf32_math = (mode == PrecMode::FP32_TF32);
        int tile_K = compute_tile_K<T>(csr, free_bytes, K);
	
	SpmmTimings t = run_spmm<T>(csr, K, tile_K, buf_count,
                                    use_tf32_math);
        t0 = 1000.0 * (wtime() - t0);

        std::cout << "\nSPMM STREAMING OVERLAP ("
                  << (use_tf32_math ? "FP32+TF32" : "FP32")
                  << ", CSR resident, K-tiled)\n";
        std::cout << "End2End="      << t0            << " ms\n";
        std::cout << "t_cpu_alloc="  << t.t_cpu_alloc << " ms\n";
        std::cout << "t_gpu_alloc="  << t.t_gpu_alloc << " ms\n";
        std::cout << "t_h2d_ms="     << t.t_h2d_ms    << " ms\n";
        std::cout << "t_d2h_ms="     << t.t_d2h_ms    << " ms\n";
        std::cout << "t_spmm_ms="    << t.t_spmm_ms   << " ms\n";
        std::cout << "t_pure_computation_and_transfers="    << t.t_compute   << " ms\n";

    }

    return 0;
}

