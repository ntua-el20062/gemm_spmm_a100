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

// -------- helpers to map T -> cudaDataType --------
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
};

//3D-tiled overlap kernel
template<typename T>
SpmmTimings run_spmm_stream_3d_overlap(const CSR<T> &csr, int K, int tile_rows, int tile_K, bool use_tf32_math)
{
    SpmmTimings Tm{};

    int m = csr.m;
    int n = csr.n;

    if (m == 0 || n == 0 || K == 0) {
        std::cout << "Trivial dimensions, nothing to do.\n";
        return Tm;
    }

    int num_row_tiles = (m + tile_rows - 1) / tile_rows;
    int num_k_tiles   = (K + tile_K   - 1) / tile_K;

    // ---- CPU allocations ----
    double t_cpu_alloc_start = wtime();

    T* h_X = nullptr;
    T* h_C = nullptr;

    CUDA_CHECK(cudaMallocHost(&h_X, (size_t)n * K * sizeof(T)));   // pinned
    CUDA_CHECK(cudaMallocHost(&h_C, (size_t)m * K * sizeof(T)));   // pinned

    std::memset(h_C, 0, (size_t)m * K * sizeof(T));

    // initialize X to something non-trivial
    for (size_t t = 0; t < (size_t)n * (size_t)K; ++t) {
        h_X[t] = (T)1;
    }

    Tm.t_cpu_alloc = 1000.0 * (wtime() - t_cpu_alloc_start);

    // Precompute max nnz per row tile
    int max_nnz_tile = 0;
    for (int t = 0; t < num_row_tiles; ++t) {
        int r0 = t * tile_rows;
        int r1 = std::min(m, r0 + tile_rows);
        int p0 = csr.row_ptr[r0];
        int p1 = csr.row_ptr[r1];
        max_nnz_tile = std::max(max_nnz_tile, p1 - p0);
    }

    // ---- GPU allocations / CUDA setup ----
    double t_gpu_alloc_start = wtime();

    // device CSR tile buffers (double buffered)
    int *d_row_ptr_tile[2] = {nullptr, nullptr};
    int *d_col_idx_tile[2] = {nullptr, nullptr};
    T   *d_vals_tile[2]    = {nullptr, nullptr};

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&d_row_ptr_tile[b],
                              (size_t)(tile_rows + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col_idx_tile[b],
                              (size_t)max_nnz_tile * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_vals_tile[b],
                              (size_t)max_nnz_tile * sizeof(T)));
    }

    // pinned host CSR tile buffers (double buffered)
    int *h_row_ptr_tile[2] = {nullptr, nullptr};
    int *h_col_idx_tile[2] = {nullptr, nullptr};
    T   *h_vals_tile[2]    = {nullptr, nullptr};

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMallocHost(&h_row_ptr_tile[b],
                                  (size_t)(tile_rows + 1) * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_col_idx_tile[b],
                                  (size_t)max_nnz_tile * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&h_vals_tile[b],
                                  (size_t)max_nnz_tile * sizeof(T)));
    }

    // device dense tiles for X and C (C is also double-buffered now)
    int K_tile_max = std::min(tile_K, K);

    T *d_X_tile = nullptr;           // n x K_tile_max
    T *d_C_tile[2] = {nullptr, nullptr}; // tile_rows x K_tile_max per buffer

    CUDA_CHECK(cudaMalloc(&d_X_tile,
                          (size_t)n * (size_t)K_tile_max * sizeof(T)));

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&d_C_tile[b],
                              (size_t)tile_rows * (size_t)K_tile_max * sizeof(T)));
    }

    // cuSPARSE + streams
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));


    cudaStream_t stream_h2d, stream_compute, stream_d2h;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_h2d, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_d2h, cudaStreamNonBlocking));
    CUSPARSE_CHECK(cusparseSetStream(handle, stream_compute));

    cudaDataType dt = getCudaType<T>();

    // SpMM buffer size query with worst-case tile sizes
    int m_tile_max = std::min(tile_rows, m);

    cusparseSpMatDescr_t matA_tmp;
    cusparseDnMatDescr_t matX_tmp, matC_tmp;

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
        d_C_tile[0],
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

    // Events for tile dependencies
    cudaEvent_t ev_X_ready;
    CUDA_CHECK(cudaEventCreate(&ev_X_ready));

    cudaEvent_t ev_A_h2d_done[2];
    cudaEvent_t ev_C_compute_done[2];
    cudaEvent_t ev_C_d2h_done[2];

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaEventCreate(&ev_A_h2d_done[b]));
        CUDA_CHECK(cudaEventCreate(&ev_C_compute_done[b]));
        CUDA_CHECK(cudaEventCreate(&ev_C_d2h_done[b]));
        // Mark C_d2h_done as "already done" initially
        CUDA_CHECK(cudaEventRecord(ev_C_d2h_done[b], 0));
    }

    // Vectors of events for timing individual segments
    std::vector<cudaEvent_t> h2d_start_events, h2d_stop_events;
    std::vector<cudaEvent_t> d2h_start_events, d2h_stop_events;
    std::vector<cudaEvent_t> spmm_start_events, spmm_stop_events;

    Tm.t_gpu_alloc = 1000.0 * (wtime() - t_gpu_alloc_start);

    // ---- main K-tile loop ----
    for (int tk = 0; tk < num_k_tiles; ++tk) {
        int k0 = tk * tile_K;
        int this_K = std::min(tile_K, K - k0);

        // 1) H2D of X tile (n x this_K) on stream_h2d
        {
            cudaEvent_t ev_start, ev_stop;
            CUDA_CHECK(cudaEventCreate(&ev_start));
            CUDA_CHECK(cudaEventCreate(&ev_stop));
            CUDA_CHECK(cudaEventRecord(ev_start, stream_h2d));

            for (int kk = 0; kk < this_K; ++kk) {
                const T *h_src = h_X + (size_t)(k0 + kk) * n;
                T       *d_dst = d_X_tile   + (size_t)kk * n;
                CUDA_CHECK(cudaMemcpyAsync(d_dst, h_src,
                                           (size_t)n * sizeof(T),
                                           cudaMemcpyHostToDevice,
                                           stream_h2d));
            }

            CUDA_CHECK(cudaEventRecord(ev_stop, stream_h2d));
            h2d_start_events.push_back(ev_start);
            h2d_stop_events.push_back(ev_stop);
        }

        CUDA_CHECK(cudaEventRecord(ev_X_ready, stream_h2d));

        int cur  = 0;
        int next = 1;

        // 2) Preload first row tile into buffer 'cur'
        if (num_row_tiles > 0) {
            int tr = 0;
            int r0 = tr * tile_rows;
            int r1 = std::min(m, r0 + tile_rows);
            int m_tile = r1 - r0;
            int p0 = csr.row_ptr[r0];
            int p1 = csr.row_ptr[r1];
            int nnz_tile = p1 - p0;

            // Wait until previous D2H on this buffer is done
            CUDA_CHECK(cudaStreamWaitEvent(stream_h2d, ev_C_d2h_done[cur], 0));

            h_row_ptr_tile[cur][0] = 0;
            for (int i = 0; i < m_tile; ++i) {
                int gr = r0 + i;
                int row_nnz = csr.row_ptr[gr + 1] - csr.row_ptr[gr];
                h_row_ptr_tile[cur][i + 1] = h_row_ptr_tile[cur][i] + row_nnz;
            }
            std::memcpy(h_col_idx_tile[cur],
                        csr.col_idx.data() + p0,
                        (size_t)nnz_tile * sizeof(int));
            std::memcpy(h_vals_tile[cur],
                        csr.vals.data() + p0,
                        (size_t)nnz_tile * sizeof(T));

            cudaEvent_t ev_start, ev_stop;
            CUDA_CHECK(cudaEventCreate(&ev_start));
            CUDA_CHECK(cudaEventCreate(&ev_stop));
            CUDA_CHECK(cudaEventRecord(ev_start, stream_h2d));

            CUDA_CHECK(cudaMemcpyAsync(d_row_ptr_tile[cur],
                                       h_row_ptr_tile[cur],
                                       (size_t)(m_tile + 1) * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream_h2d));
            CUDA_CHECK(cudaMemcpyAsync(d_col_idx_tile[cur],
                                       h_col_idx_tile[cur],
                                       (size_t)nnz_tile * sizeof(int),
                                       cudaMemcpyHostToDevice,
                                       stream_h2d));
            CUDA_CHECK(cudaMemcpyAsync(d_vals_tile[cur],
                                       h_vals_tile[cur],
                                       (size_t)nnz_tile * sizeof(T),
                                       cudaMemcpyHostToDevice,
                                       stream_h2d));

            CUDA_CHECK(cudaEventRecord(ev_stop, stream_h2d));
            h2d_start_events.push_back(ev_start);
            h2d_stop_events.push_back(ev_stop);

            CUDA_CHECK(cudaEventRecord(ev_A_h2d_done[cur], stream_h2d));
        }

        // 3) Iterate over row tiles
        for (int tr = 0; tr < num_row_tiles; ++tr) {
            int r0 = tr * tile_rows;
            int r1 = std::min(m, r0 + tile_rows);
            int m_tile = r1 - r0;
            int p0 = csr.row_ptr[r0];
            int p1 = csr.row_ptr[r1];
            int nnz_tile = p1 - p0;

            next = 1 - cur;

            // 3a) Preload next row tile into 'next' (if exists)
            if (tr + 1 < num_row_tiles) {
                int r0n = (tr + 1) * tile_rows;
                int r1n = std::min(m, r0n + tile_rows);
                int m_tile_n = r1n - r0n;
                int p0n = csr.row_ptr[r0n];
                int p1n = csr.row_ptr[r1n];
                int nnz_tile_n = p1n - p0n;

                // Wait until previous D2H for 'next' buffer is done
                CUDA_CHECK(cudaStreamWaitEvent(stream_h2d, ev_C_d2h_done[next], 0));

                h_row_ptr_tile[next][0] = 0;
                for (int i = 0; i < m_tile_n; ++i) {
                    int gr = r0n + i;
                    int row_nnz = csr.row_ptr[gr + 1] - csr.row_ptr[gr];
                    h_row_ptr_tile[next][i + 1] =
                        h_row_ptr_tile[next][i] + row_nnz;
                }

                std::memcpy(h_col_idx_tile[next],
                            csr.col_idx.data() + p0n,
                            (size_t)nnz_tile_n * sizeof(int));
                std::memcpy(h_vals_tile[next],
                            csr.vals.data() + p0n,
                            (size_t)nnz_tile_n * sizeof(T));

                cudaEvent_t ev_start, ev_stop;
                CUDA_CHECK(cudaEventCreate(&ev_start));
                CUDA_CHECK(cudaEventCreate(&ev_stop));
                CUDA_CHECK(cudaEventRecord(ev_start, stream_h2d));

                CUDA_CHECK(cudaMemcpyAsync(d_row_ptr_tile[next],
                                           h_row_ptr_tile[next],
                                           (size_t)(m_tile_n + 1) * sizeof(int),
                                           cudaMemcpyHostToDevice,
                                           stream_h2d));
                CUDA_CHECK(cudaMemcpyAsync(d_col_idx_tile[next],
                                           h_col_idx_tile[next],
                                           (size_t)nnz_tile_n * sizeof(int),
                                           cudaMemcpyHostToDevice,
                                           stream_h2d));
                CUDA_CHECK(cudaMemcpyAsync(d_vals_tile[next],
                                           h_vals_tile[next],
                                           (size_t)nnz_tile_n * sizeof(T),
                                           cudaMemcpyHostToDevice,
                                           stream_h2d));

                CUDA_CHECK(cudaEventRecord(ev_stop, stream_h2d));
                h2d_start_events.push_back(ev_start);
                h2d_stop_events .push_back(ev_stop);

                CUDA_CHECK(cudaEventRecord(ev_A_h2d_done[next], stream_h2d));
            }

            // 3b) Compute current tile on 'cur' buffer
            CUDA_CHECK(cudaStreamWaitEvent(stream_compute, ev_X_ready,       0));
            CUDA_CHECK(cudaStreamWaitEvent(stream_compute, ev_A_h2d_done[cur], 0));

            // zero C tile for this tile
            CUDA_CHECK(cudaMemsetAsync(d_C_tile[cur], 0,
                                       (size_t)m_tile * (size_t)this_K * sizeof(T),
                                       stream_compute));

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
                d_C_tile[cur],
                dt,
                CUSPARSE_ORDER_COL));

            // SpMM timing with events (no sync here)
            cudaEvent_t ev_spmm_start, ev_spmm_stop;
            CUDA_CHECK(cudaEventCreate(&ev_spmm_start));
            CUDA_CHECK(cudaEventCreate(&ev_spmm_stop));
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
            spmm_start_events.push_back(ev_spmm_start);
            spmm_stop_events.push_back(ev_spmm_stop);

            // mark C_tile[cur] compute done
            CUDA_CHECK(cudaEventRecord(ev_C_compute_done[cur], stream_compute));

            // 3c) D2H C tile on stream_d2h after compute is done
            CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, ev_C_compute_done[cur], 0));

            cudaEvent_t ev_d2h_start, ev_d2h_stop;
            CUDA_CHECK(cudaEventCreate(&ev_d2h_start));
            CUDA_CHECK(cudaEventCreate(&ev_d2h_stop));
            CUDA_CHECK(cudaEventRecord(ev_d2h_start, stream_d2h));

            for (int kk = 0; kk < this_K; ++kk) {
                const T *d_src = d_C_tile[cur] + (size_t)kk * (size_t)tile_rows;
                T       *h_dst =
                    h_C + (size_t)(k0 + kk) * (size_t)m + (size_t)r0;
                CUDA_CHECK(cudaMemcpyAsync(h_dst,
                                           d_src,
                                           (size_t)m_tile * sizeof(T),
                                           cudaMemcpyDeviceToHost,
                                           stream_d2h));
            }

            CUDA_CHECK(cudaEventRecord(ev_d2h_stop, stream_d2h));
            d2h_start_events.push_back(ev_d2h_start);
            d2h_stop_events .push_back(ev_d2h_stop);

            CUDA_CHECK(cudaEventRecord(ev_C_d2h_done[cur], stream_d2h));

            // cleanup descriptors
            CUSPARSE_CHECK(cusparseDestroySpMat(matA_tile));
            CUSPARSE_CHECK(cusparseDestroyDnMat(matX_tile));
            CUSPARSE_CHECK(cusparseDestroyDnMat(matC_tile));

            cur = next;
        }
    }

    // Final syncs (already existed logically; do not break overlap inside the loop)
    CUDA_CHECK(cudaStreamSynchronize(stream_h2d));
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_d2h));

    // Sum up timings from event pairs
    float ms = 0.0f;

    for (size_t i = 0; i < h2d_start_events.size(); ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms, h2d_start_events[i], h2d_stop_events[i]));
        Tm.t_h2d_ms += ms;
    }
    for (size_t i = 0; i < d2h_start_events.size(); ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms, d2h_start_events[i], d2h_stop_events[i]));
        Tm.t_d2h_ms += ms;
    }
    for (size_t i = 0; i < spmm_start_events.size(); ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms, spmm_start_events[i], spmm_stop_events[i]));
        Tm.t_spmm_ms += ms;
    }

    // cleanup timing events
    auto destroy_event_vec_pair = [](std::vector<cudaEvent_t> &starts,
                                     std::vector<cudaEvent_t> &stops) {
        for (auto &e : starts) CUDA_CHECK(cudaEventDestroy(e));
        for (auto &e : stops)  CUDA_CHECK(cudaEventDestroy(e));
        starts.clear();
        stops.clear();
    };

    destroy_event_vec_pair(h2d_start_events, h2d_stop_events);
    destroy_event_vec_pair(d2h_start_events, d2h_stop_events);
    destroy_event_vec_pair(spmm_start_events, spmm_stop_events);

    // cleanup dependency events
    CUDA_CHECK(cudaEventDestroy(ev_X_ready));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaEventDestroy(ev_A_h2d_done[b]));
        CUDA_CHECK(cudaEventDestroy(ev_C_compute_done[b]));
        CUDA_CHECK(cudaEventDestroy(ev_C_d2h_done[b]));
    }

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFree(d_row_ptr_tile[b]));
        CUDA_CHECK(cudaFree(d_col_idx_tile[b]));
        CUDA_CHECK(cudaFree(d_vals_tile[b]));
        CUDA_CHECK(cudaFree(d_C_tile[b]));
        CUDA_CHECK(cudaFreeHost(h_row_ptr_tile[b]));
        CUDA_CHECK(cudaFreeHost(h_col_idx_tile[b]));
        CUDA_CHECK(cudaFreeHost(h_vals_tile[b]));
    }

    CUDA_CHECK(cudaFree(d_X_tile));
    CUDA_CHECK(cudaFree(dBuffer));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream_h2d));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaStreamDestroy(stream_d2h));

    CUDA_CHECK(cudaFreeHost(h_X));
    CUDA_CHECK(cudaFreeHost(h_C));
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

int main(int argc, char **argv)
{
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

    PrecMode mode = parse_precision(prec_str);

    double t0 = wtime();

    if (mode == PrecMode::FP64) {
        using T = double;
        CSR<T> csr = read_matrix_market_to_csr<T>(filename);
	double t_csr = 1e3*(wtime() - t0);
        SpmmTimings t = run_spmm_stream_3d_overlap<T>(csr, K, tile_rows, tile_K,
                                                      /*use_tf32_math=*/false);
        t0 = 1000.0 * (wtime() - t0);
        std::cout << "\nSPMM OVERLAP STREAM (FP64)\n";
        std::cout << "End2End=" << t0 << " ms\n";
        std::cout << "t_csr=" << t_csr << " ms\n";
	std::cout << "t_cpu_alloc=" << t.t_cpu_alloc << " ms\n";
        std::cout << "t_gpu_alloc=" << t.t_gpu_alloc << " ms\n";
        std::cout << "t_h2d_ms="    << t.t_h2d_ms    << " ms\n";
        std::cout << "t_d2h_ms="    << t.t_d2h_ms    << " ms\n";
        std::cout << "t_spmm_ms="   << t.t_spmm_ms   << " ms\n";
    } else {
        using T = float;
        CSR<T> csr = read_matrix_market_to_csr<T>(filename);
        double t_csr = 1e3*(wtime() - t0);

        bool use_tf32_math = (mode == PrecMode::FP32_TF32);
        SpmmTimings t = run_spmm_stream_3d_overlap<T>(csr, K, tile_rows, tile_K,
                                                      use_tf32_math);
        t0 = 1000.0 * (wtime() - t0);

        std::cout << "\nSPMM STREAMING OVERLAP ("
                  << (use_tf32_math ? "FP32+TF32" : "FP32") << ")\n";
        std::cout << "End2End=" << t0 << " ms\n";
	        std::cout << "t_csr=" << t_csr << " ms\n";
        std::cout << "t_cpu_alloc=" << t.t_cpu_alloc << " ms\n";
        std::cout << "t_gpu_alloc=" << t.t_gpu_alloc << " ms\n";
        std::cout << "t_h2d_ms="    << t.t_h2d_ms    << " ms\n";
        std::cout << "t_d2h_ms="    << t.t_d2h_ms    << " ms\n";
        std::cout << "t_spmm_ms="   << t.t_spmm_ms   << " ms\n";
    }

    return 0;
}

