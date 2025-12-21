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
};

template<typename T>
SpmmTimings run_spmm(const CSR<T> &csr,int K,int tile_K, bool use_tf32_math)
{
    SpmmTimings Tm{};

    int m = csr.m;
    int n = csr.n;
    int nnz = csr.nnz;

    if (m == 0 || n == 0 || K == 0) {
        std::cout << "Trivial dimensions, nothing to do.\n";
        return Tm;
    }

    //CPU alloc: dense x,c
    double t_cpu_alloc_start = wtime();

    //full dense X and C on host (column-major: n x K, m x K)
    T *h_X, *h_C;
    CUDA_CHECK(cudaMallocHost(&h_X, sizeof(T) * n * K));
    CUDA_CHECK(cudaMallocHost(&h_C, sizeof(T) * m * K));
    for (size_t t = 0; t < (size_t)n * (size_t)K; ++t) {
        h_X[t] = (T)1;
    }

    Tm.t_cpu_alloc = 1000.0 * (wtime() - t_cpu_alloc_start);

    //GPU alloc
    double t_gpu_alloc_start = wtime();

    //full CSR on device
    int *d_row_ptr = nullptr;
    int *d_col_idx = nullptr;
    T   *d_vals    = nullptr;

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (size_t)(m + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, (size_t)nnz      * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vals,    (size_t)nnz      * sizeof(T)));

    double t1 = wtime();
    CUDA_CHECK(cudaMemcpy(d_row_ptr, csr.row_ptr.data(), (size_t)(m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, csr.col_idx.data(), (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vals, csr.vals.data(), (size_t)nnz * sizeof(T), cudaMemcpyHostToDevice));
    Tm.t_h2d_ms += 1e3*(wtime() - t1);

    //dense tiles on device (only tiling over columns)
    int K_tile_max = std::min(tile_K, K);

    //buffers that will store the tiles for spmm computation
    T *d_X_tile = nullptr;  // n x K_tile_max
    T *d_C_tile = nullptr;  // m x K_tile_max

    CUDA_CHECK(cudaMalloc(&d_X_tile,(size_t)n * (size_t)K_tile_max * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_C_tile,(size_t)m * (size_t)K_tile_max * sizeof(T)));

    //cuSPARSE + streams
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

#if defined(CUSPARSE_MATH_TF32) && defined(CUSPARSE_DEFAULT_MATH)
    if (use_tf32_math && std::is_same<T,float>::value) {
        CUSPARSE_CHECK(cusparseSetMathMode(handle, CUSPARSE_MATH_TF32));
    } else {
        CUSPARSE_CHECK(cusparseSetMathMode(handle, CUSPARSE_DEFAULT_MATH));
    }
#else
    (void)use_tf32_math;
#endif

    cudaStream_t stream_h2d, stream_compute, stream_d2h;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_h2d, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_d2h, cudaStreamNonBlocking));
    CUSPARSE_CHECK(cusparseSetStream(handle, stream_compute));

    cudaDataType dt = getCudaType<T>();

    // SpMat for full CSR
    cusparseSpMatDescr_t matA;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &matA,
        m, n, nnz,
        d_row_ptr,
        d_col_idx,
        d_vals,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        dt));

    // Dummy dense descriptors for buffer size (max tile_K)
    cusparseDnMatDescr_t matX_tmp, matC_tmp;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matX_tmp,
        n, K_tile_max, n,          // rows=n, ld=n
        d_X_tile,
        dt,
        CUSPARSE_ORDER_COL));

    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matC_tmp,
        m, K_tile_max, m,          // rows=m, ld=m
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
        &alpha, matA, matX_tmp, &beta, matC_tmp,
        dt,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));

    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));

    CUSPARSE_CHECK(cusparseDestroyDnMat(matX_tmp));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matC_tmp));

    //events
    cudaEvent_t ev_X_ready;
    CUDA_CHECK(cudaEventCreate(&ev_X_ready));

    std::vector<cudaEvent_t> h2d_start_events, h2d_stop_events;
    std::vector<cudaEvent_t> d2h_start_events, d2h_stop_events;
    std::vector<cudaEvent_t> spmm_start_events, spmm_stop_events;

    //avoid overwriting d_C_tile before previous D2H finished
    cudaEvent_t ev_last_d2h_done;
    CUDA_CHECK(cudaEventCreate(&ev_last_d2h_done));
    CUDA_CHECK(cudaEventRecord(ev_last_d2h_done, 0)); // initially satisfied

    Tm.t_gpu_alloc = 1000.0 * (wtime() - t_gpu_alloc_start);

    int num_k_tiles = (K + tile_K - 1) / tile_K;

    //main tile loop
    for (int tk = 0; tk < num_k_tiles; ++tk) {
        int k0      = tk * tile_K;
        int this_K  = std::min(tile_K, K - k0);

        //wait until previous D2H of d_C_tile (if any) is done before reusing it
        CUDA_CHECK(cudaStreamWaitEvent(stream_compute, ev_last_d2h_done, 0));

        //spmm using full sparse A (csr) and curr x tile on compute stream
        CUDA_CHECK(cudaStreamWaitEvent(stream_compute, ev_X_ready, 0));

        //zero C tile
        CUDA_CHECK(cudaMemsetAsync(d_C_tile, 0, (size_t)m * (size_t)this_K * sizeof(T), stream_compute));

        cusparseDnMatDescr_t matX_tile, matC_tile;
        CUSPARSE_CHECK(cusparseCreateDnMat(
            &matX_tile,
            n, this_K, n,
            d_X_tile,
            dt,
            CUSPARSE_ORDER_COL));

        CUSPARSE_CHECK(cusparseCreateDnMat(
            &matC_tile,
            m, this_K, m,
            d_C_tile,
            dt,
            CUSPARSE_ORDER_COL));

        cudaEvent_t ev_spmm_start, ev_spmm_stop;
        CUDA_CHECK(cudaEventCreate(&ev_spmm_start));
        CUDA_CHECK(cudaEventCreate(&ev_spmm_stop));
        CUDA_CHECK(cudaEventRecord(ev_spmm_start, stream_compute));

        CUSPARSE_CHECK(cusparseSpMM(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matX_tile, &beta, matC_tile,
            dt,
            CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer));

        CUDA_CHECK(cudaEventRecord(ev_spmm_stop, stream_compute));
        spmm_start_events.push_back(ev_spmm_start);
        spmm_stop_events .push_back(ev_spmm_stop);

        //D2H
	{
        cudaEvent_t ev_d2h_start, ev_d2h_stop;
        CUDA_CHECK(cudaEventCreate(&ev_d2h_start));
        CUDA_CHECK(cudaEventCreate(&ev_d2h_stop));
        CUDA_CHECK(cudaEventRecord(ev_d2h_start, stream_d2h));

        CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, ev_spmm_stop, 0));

	const T *d_src_tile = d_C_tile;                         //tile starts at col 0 on device
	T *h_dst_tile = h_C + (size_t)k0 * m;      //first column of this tile on host
	size_t   elems      = (size_t)m * (size_t)this_K;

	CUDA_CHECK(cudaMemcpyAsync(h_dst_tile, d_src_tile,elems * sizeof(T), cudaMemcpyDeviceToHost, stream_d2h));

        CUDA_CHECK(cudaEventRecord(ev_d2h_stop, stream_d2h));
        d2h_start_events.push_back(ev_d2h_start);
        d2h_stop_events .push_back(ev_d2h_stop);
        }
        //H2D
        {
        cudaEvent_t ev_start, ev_stop;
        CUDA_CHECK(cudaEventCreate(&ev_start));
        CUDA_CHECK(cudaEventCreate(&ev_stop));
        CUDA_CHECK(cudaEventRecord(ev_start, stream_h2d));

        const T *h_src_tile = h_X + (size_t)k0 * n;   //first col of tile
        T *d_dst_tile = d_X_tile;                      //tile starts at col 0
        size_t tile_elems = (size_t)n * (size_t)this_K;

        CUDA_CHECK(cudaMemcpyAsync(d_dst_tile, h_src_tile, tile_elems * sizeof(T), cudaMemcpyHostToDevice, stream_h2d));

        CUDA_CHECK(cudaEventRecord(ev_stop, stream_h2d));
        h2d_start_events.push_back(ev_start);
        h2d_stop_events .push_back(ev_stop);
      	}
        //signal that X tile is ready
        CUDA_CHECK(cudaEventRecord(ev_X_ready, stream_h2d));



        CUSPARSE_CHECK(cusparseDestroyDnMat(matX_tile));
        CUSPARSE_CHECK(cusparseDestroyDnMat(matC_tile));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_h2d));
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_d2h));

    //sum up timings from event pairs
    float ms = 0.0f;

    for (size_t i = 0; i < h2d_start_events.size(); ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms,
                                        h2d_start_events[i],
                                        h2d_stop_events[i]));
        Tm.t_h2d_ms += ms;
    }
    for (size_t i = 0; i < d2h_start_events.size(); ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms,
                                        d2h_start_events[i],
                                        d2h_stop_events[i]));
        Tm.t_d2h_ms += ms;
    }
    for (size_t i = 0; i < spmm_start_events.size(); ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms,
                                        spmm_start_events[i],
                                        spmm_stop_events[i]));
        Tm.t_spmm_ms += ms;
    }

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

    CUDA_CHECK(cudaEventDestroy(ev_X_ready));
    //CUDA_CHECK(cudaEventDestroy(ev_last_d2h_done));

    CUDA_CHECK(cudaFree(d_X_tile));
    CUDA_CHECK(cudaFree(d_C_tile));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(dBuffer));
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream_h2d));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaStreamDestroy(stream_d2h));

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

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " matrix.mtx [K=32] [tile_rows=10000] [tile_K=64] [precision]\n"
                  << "  precision: fp32 | tf32 | fp64 | double | float64\n";
        return EXIT_FAILURE;
    }

    std::string filename = argv[1];
    int K         = (argc >= 3) ? std::atoi(argv[2]) : 32; //number of columns of dense matrix
    int tile_K = (argc >= 4) ? std::atoi(argv[3]) : 64; //we tile along columns of dense matrix, this is the size of the slice we take
    std::string prec_str = (argc >= 5) ? argv[4] : "fp32";

    PrecMode mode = parse_precision(prec_str);

    double t0 = wtime();

    if (mode == PrecMode::FP64) {
        using T = double;
        CSR<T> csr = read_matrix_market_to_csr<T>(filename);
        SpmmTimings t = run_spmm<T>(csr, K, tile_K, /*use_tf32_math=*/false);
        t0 = 1000.0 * (wtime() - t0);
        std::cout << "\nSPMM OVERLAP STREAM (FP64, CSR resident, K-tiled)\n";
        std::cout << "End2End="      << t0            << " ms\n";
        std::cout << "t_cpu_alloc="  << t.t_cpu_alloc << " ms\n";
        std::cout << "t_gpu_alloc="  << t.t_gpu_alloc << " ms\n";
        std::cout << "t_h2d_ms="     << t.t_h2d_ms    << " ms\n";
        std::cout << "t_d2h_ms="     << t.t_d2h_ms    << " ms\n";
        std::cout << "t_spmm_ms="    << t.t_spmm_ms   << " ms\n";
    } else {
        using T = float;
        CSR<T> csr = read_matrix_market_to_csr<T>(filename);
        bool use_tf32_math = (mode == PrecMode::FP32_TF32);
        SpmmTimings t = run_spmm<T>(csr, K,tile_K, use_tf32_math);
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
    }

    return 0;
}

