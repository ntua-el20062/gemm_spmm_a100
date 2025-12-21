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
#include <cublas_v2.h>
#include <cstdio>
#include <unistd.h>
#include <math.h>
#include <limits>
#include "utils.h"

double wtime()
{
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
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

// Simple check: A*I == A (dense, column-major)
bool check_A_times_I_equals_A_dense(const double* A, int m, int n,
                                    const double* C, int ldc, double tol)
{
    bool ok = true;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            double a = A[(size_t)i + (size_t)j * (size_t)m];
            double c = C[(size_t)i + (size_t)j * (size_t)ldc];
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


SpmmTimings run_gemm(int M, int N, int K, int tile_K, int buff_size) {
    SpmmTimings Tm{};

    print_memory();
    int m = M;
    int n = N;

    if (m == 0 || n == 0 || K == 0) return Tm;

    //CPU alloc & init
    double t_cpu_start = wtime();

    double *h_A = nullptr;
    double *h_X = nullptr;
    double *h_C = nullptr;

    CUDA_CHECK(cudaMallocHost(&h_A, sizeof(double) * (size_t)m * n));
    CUDA_CHECK(cudaMallocHost(&h_X, sizeof(double) * (size_t)n * K));
    CUDA_CHECK(cudaMallocHost(&h_C, sizeof(double) * (size_t)m * K));

    //random A
    std::srand(1234);
    for (size_t i = 0; i < (size_t)m * n; ++i) {
        double r = (double)std::rand() / (double)RAND_MAX;
        h_A[i] = r;
    }

    //X: for small matrices use X = I for self-check; else ones
    if (m < 110000) {
        for (int j = 0; j < K; ++j) {
            for (int i = 0; i < n; ++i) {
                h_X[(size_t)i + (size_t)j * (size_t)n] = (i == j) ? 1.0 : 0.0;
            }
        }
    } else {
        for (size_t i = 0; i < (size_t)n * K; ++i) {
            h_X[i] = 1.0;
        }
    }

    Tm.t_cpu_alloc = 1e3 * (wtime() - t_cpu_start);
    //GPU alloc
    double t_gpu_start = wtime();
    double *d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, sizeof(double) * (size_t)m * n));
    //H2D A copy (once)
    double t1 = wtime();
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(double) * (size_t)m * n, cudaMemcpyHostToDevice));
    Tm.t_h2d_ms += 1e3 * (wtime() - t1);

        int K_tile_max = tile_K;
    int num_tiles  = (K + tile_K - 1) / tile_K;

    const int NUM_BUF = buff_size;
    int buf_count = std::min(NUM_BUF, num_tiles > 0 ? num_tiles : 1); 
    std::vector<double*> d_X_tile(buf_count, nullptr);
    std::vector<double*> d_C_tile(buf_count, nullptr);

    print_memory();


    for (int b = 0; b < buf_count; ++b) {
        CUDA_CHECK(cudaMalloc(&d_X_tile[b], sizeof(double) * (size_t)n * K_tile_max));
        CUDA_CHECK(cudaMalloc(&d_C_tile[b],sizeof(double) * (size_t)m * K_tile_max));
    }

    //cuBLAS + streams
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    cudaStream_t stream_h2d, stream_compute, stream_d2h;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_h2d,     cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_d2h,     cudaStreamNonBlocking));

    CUBLAS_CHECK(cublasSetStream(handle, stream_compute));

    Tm.t_gpu_alloc = 1e3 * (wtime() - t_gpu_start);

    //events for timing and sync
    std::vector<cudaEvent_t> h2d_start, h2d_stop;
    std::vector<cudaEvent_t> gemm_start, gemm_stop;
    std::vector<cudaEvent_t> d2h_start, d2h_stop;

    std::vector<cudaEvent_t> ev_buf_free(buf_count);
    std::vector<cudaEvent_t> ev_h2d_done(buf_count);
    std::vector<cudaEvent_t> ev_compute_done(buf_count);

    for (int b = 0; b < buf_count; ++b) {
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_buf_free[b],     cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_h2d_done[b],     cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_compute_done[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(ev_buf_free[b], 0));  // initially free
    }

    //preload tile 0
    if (num_tiles > 0) {
        int k0 = 0;
        int this_K0 = std::min(tile_K, K - k0);
        int buf0 = 0 % buf_count;

        //CUDA_CHECK(cudaStreamWaitEvent(stream_h2d, ev_buf_free[buf0], 0));
        cudaEvent_t ev_s, ev_e;
        CUDA_CHECK(cudaEventCreate(&ev_s));
        CUDA_CHECK(cudaEventCreate(&ev_e));
        CUDA_CHECK(cudaEventRecord(ev_s, stream_h2d));
        CUDA_CHECK(cudaMemcpyAsync(d_X_tile[buf0],h_X + (size_t)k0 * n,sizeof(double) * (size_t)n * this_K0,cudaMemcpyHostToDevice,stream_h2d));
        CUDA_CHECK(cudaEventRecord(ev_e, stream_h2d));
        h2d_start.push_back(ev_s);
        h2d_stop.push_back(ev_e);

        CUDA_CHECK(cudaEventRecord(ev_h2d_done[buf0], stream_h2d));
    }

    //main tiled loop
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

            CUDA_CHECK(cudaMemcpyAsync(h_C + (size_t)prev_k0 * m, d_C_tile[prev_buf], sizeof(double) * (size_t)m * prev_K,cudaMemcpyDeviceToHost,stream_d2h));
            CUDA_CHECK(cudaEventRecord(ev_e, stream_d2h));
            d2h_start.push_back(ev_s);
            d2h_stop.push_back(ev_e);

            CUDA_CHECK(cudaEventRecord(ev_buf_free[prev_buf], stream_d2h));
        }

        //gemm(t): c_tile=A*x_tile
        {
            CUDA_CHECK(cudaStreamWaitEvent(stream_compute,
                                           ev_h2d_done[buf], 0));

            cudaEvent_t ev_s, ev_e;
            CUDA_CHECK(cudaEventCreate(&ev_s));
            CUDA_CHECK(cudaEventCreate(&ev_e));
            CUDA_CHECK(cudaEventRecord(ev_s, stream_compute));

            CUDA_CHECK(cudaMemsetAsync(d_C_tile[buf], 0,
                                       sizeof(double) * (size_t)m * this_K,
                                       stream_compute));

            const double alpha = 1.0;
            const double beta  = 0.0;

            CUBLAS_CHECK(cublasDgemm(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     m, this_K, n,
                                     &alpha,
                                     d_A, m,
                                     d_X_tile[buf], n,
                                     &beta,
                                     d_C_tile[buf], m));

            CUDA_CHECK(cudaEventRecord(ev_e, stream_compute));
            gemm_start.push_back(ev_s);
            gemm_stop.push_back(ev_e);

            CUDA_CHECK(cudaEventRecord(ev_compute_done[buf], stream_compute));
        }

        //h2d(t+1)
        if (t + 1 < num_tiles) {
            int next      = t + 1;
            int next_buf  = next % buf_count;
            int next_k0   = next * tile_K;
            int next_K    = std::min(tile_K, K - next_k0);

            CUDA_CHECK(cudaStreamWaitEvent(stream_h2d,ev_buf_free[next_buf], 0));
            cudaEvent_t ev_s, ev_e;
            CUDA_CHECK(cudaEventCreate(&ev_s));
            CUDA_CHECK(cudaEventCreate(&ev_e));
            CUDA_CHECK(cudaEventRecord(ev_s, stream_h2d));
            CUDA_CHECK(cudaMemcpyAsync(d_X_tile[next_buf], h_X + (size_t)next_k0 * n, sizeof(double) * (size_t)n * next_K, cudaMemcpyHostToDevice, stream_h2d));
            CUDA_CHECK(cudaEventRecord(ev_e, stream_h2d));
            h2d_start.push_back(ev_s);
            h2d_stop.push_back(ev_e);

            CUDA_CHECK(cudaEventRecord(ev_h2d_done[next_buf], stream_h2d));
        }
    }

    //d2h last tile
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

        CUDA_CHECK(cudaMemcpyAsync(h_C + (size_t)last_k0 * m, d_C_tile[last_buf], sizeof(double) * (size_t)m * last_K,cudaMemcpyDeviceToHost, stream_d2h));

        CUDA_CHECK(cudaEventRecord(ev_e, stream_d2h));
        d2h_start.push_back(ev_s);
        d2h_stop.push_back(ev_e);

        CUDA_CHECK(cudaEventRecord(ev_buf_free[last_buf], stream_d2h));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_h2d));
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_d2h));

    float ms;
    for (size_t i = 0; i < h2d_start.size(); ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms, h2d_start[i], h2d_stop[i]));
        Tm.t_h2d_ms += ms;
    }
    for (size_t i = 0; i < gemm_start.size(); ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms, gemm_start[i], gemm_stop[i]));
        Tm.t_spmm_ms += ms;  // reuse field name
    }
    for (size_t i = 0; i < d2h_start.size(); ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms, d2h_start[i], d2h_stop[i]));
        Tm.t_d2h_ms += ms;
    }

    //check for small matrices (A * I == A)
    if (m < 110000) {
        double tol = 1e-10;
        bool same  = check_A_times_I_equals_A_dense(h_A, m, n, h_C, m, tol);
        if (same) {
            std::cout << "Dense self-check: PASSED (A*I == A numerically)\n";
        } else {
            std::cout << "Dense self-check: FAILED (numerical mismatch)\n";
        }
    }

    for (auto &e : h2d_start) CUDA_CHECK(cudaEventDestroy(e));
    for (auto &e : h2d_stop)  CUDA_CHECK(cudaEventDestroy(e));
    for (auto &e : gemm_start) CUDA_CHECK(cudaEventDestroy(e));
    for (auto &e : gemm_stop)  CUDA_CHECK(cudaEventDestroy(e));
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

    CUDA_CHECK(cudaFree(d_A));

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaStreamDestroy(stream_h2d));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaStreamDestroy(stream_d2h));

    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_X));
    CUDA_CHECK(cudaFreeHost(h_C));

    return Tm;
}

int compute_tile_K(int m, int n, size_t free_bytes, int K) {
    const size_t mm=static_cast<size_t>(m);
    const size_t nn=static_cast<size_t>(n);
    const size_t A_bytes = mm*nn*sizeof(double);          //A(m x n)
    const double safety_fraction = 0.9;
    size_t usable_free = static_cast<size_t>(free_bytes * safety_fraction);
    printf("usable free= %zu\n", usable_free);
    if (A_bytes <= usable_free) {
    printf("A = %zu\n", A_bytes);
    const size_t bytes_per_K = 2*(mm+nn)*sizeof(double); //X_tile+C_tile
    printf("bytes per k = %zu\n", bytes_per_K);
    const size_t available_for_tiles = usable_free - A_bytes;
    printf("available for tiles= %zu\n", available_for_tiles);

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

    return static_cast<int>(max_tile_K);
    } else {
	    printf("A doesn't fit into HBM");
	    return 0;
    }
}

int main(int argc, char **argv) {
    int m = 20000;   // rows of X and C
    int k = 20000;   // cols of X, rows of B
    int n = 20000;   // cols of B and C

    int num_steps  = 10; // number of B_i
    int buff_size  = 2; // cyclic buffer size

    if (argc >= 2) {
      int tmp1 = std::atoi(argv[1]);
      if (tmp1 > 0) {
        n = m = k = tmp1;
      }
    }
    if (argc >= 3) {
      int tmp = std::atoi(argv[2]);
      if (tmp > 0) {
        buff_size = tmp;
      }
    }
    if (argc >= 4) {
      int s = std::atoi(argv[3]);
      if (s > 0) {
        num_steps = s;
      }
    }
    if (buff_size > num_steps)
        buff_size = num_steps;
    if (buff_size <= 0)
        buff_size = 1;

    std::printf("Using N=K=M= %d, buff_size = %d (num_steps = %d)\n", n, buff_size, num_steps);

    size_t free_bytes;
    size_t total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);

    double t0 = wtime();
    int tile_K = compute_tile_K(m, n, free_bytes, k);
    
    //std::cout << "tile_K = " << tile_K << std::endl;
    if(tile_K != 0 ) {
    std::cout << "tile_K = " << tile_K << std::endl;
    SpmmTimings t = run_gemm(m, n, k, tile_K , buff_size);
    t0 = 1000.0 * (wtime() - t0);
    std::cout << "\nGEMM K-TILED\n";
    std::cout << "End2End="      << t0            << " ms\n";
    std::cout << "t_cpu_alloc="  << t.t_cpu_alloc << " ms\n";
    std::cout << "t_gpu_alloc="  << t.t_gpu_alloc << " ms\n";
    std::cout << "t_h2d_ms="     << t.t_h2d_ms    << " ms\n";
    std::cout << "t_d2h_ms="     << t.t_d2h_ms    << " ms\n";
    std::cout << "t_gemm_ms="    << t.t_spmm_ms   << " ms\n";

    return 0;
    }
    else {
	    return 0;
    }
}

