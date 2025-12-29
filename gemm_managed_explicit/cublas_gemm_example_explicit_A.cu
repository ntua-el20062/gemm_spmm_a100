#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <ctime>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <atomic>
#include <thread>
#include <chrono>
#include "cublas_utils.h"
#include <random>
#include <iostream>

#include <unistd.h>   // sysconf for RSS

double wtime(void)
{
    double now_time;
    struct timeval  etstart;
    struct timezone tzp;

    if (gettimeofday(&etstart, &tzp) == -1)
        perror("Error: calling gettimeofday() not successful.\n");

    now_time = ((double)etstart.tv_sec) +              // in seconds
               ((double)etstart.tv_usec) / 1000000.0;  // in microseconds
    return now_time;
}

static double rss_mb_linux() {
    long rss_pages = 0, dummy = 0;
    FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return -1.0;
    if (std::fscanf(f, "%ld %ld", &dummy, &rss_pages) != 2) {
        std::fclose(f);
        return -1.0;
    }
    std::fclose(f);
    const long page_size = sysconf(_SC_PAGESIZE);
    return (double)rss_pages * (double)page_size / (1024.0 * 1024.0);
}

static double gpu_used_mb() {
    size_t freeB = 0, totalB = 0;
    cudaError_t st = cudaMemGetInfo(&freeB, &totalB);
    if (st != cudaSuccess) return -1.0;
    return (double)(totalB - freeB) / (1024.0 * 1024.0);
}

struct MemLogger {
    FILE* f = nullptr;
    double t0 = 0.0;

    void open(const char* path, double t_start) {
        t0 = t_start;
        f = std::fopen(path, "w");
        if (f) {
            std::fprintf(f, "t_s,sys_rss_mb,gpu_used_mb\n");
            std::fflush(f);
        }
    }

    void sample(double t_now) {
        if (!f) return;
        std::fprintf(
            f, "%.6f,%.3f,%.3f\n",
            t_now - t0,
            rss_mb_linux(),
            gpu_used_mb()
        );
        std::fflush(f);
    }

    void close() {
        if (f) std::fclose(f);
        f = nullptr;
    }
};

using data_type = double;

int main(int argc, char *argv[]) {
    printf("EXPLICIT MEMCPY VERSION (no cudaMallocManaged)\n");

    cublasHandle_t cublasH = NULL;
    cudaStream_t   stream  = NULL;

    int m  = 2;
    int n  = 2;
    int k  = 2;
    int iters = 1250;
    int perc = 25;

    if (argc == 5 || argc == 6) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        k = std::atoi(argv[3]);
        if (argc == 5) { perc = std::atoi(argv[4]); }
        if (argc == 6) {
            iters = std::atoi(argv[5]);
            if (iters < 1) iters = 1;
        }
    } else {
        std::printf("Usage: %s M N K [PERC] [ITERS]\n", argv[0]);
        std::printf("No valid sizes given, using default M=N=K=2, ITERS=1\n");
    }

    size_t lda = m;
    size_t ldb = k;
    size_t ldc = m;

    const size_t sizeA = (size_t)m * (size_t)k;
    const size_t sizeB = (size_t)k * (size_t)n;
    const size_t sizeC = (size_t)m * (size_t)n;

    const size_t bytesA = sizeA * sizeof(data_type);
    const size_t bytesB = sizeB * sizeof(data_type);
    const size_t bytesC = sizeC * sizeof(data_type);

    std::printf("Running GEMM with m=%d, n=%d, k=%d, iters=%d\n", m, n, k, iters);

    double t_end2end = wtime();

    MemLogger memlog;
    memlog.open("memlog_explicit_A.csv", t_end2end);
    memlog.sample(wtime());

    data_type alpha = 1.0;
    data_type beta  = 0.0;

    // host buffers (pinned for fast async copies)
    data_type *h_A = nullptr;
    data_type *h_B = nullptr;
    data_type *h_C = nullptr;

    // device buffers
    data_type *d_A = nullptr;
    data_type *d_B = nullptr;
    data_type *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    memlog.sample(wtime()); // after CUDA/cuBLAS init

   
    CUDA_CHECK(cudaMallocHost((void**)&h_A, bytesA));
    CUDA_CHECK(cudaMallocHost((void**)&h_B, bytesB));
    CUDA_CHECK(cudaMallocHost((void**)&h_C, bytesC));
    memlog.sample(wtime()); // after host alloc

    
    double t_cuda_alloc = wtime();
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytesA));
    memlog.sample(wtime());
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytesB));
    memlog.sample(wtime());
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytesC));
    t_cuda_alloc = 1e3*(wtime() - t_cuda_alloc);

    memlog.sample(wtime()); // after device alloc

    // init data on host
    std::srand((unsigned)std::time(nullptr));
    for (size_t i = 0; i < sizeA; ++i) h_A[i] = (data_type)std::rand() / (data_type)RAND_MAX;
    memlog.sample(wtime());
    for (size_t i = 0; i < sizeB; ++i) h_B[i] = (data_type)std::rand() / (data_type)RAND_MAX;
    memlog.sample(wtime());
    for (size_t i = 0; i < sizeC; ++i) h_C[i] = 0.0;
    memlog.sample(wtime());

    double t_h2d_total = 0.0; // milliseconds
    double t_d2h_total = 0.0; // milliseconds

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    float ms = 0.0f;

    // Initial H2D copies (timed)
    CUDA_CHECK(cudaEventRecord(ev0, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, bytesA, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(ev1, stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    t_h2d_total += (double)ms;
    memlog.sample(wtime());

    CUDA_CHECK(cudaEventRecord(ev0, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, bytesB, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(ev1, stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    t_h2d_total += (double)ms;
    memlog.sample(wtime());

    CUDA_CHECK(cudaEventRecord(ev0, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C, h_C, bytesC, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(ev1, stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    t_h2d_total += (double)ms;
    memlog.sample(wtime());
    iters = 1250;
    const int spike_period = 250;

    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> dist(0, sizeA - 1);

    std::cout << "Total A size (elements) = " << sizeA << std::endl;
    std::cout << "iters=" << iters << ", spikes at 250/500/750/1000 (25/50/75/100%)\n";

    double t_compute_total = 0.0;

    for (int it = 0; it < iters; ++it) {
        memlog.sample(wtime());

        //spike simulates CPU reading A by copying a fraction of A back to host, touching it,
        //then copying that fraction back to device.
        if (it > 0 && (it % spike_period == 0) && it <= 1000) {
            int block = it / spike_period;   // 1,2,3,4
            double frac = 0.25 * block;      // 0.25, 0.50, 0.75, 1.00

            size_t read_elems = static_cast<size_t>(frac * sizeA);
            if (read_elems > sizeA) read_elems = sizeA;
            const size_t read_bytes = read_elems * sizeof(data_type);

            CUDA_CHECK(cudaEventRecord(ev0, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_A, d_A, read_bytes, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaEventRecord(ev1, stream));
            CUDA_CHECK(cudaEventSynchronize(ev1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            t_d2h_total += (double)ms;
            memlog.sample(wtime());

            volatile double sink = 0.0;
            for (size_t i = 0; i < read_elems; ++i) {
                sink += h_A[dist(rng) % read_elems];
                if (i % 100 == 0) memlog.sample(wtime());
            }
            (void)sink;

     
            CUDA_CHECK(cudaEventRecord(ev0, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, read_bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaEventRecord(ev1, stream));
            CUDA_CHECK(cudaEventSynchronize(ev1));
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            t_h2d_total += (double)ms;
            memlog.sample(wtime());
        }

        CUDA_CHECK(cudaEventRecord(ev0, stream));
        memlog.sample(wtime());

        CUBLAS_CHECK(cublasDgemm(
            cublasH,
            transa, transb,
            m, n, k,
            &alpha,
            d_A, (int)lda,
            d_B, (int)ldb,
            &beta,
            d_C, (int)ldc
        ));

        memlog.sample(wtime());
        CUDA_CHECK(cudaEventRecord(ev1, stream));
        CUDA_CHECK(cudaEventSynchronize(ev1));

        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        std::cout << "iter: " << it << ", t_iter: " << ms << " ms" << std::endl;
        t_compute_total += (double)ms;
    }

    memlog.sample(wtime()); // after loop

    CUDA_CHECK(cudaEventRecord(ev0, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, bytesC, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(ev1, stream));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    t_d2h_total += (double)ms;
    memlog.sample(wtime()); // after final D2H
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    double t_compute_avg = t_compute_total / iters;
    t_end2end = wtime() - t_end2end;

    printf("t_cuda_alloc    = %.3f ms\n", t_cuda_alloc);
    printf("t_h2d_total     = %.3f ms\n", t_h2d_total);
    printf("t_d2h_total     = %.3f ms\n", t_d2h_total);
    printf("t_memcpy_total  = %.3f ms\n", t_h2d_total + t_d2h_total);
    printf("t_compute_total = %.3f ms\n", t_compute_total);
    printf("t_compute_avg   = %.3f ms\n", t_compute_avg);
    printf("t_end2end       = %.3f ms \n ================== \n", 1e3 * t_end2end);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    memlog.sample(wtime());
    memlog.close();

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

