#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <ctime>
#include <random>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <unistd.h>   // sysconf for RSS
#include <iostream>

#include "cublas_utils.h"

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
    printf("MALLOC VERSION\n");
    cublasHandle_t cublasH = NULL;
    cudaStream_t   stream  = NULL;
    int iters = 1000;
    int m   = 2;
    int n   = 2;
    int k   = 2;
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

    const size_t lda = m;
    const size_t ldb = k;
    const size_t ldc = m;

    const size_t sizeA = (size_t)m * (size_t)k;
    const size_t sizeB = (size_t)k * (size_t)n;
    const size_t sizeC = (size_t)m * (size_t)n;

    std::printf("Running GEMM with m=%d, n=%d, k=%d\n", m, n, k);

    double t_end2end = wtime();

    // ---- start mem logging
    MemLogger memlog;
    memlog.open("memlog_malloc_no_cpu.csv", t_end2end);
    memlog.sample(wtime()); // at start
    // ----------------------

    data_type alpha = 1.0;
    data_type beta  = 0.0;

    double t_alloc = wtime();

    data_type *A = (data_type*)malloc(sizeA * sizeof(data_type));
    memlog.sample(wtime());
    data_type *B = (data_type*)malloc(sizeB * sizeof(data_type));
    memlog.sample(wtime());
    data_type *C = (data_type*)malloc(sizeC * sizeof(data_type));
    memlog.sample(wtime());

    t_alloc = (wtime() - t_alloc)*1000;
    if (!A || !B || !C) {
        fprintf(stderr, "malloc failed\n");
        return EXIT_FAILURE;
    }

    memlog.sample(wtime()); // after malloc

    std::srand((unsigned)std::time(nullptr));

    for (int i = 0; i < (int)sizeA; ++i) {
        A[i] = (data_type)std::rand() / (data_type)RAND_MAX;
    }
    memlog.sample(wtime());
    for (int i = 0; i < (int)sizeB; ++i) {
        B[i] = (data_type)std::rand() / (data_type)RAND_MAX;
    }
    memlog.sample(wtime());
    for (int i = 0; i < (int)sizeC; ++i) {
        C[i] = 0.0;
    }
    memlog.sample(wtime());
    iters        = 20;
    const int spike_period = 250;

    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> dist(0, sizeA - 1);

    std::cout << "Total A size (elements) = " << sizeA << std::endl;

    double t_compute_total = 0.0;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    memlog.sample(wtime()); // after CUDA/cuBLAS init

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    for (int it = 0; it < iters; ++it) {
        memlog.sample(wtime());
        /*if (it > 0 && (it % spike_period == 0) && it <= 1000) {

            int block = it / spike_period;   // 1,2,3,4
            double frac = 0.25 * block;      // 0.25, 0.50, 0.75, 1.00

            size_t read_elems = static_cast<size_t>(frac * sizeA);
            if (read_elems > sizeA) read_elems = sizeA;
            volatile double sink = 0.0;
            for (size_t i = 0; i < read_elems; ++i) {
		//memlog.sample(wtime());
                sink += A[dist(rng)];
            }
            (void)sink;
        }*/
        //memlog.sample(wtime());
        CUDA_CHECK(cudaEventRecord(ev0, stream));
        memlog.sample(wtime());
	CUBLAS_CHECK(
            cublasDgemm(
                cublasH,
                transa, transb,
                m, n, k,
                &alpha,
                A, (int)lda,
                B, (int)ldb,
                &beta,
                C, (int)ldc
            )
        );
	memlog.sample(wtime());
        CUDA_CHECK(cudaEventRecord(ev1, stream));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        //memlog.sample(wtime());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        std::cout << "iter: " << it << ", t_iter: " << ms << " ms" << std::endl;
        t_compute_total += ms;
    }

    memlog.sample(wtime()); // after loop

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    t_end2end = wtime() - t_end2end;

    double t_compute_avg = t_compute_total / iters;

    printf("t_compute_total = %.3f ms\n", t_compute_total);
    printf("t_compute_avg   = %.3f ms\n", t_compute_avg);
    printf("t_alloc   = %.3f ms\n", t_alloc);
    printf("t_end2end       = %.3f ms \n ============================================= \n", 1e3 * t_end2end);

    free(A);
    free(B);
    free(C);

    memlog.sample(wtime()); // after free
    memlog.close();

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

