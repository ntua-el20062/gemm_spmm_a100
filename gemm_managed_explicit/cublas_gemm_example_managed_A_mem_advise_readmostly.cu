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
    printf("MANAGED VERSION\n");

    cublasHandle_t cublasH = NULL;
    cudaStream_t   stream  = NULL;

    int m  = 2;
    int n  = 2;
    int k  = 2;
    int iters = 500;
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
    memlog.open("memlog_managed_A_mem_advise_readmostly.csv", t_end2end);
    memlog.sample(wtime()); 

    data_type alpha = 1.0;
    data_type beta  = 0.0;

    data_type *d_A = nullptr;
    data_type *d_B = nullptr;
    data_type *d_C = nullptr;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    memlog.sample(wtime()); //after CUDA/cuBLAS init

    double t_cuda_alloc = wtime();
    CUDA_CHECK(cudaMallocManaged(&d_A, bytesA));
    int deviceId = 0;
    
    cudaMemAdvise(d_A, bytesA, cudaMemAdviseSetAccessedBy, (int)-1);
    cudaMemAdvise(d_A, bytesA, cudaMemAdviseSetReadMostly, deviceId);
    
    memlog.sample(wtime());
    CUDA_CHECK(cudaMallocManaged(&d_B, bytesB));
    memlog.sample(wtime());
    CUDA_CHECK(cudaMallocManaged(&d_C, bytesC));
    t_cuda_alloc = 1e3*(wtime() - t_cuda_alloc);

    memlog.sample(wtime()); //after UM alloc

    std::srand((unsigned)std::time(nullptr));
    for (size_t i = 0; i < sizeA; ++i) {
        d_A[i] = (data_type)std::rand() / (data_type)RAND_MAX;
    }
    memlog.sample(wtime());
    for (size_t i = 0; i < sizeB; ++i) {
        d_B[i] = (data_type)std::rand() / (data_type)RAND_MAX;
    }
    memlog.sample(wtime());
    for (size_t i = 0; i < sizeC; ++i) {
        d_C[i] = 0.0;
    }
    memlog.sample(wtime());
    iters        = 1250;
    const int spike_period = 250;

    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> dist(0, sizeA - 1);

    std::cout << "Total A size (elements) = " << sizeA << std::endl;
    std::cout << "iters=" << iters << ", spikes at 250/500/750/1000 (25/50/75/100%)\n";

    const size_t max_reads = sizeA;

    /*std::vector<size_t> rand_idx(max_reads);
    for (size_t i = 0; i < max_reads; ++i)
       rand_idx[i] = dist(rng);*/

    double t_compute_total = 0.0;

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    for (int it = 0; it < iters; ++it) {
	memlog.sample(wtime());
        if (it > 0 && (it % spike_period == 0) && it <= 1000) {

            int block = it / spike_period;   // 1,2,3,4
            double frac = 0.25 * block;      // 0.25, 0.50, 0.75, 1.00

            size_t read_elems = static_cast<size_t>(frac * sizeA);
            if (read_elems > sizeA) read_elems = sizeA;

            volatile double sink = 0.0;
            for (size_t i = 0; i < read_elems; ++i) {
                sink += d_A[dist(rng)]; // CPU touch => UM migration
                if(i % 100 == 0) { memlog.sample(wtime()); }
	    }
            (void)sink;

        }
	//memlog.sample(wtime());
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
        //memlog.sample(wtime());
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        std::cout << "iter: " << it << ", t_iter: " << ms << " ms" << std::endl;
        t_compute_total += ms;
    }

    memlog.sample(wtime()); // after loop

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    double t_compute_avg = t_compute_total / iters;

    t_end2end = wtime() - t_end2end;

    printf("t_cuda_alloc = %.3f ms\n", t_cuda_alloc);
    printf("t_compute_total = %.3f ms\n", t_compute_total);
    printf("t_compute_avg   = %.3f ms\n", t_compute_avg);
    printf("t_end2end       = %.3f ms \n ================== \n", 1e3 * t_end2end);

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    memlog.sample(wtime()); // after free
    memlog.close();

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

