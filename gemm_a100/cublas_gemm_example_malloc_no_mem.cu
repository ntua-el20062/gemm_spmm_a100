#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <ctime>
#include <random>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

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
        std::printf("Usage: %s M N K [ITERS]\n", argv[0]);
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

    
    data_type alpha = 1.0;
    data_type beta  = 0.0;

    double t_alloc = wtime();
   
    data_type *A = (data_type*)malloc(sizeA * sizeof(data_type));
    data_type *B = (data_type*)malloc(sizeB * sizeof(data_type));
    data_type *C = (data_type*)malloc(sizeC * sizeof(data_type));

    t_alloc = (wtime() - t_alloc)*1000;
    if (!A || !B || !C) {
        fprintf(stderr, "malloc failed\n");
        return EXIT_FAILURE;
    }

   std::srand((unsigned)std::time(nullptr));

    for (int i = 0; i < sizeA; ++i) {
        A[i] = (data_type)std::rand() / (data_type)RAND_MAX; 
    }

    for (int i = 0; i < sizeB; ++i) {
        B[i] = (data_type)std::rand() / (data_type)RAND_MAX;
    }

    for (int i = 0; i < sizeC; ++i) {
        C[i] = 0.0;
    }
    iters        = 1250;
    const int spike_period = 250;

    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> dist(0, sizeA - 1);

    std::cout << "Total A size (elements) = " << sizeA << std::endl;
    std::cout << "iters=" << iters << ", spikes at 250/500/750/1000 (25/50/75/100%)\n";

    const size_t max_reads = sizeA;

    std::vector<size_t> rand_idx(max_reads);
    for (size_t i = 0; i < max_reads; ++i)
       rand_idx[i] = dist(rng);

    double t_compute_total = 0.0;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream)); 
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
   
    for (int it = 0; it < iters; ++it) {
     if (it > 0 && (it % spike_period == 0) && it <= 1000) {

        int block = it / spike_period;   // 1,2,3,4
        double frac = 0.25 * block;      // 0.25, 0.50, 0.75, 1.00

        size_t read_elems = static_cast<size_t>(frac * sizeA);
        if (read_elems > sizeA) read_elems = sizeA;
        volatile double sink = 0.0;
        for (size_t i = 0; i < read_elems; ++i) {
            sink += A[rand_idx[i]];
        }
        (void)sink;
      }

      CUDA_CHECK(cudaEventRecord(ev0, stream));
      CUBLAS_CHECK(
        cublasDgemm(
            cublasH,
            transa, transb,
            m, n, k,
            &alpha,
            A, lda,
            B, ldb,
            &beta,
            C, ldc
        )
      );
      //CUDA_CHECK(cudaEventRecord(ev1, stream));
      //CUDA_CHECK(cudaEventSynchronize(ev1));

     /* if (it > 0 && (it % spike_period == 0) && it <= 1000) {

        int block = it / spike_period;   // 1,2,3,4
        double frac = 0.25 * block;      // 0.25, 0.50, 0.75, 1.00

        size_t read_elems = static_cast<size_t>(frac * sizeA);
        if (read_elems > sizeA) read_elems = sizeA;
        volatile double sink = 0.0;
        for (size_t i = 0; i < read_elems; ++i) {
            sink += A[rand_idx[i]];
        }
        (void)sink;
      }*/

      CUDA_CHECK(cudaEventRecord(ev1, stream));
      CUDA_CHECK(cudaEventSynchronize(ev1));

      float ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
      std::cout << "iter: " << it << ", t_iter: " << ms << " ms" << std::endl;
      t_compute_total += ms;
    }

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    t_end2end = wtime() - t_end2end;

    double t_compute_avg = t_compute_total / iters;

    printf("t_compute_total = %.3f ms\n", t_compute_total);
    printf("t_compute_avg   = %.3f ms\n", t_compute_avg);
    printf("t_alloc   = %.3f ms\n", t_alloc);
    //printf("t_transfers   = %.3f ms\n", t_transfers);
    printf("t_end2end       = %.3f ms \n ============================================= \n", 1e3 * t_end2end);
    //}
    free(A);
    free(B);
    free(C);

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}

