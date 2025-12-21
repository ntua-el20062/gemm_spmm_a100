#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <ctime>
#include "utils.h"
#include <fstream>
#include <string>
#include <unordered_map>
#include <cstdio>
#include <unistd.h>
#include <cmath>   // for fabs

/*
cyclic-buffer execution logic
use a device buffer, each position holding one B matrix:
data_type* d_B_buff[buff_size];

I also have 2 events for each slot, so I can achieve synchonization:
cudaEvent_t h2d_done[buff_size];   //B in this slot is ready for GEMM
cudaEvent_t gemm_done[buff_size];  //GEMM finished using this slot

1) preload the first buff_size matrices:
for (int i = 0; i < buff_size; ++i) {
    cudaMemcpyAsync(d_B_buff[i], B[i], ... , stream_h2d);
    cudaEventRecord(h2d_done[i], stream_h2d);
}

2) main loop: at step s, use slot = s % buff_size.
before launching GEMM, ensure the B in that slot has completed copying:
cudaStreamWaitEvent(stream_compute, h2d_done[slot], 0);
cublasDgemm(... d_B_buff[slot] ... );
cudaEventRecord(gemm_done[slot], stream_compute);

prefetch the next B that will eventually map to this same slot B[s + buff_size]
but we must not overwrite the slot until the previous GEMM is done
int next_B = s + buff_size;
if (next_B < num_steps) {
    cudaStreamWaitEvent(stream_h2d, gemm_done[slot], 0);
    cudaMemcpyAsync(d_B_buff[slot], B[next_B], ..., stream_h2d);
    cudaEventRecord(h2d_done[slot], stream_h2d);
}

this enforces the order:
(slot)  H2D(Bs) → h2d_done → GEMM(s) → gemm_done → H2D(B{s+buff_size}) → h2d_done → GEMM(s+buff_size)
 */

void print_memory() {
    // GPU
    size_t free_bytes;
    size_t total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    const float GB = 1024.0f * 1024.0f * 1024.0f;

    float free_gb  = (float)free_bytes / GB;
    float total_gb = (float)total_bytes / GB;
    float used_gb  = total_gb - free_gb;

    printf("--- GPU Memory Usage ---\n");
    printf("Total GPU Memory: %.2f GB\n", total_gb);
    printf("Free GPU Memory:  %.2f GB\n", free_gb);
    printf("Used GPU Memory:  %.2f GB\n", used_gb);
    printf("------------------------\n\n");

    // CPU
    long rss_pages, virt_pages;
    FILE* f = fopen("/proc/self/statm", "r");
    fscanf(f, "%ld %ld", &virt_pages, &rss_pages);
    fclose(f);

    long page_size_kb = sysconf(_SC_PAGESIZE) / 1024;

    printf("Process RSS:  %.2f GB\n",
           rss_pages * page_size_kb / 1024.0 / 1024.0);
}

double wtime(void)
{
    struct timeval  etstart;
    struct timezone tzp;
    if (gettimeofday(&etstart, &tzp) == -1)
        perror("Error: calling gettimeofday() not successful.\n");

    return ((double)etstart.tv_sec) +
           ((double)etstart.tv_usec) / 1000000.0;
}

using data_type = double;

int main(int argc, char** argv) {
    int m = 20000;   // rows of X and C
    int k = 20000;   // cols of X, rows of B
    int n = 20000;   // cols of B and C

    int num_steps  = 14; // number of B_i
    int buff_size  = 10; // cyclic buffer size
    bool identity_test = false; // if true, B_i = I and no normalization

    // Args:
    // argv[1] = N => m = n = k = N
    // argv[2] = buff_size
    // argv[3] = num_steps
    // argv[4] = identity_test (0 or 1)
    if (argc >= 4) {
        int tmp1 = std::atoi(argv[1]);
        if (tmp1 > 0) {
            n = m = k = tmp1;
        }
        int tmp = std::atoi(argv[2]);
        if (tmp > 0) {
            buff_size = tmp;
        }
        int s = std::atoi(argv[3]);
        if (s > 0) {
            num_steps = s;
        }
    }
    if (argc >= 5) {
        identity_test = (std::atoi(argv[4]) != 0);
    }

    if (buff_size > num_steps)
        buff_size = num_steps;
    if (buff_size <= 0)
        buff_size = 1;

    std::printf("Using N = %d, buff_size = %d (num_steps = %d), identity_test = %d\n",
                n, buff_size, num_steps, (int)identity_test);

    double t_e2e = wtime();

    // column-major layout
    int lda_X = m;
    int ldb   = k;
    int ldc   = m;

    int max_kn = std::max(k, n);

    size_t bytes_X     = (size_t)m * max_kn * sizeof(data_type);
    size_t bytes_B_one = (size_t)k * n   * sizeof(data_type);
    size_t bytes_C     = (size_t)m * n   * sizeof(data_type);

    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 8000
    CUBLAS_CHECK(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED));
#endif

    // two streams: one for compute, one for H2D transfers
    cudaStream_t stream_compute, stream_h2d;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_h2d,     cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(handle, stream_compute));

    data_type* d_X = nullptr;
    data_type* d_C = nullptr;

    double t_gpu_alloc = wtime();
    CUDA_CHECK(cudaMalloc(&d_X, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    // cyclic buffer of Bs on device: buff_size slots
    data_type** d_B_buff = (data_type**)malloc(buff_size * sizeof(data_type*));
    for (int i = 0; i < buff_size; ++i) {
        CUDA_CHECK(cudaMalloc(&d_B_buff[i], bytes_B_one));
    }
    t_gpu_alloc = wtime() - t_gpu_alloc;

    double t_cpu_alloc = wtime();
    size_t bytes_B_all = (size_t)num_steps * bytes_B_one;
    data_type* h_B_all = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_B_all, bytes_B_all));
    t_cpu_alloc = wtime() - t_cpu_alloc;

    // initialize Bs on host
    double t_init = wtime();
    for (int step = 0; step < num_steps; ++step) {
        data_type* h_Bi = h_B_all + (size_t)step * k * n;

        if (identity_test) {
            // B_i = I  (k x n, assume k == n)
            for (int col = 0; col < n; ++col) {
                for (int row = 0; row < k; ++row) {
                    h_Bi[(size_t)row + (size_t)col * k] =
                        (row == col) ? (data_type)1.0 : (data_type)0.0;
                }
            }
        } else {
            // toy values
            for (size_t idx = 0; idx < (size_t)k * n; ++idx) {
                h_Bi[idx] = (data_type)((step + 1) * 0.001);
            }
        }
    }

    // initial X on host
    std::vector<data_type> h_X_init((size_t)m * max_kn, 0.0);
    for (int i = 0; i < m * k; ++i) {
        h_X_init[i] = 1.0;
    }
    // reference copy of the first m*n entries for identity check
    std::vector<data_type> h_X_ref(h_X_init.begin(),
                                   h_X_init.begin() + (size_t)m * n);

    t_init = wtime() - t_init;

    // H2D: X
    double t_transfers = wtime();
    CUDA_CHECK(cudaMemcpy(d_X, h_X_init.data(), bytes_X, cudaMemcpyHostToDevice));
    t_transfers = wtime() - t_transfers;

    data_type alpha = 1.0;
    data_type beta  = 0.0;

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;

    // h2d_done[i]: B currently in slot i is ready to be used
    // gemm_done[i]: GEMM that used slot i has finished so slot i can be overwritten
    cudaEvent_t* h2d_done  = (cudaEvent_t*)malloc(buff_size * sizeof(cudaEvent_t));
    cudaEvent_t* gemm_done = (cudaEvent_t*)malloc(buff_size * sizeof(cudaEvent_t));

    for (int i = 0; i < buff_size; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&h2d_done[i],  cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&gemm_done[i], cudaEventDisableTiming));
    }

    // timing for H2D
    std::vector<cudaEvent_t> h2d_start(num_steps), h2d_end(num_steps);
    for (int i = 0; i < num_steps; ++i) {
        CUDA_CHECK(cudaEventCreate(&h2d_start[i]));
        CUDA_CHECK(cudaEventCreate(&h2d_end[i]));
    }

    // preload first window of Bs
    int preload = std::min(num_steps, buff_size);
    for (int step = 0; step < preload; ++step) {
        data_type* src = h_B_all + (size_t)step * k * n;

        CUDA_CHECK(cudaEventRecord(h2d_start[step], stream_h2d));
        CUDA_CHECK(cudaMemcpyAsync(d_B_buff[step], src, bytes_B_one,
                                   cudaMemcpyHostToDevice, stream_h2d));
        CUDA_CHECK(cudaEventRecord(h2d_end[step], stream_h2d));

        CUDA_CHECK(cudaEventRecord(h2d_done[step], stream_h2d));
    }

    // timing for GEMM
    std::vector<cudaEvent_t> gemm_start(num_steps), gemm_end(num_steps);
    for (int i = 0; i < num_steps; ++i) {
        CUDA_CHECK(cudaEventCreate(&gemm_start[i]));
        CUDA_CHECK(cudaEventCreate(&gemm_end[i]));
    }

    double t_gemm = 0.0;

    // GEMM loop with cyclic buffer
    for (int step = 0; step < num_steps; ++step) {
        int slot = step % buff_size;

        // wait until the B needed for this step is present in this slot
        CUDA_CHECK(cudaStreamWaitEvent(stream_compute, h2d_done[slot], 0));

        int inner_dim = (step == 0 ? k : n);  // k==n typically
        data_type* d_B_step = d_B_buff[slot];

        CUDA_CHECK(cudaEventRecord(gemm_start[step], stream_compute));

        CUBLAS_CHECK(
            cublasDgemm(
                handle,
                transa, transb,
                m, n, inner_dim,
                &alpha,
                d_X, lda_X,
                d_B_step, ldb,
                &beta,
                d_C, ldc
            )
        );

        CUDA_CHECK(cudaEventRecord(gemm_end[step], stream_compute));
        // after GEMM has been queued, mark this slot as used for this step
        CUDA_CHECK(cudaEventRecord(gemm_done[slot], stream_compute));

        // X{i+1} <- C via pointer swap
        data_type* tmp = d_X;
        d_X = d_C;
        d_C = tmp;

        // normalization only in non-identity test mode
        if (!identity_test) {
            int len = m * n;
            data_type normX = 0.0;
            CUBLAS_CHECK(cublasDnrm2(handle, len, d_X, 1, &normX));
            if (normX > 0.0) {
                data_type inv_norm = 1.0 / normX;
                CUBLAS_CHECK(cublasDscal(handle, len, &inv_norm, d_X, 1));
            }
        }

        // prefetch the B that will be needed buff_size steps ahead
        int next_B = step + buff_size;
        if (next_B < num_steps) {
            data_type* src_next = h_B_all + (size_t)next_B * k * n;

            // don't overwrite this slot until GEMM has finished with it
            CUDA_CHECK(cudaStreamWaitEvent(stream_h2d, gemm_done[slot], 0));

            CUDA_CHECK(cudaEventRecord(h2d_start[next_B], stream_h2d));
            CUDA_CHECK(cudaMemcpyAsync(d_B_buff[slot], src_next, bytes_B_one,
                                       cudaMemcpyHostToDevice, stream_h2d));
            CUDA_CHECK(cudaEventRecord(h2d_end[next_B], stream_h2d));

            CUDA_CHECK(cudaEventRecord(h2d_done[slot], stream_h2d));
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_h2d));

    for (int step = 0; step < num_steps; ++step) {
        float gemm_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gemm_ms, gemm_start[step], gemm_end[step]));
        t_gemm += gemm_ms;
    }

    float t_h2d = 0.0f;
    for (int step = 0; step < num_steps; ++step) {
        float h2d_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, h2d_start[step], h2d_end[step]));
        t_h2d += h2d_ms;
    }

    double t_d2h = wtime();
    std::vector<data_type> h_X_final((size_t)m * n);
    CUDA_CHECK(cudaMemcpy(h_X_final.data(), d_X, bytes_C, cudaMemcpyDeviceToHost));
    t_d2h = wtime() - t_d2h;

    print_memory();

    // sanity: print first few entries
    for (int i = 0; i < 5 && i < m * n; ++i) {
        std::printf("h_X_final[%d] = %f\n", i, h_X_final[i]);
    }

    // IDENTITY CHECK: X_final should equal original X (for m x n)
    if (identity_test) {
        double max_abs_diff = 0.0;
        for (size_t idx = 0; idx < (size_t)m * n; ++idx) {
            double diff = std::fabs((double)h_X_final[idx] -
                                    (double)h_X_ref[idx]);
            if (diff > max_abs_diff) max_abs_diff = diff;
        }
        double tol = 1e-10; // for double
        if (max_abs_diff <= tol) {
            std::printf("IDENTITY CHECK PASSED: max |X_final - X_init| = %e\n",
                        max_abs_diff);
        } else {
            std::printf("IDENTITY CHECK FAILED: max |X_final - X_init| = %e\n",
                        max_abs_diff);
        }
    }

    // cleanup
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_C));
    for (int i = 0; i < buff_size; ++i) {
        CUDA_CHECK(cudaFree(d_B_buff[i]));
        CUDA_CHECK(cudaEventDestroy(h2d_done[i]));
        CUDA_CHECK(cudaEventDestroy(gemm_done[i]));
    }
    free(d_B_buff);
    free(h2d_done);
    free(gemm_done);
    CUDA_CHECK(cudaFreeHost(h_B_all));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaStreamDestroy(stream_h2d));
    CUDA_CHECK(cudaDeviceReset());

    printf("t_alloc_gpu = %f ms\n", 1e3 * t_gpu_alloc);
    printf("t_alloc_cpu = %f ms\n", 1e3 * t_cpu_alloc);
    printf("t_init      = %f ms\n", 1e3 * t_init);
    printf("t_h2d       = %f ms\n", 1e3 * (t_transfers) + t_h2d);
    printf("t_d2h       = %f ms\n", 1e3 * t_d2h);
    printf("t_gemm      = %f ms\n", t_gemm);

    t_e2e = 1e3 * (wtime() - t_e2e);
    printf("t_end_2_end = %f ms\n", t_e2e);

    return 0;
}

