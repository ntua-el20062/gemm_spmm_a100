#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <ctime>

#include "utils.h"

/*
   initialization of matrices and normalization:
     X0: m*k, all 1's
     Bi: a k*n matrix where every entry is the same constant = c0
     compute C1 = X*B0 = (m,k)*(k,n) = k*c0
     X <- C1
     then I normilize with 2-norm: ||X||2 = (sum from 1 to m*n of (xi)^2)^1/2 
     so I end up with scaling by X*2-norm = 1/(n*m)^1/2
 */

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

int main(int argc, char** argv) {
    int m = 20000;   //rows of X and C
    int k = 20000;   //cols of X, rows of B
    int n = 20000;   //cols of B and C

    int num_steps  = 14; //B_is, default
    int batch_size = 10;  //default

    if (argc >= 4) {
        int tmp1 = std::atoi(argv[1]);
        if (tmp1 > 0) {
            n = m = k  = tmp1;
        }
        int tmp = std::atoi(argv[2]);
        if (tmp > 0) {
            batch_size = tmp;
        }
	int s = std::atoi(argv[3]);
        if (s > 0) {
            num_steps = s;
        }
    }

    if (batch_size > num_steps)
        batch_size = num_steps;
    if (batch_size <= 0)
        batch_size = 1;

    std::printf("Using N = %d, batch_size = %d (num_steps = %d)\n",
                n, batch_size, num_steps);

    double t_e2e = wtime();
    //column-major layout
    int lda_X = m;
    int ldb   = k;
    int ldc   = m;

    //X must be large enough to hold m x max(k,n)
    int max_kn = std::max(k, n);

    size_t bytes_X       = (size_t)m * max_kn * sizeof(data_type);
    size_t bytes_B_one   = (size_t)k * n * sizeof(data_type);     //one B_i
    size_t bytes_B_batch = (size_t)batch_size * bytes_B_one;      //up to batch_size B_i
    size_t bytes_C       = (size_t)m * n * sizeof(data_type);

    int num_batches = (num_steps + batch_size - 1) / batch_size;

    printf("number of batches= %d\n", num_batches);
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    #if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 8000
    CUBLAS_CHECK(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED));
    #endif
    
    cudaStream_t stream_compute, stream_h2d;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_h2d,      cudaStreamNonBlocking));

    CUBLAS_CHECK(cublasSetStream(handle, stream_compute));

    data_type* d_X = nullptr;
    data_type* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_X, bytes_X));
    CUDA_CHECK(cudaMalloc(&d_C, bytes_C));

    double t_gpu_alloc = wtime();
    //two device buffers for B batches (double-buffered, one cor computation one for h2d)
    data_type* d_B_batch[2] = {nullptr, nullptr};
    CUDA_CHECK(cudaMalloc(&d_B_batch[0], bytes_B_batch));
    CUDA_CHECK(cudaMalloc(&d_B_batch[1], bytes_B_batch));
    t_gpu_alloc = wtime() - t_gpu_alloc;


    double t_cpu_alloc = wtime();
    size_t bytes_B_all = (size_t)num_steps * bytes_B_one;
    data_type* h_B_all = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_B_all, bytes_B_all));  //pinned host memory for cudamemcpyasync
    t_cpu_alloc = wtime() - t_cpu_alloc;


    double t_init = wtime();
    //init B_i on cpu -> first touch -> DDR
    for (int step = 0; step < num_steps; ++step) {
        data_type* h_Bi = h_B_all + (size_t)step * k * n;
        for (size_t idx = 0; idx < (size_t)k * n; ++idx) {
            h_Bi[idx] = (data_type)((step + 1) * 0.001);
        }
    }

    //init starting matrix A or C0 on CPU/GPU
    std::vector<data_type> h_X_init((size_t)m * max_kn, 0.0);
    for (int i = 0; i < m * k; ++i) {
        h_X_init[i] = 1.0;  // whatever you want
    }
    t_init = wtime() - t_init;

    double t_transfers = wtime();
    CUDA_CHECK(cudaMemcpy(d_X, h_X_init.data(), bytes_X, cudaMemcpyHostToDevice));
    t_transfers = wtime() - t_transfers;

    data_type alpha = 1.0;
    data_type beta  = 0.0;

    cublasOperation_t transa = CUBLAS_OP_N; 
    cublasOperation_t transb = CUBLAS_OP_N;

    //signal H2D completion for each device batch buffer
    cudaEvent_t h2d_done[2];
    CUDA_CHECK(cudaEventCreateWithFlags(&h2d_done[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&h2d_done[1], cudaEventDisableTiming));

    //per batch timing events
    std::vector<cudaEvent_t> ev_h2d_start(num_batches),  ev_h2d_stop(num_batches);

    for (int b = 0; b < num_batches; ++b) {
      CUDA_CHECK(cudaEventCreate(&ev_h2d_start[b]));
      CUDA_CHECK(cudaEventCreate(&ev_h2d_stop[b]));
    }

    //time GEMM
    std::vector<cudaEvent_t> gemm_start(num_steps), gemm_end(num_steps);
    for (int i = 0; i < num_steps; ++i) {
        CUDA_CHECK(cudaEventCreate(&gemm_start[i]));
        CUDA_CHECK(cudaEventCreate(&gemm_end[i]));
    }
    
    //preload first batch (batch 0) into device (buffer 0)
    if (num_steps > 0) {
        int batch_idx      = 0;
        int start_step     = batch_idx * batch_size;
        int end_step       = std::min(num_steps, start_step + batch_size);
        int steps_in_batch = end_step - start_step;
        size_t bytes_this_batch = (size_t)steps_in_batch * bytes_B_one;

        //pointer to first B in this batch inside the big host array
        data_type* src = h_B_all + (size_t)start_step * k * n;

/*      CUDA_CHECK(cudaMemcpyAsync(d_B_batch[0], src, bytes_this_batch,cudaMemcpyHostToDevice, stream_h2d));
        CUDA_CHECK(cudaEventRecord(h2d_done[0], stream_h2d));
*/  
	CUDA_CHECK(cudaEventRecord(ev_h2d_start[batch_idx], stream_h2d));
        CUDA_CHECK(cudaMemcpyAsync(d_B_batch[0], src, bytes_this_batch,
                               cudaMemcpyHostToDevice, stream_h2d));
        CUDA_CHECK(cudaEventRecord(ev_h2d_stop[batch_idx], stream_h2d));

        CUDA_CHECK(cudaEventRecord(h2d_done[0], stream_h2d));
     }

    //loop over batches, pipeline with double-buffered overlap of memory transfers and computation
    for (int batch = 0; batch < num_batches; ++batch) {
        int curr = batch % 2;      //which device batch buffer to use now
        int next = 1 - curr;       //the other buffer is for prefetching
        int start_step     = batch * batch_size;
        int end_step       = std::min(num_steps, start_step + batch_size);
        int steps_in_batch = end_step - start_step;
        size_t bytes_this_batch= (size_t)steps_in_batch*bytes_B_one;
        //wait until this batch's H2D transfer is complete
        CUDA_CHECK(cudaStreamWaitEvent(stream_compute, h2d_done[curr], 0));
        //start prefetching next batch while this batch computes
        if (batch + 1 < num_batches) {
            int next_batch      = batch + 1;
            int next_start_step = next_batch * batch_size;
            int next_end_step   = std::min(num_steps, next_start_step + batch_size);
            int next_steps      = next_end_step - next_start_step;
            size_t bytes_next_batch = (size_t)next_steps *bytes_B_one;
            data_type* src_next = h_B_all + (size_t)next_start_step * k * n;

/*          CUDA_CHECK(cudaMemcpyAsync(d_B_batch[next], src_next, bytes_next_batch,cudaMemcpyHostToDevice, stream_h2d));
            CUDA_CHECK(cudaEventRecord(h2d_done[next], stream_h2d));
*/
	    CUDA_CHECK(cudaEventRecord(ev_h2d_start[next_batch], stream_h2d));
            CUDA_CHECK(cudaMemcpyAsync(d_B_batch[next], src_next, bytes_next_batch,
                               cudaMemcpyHostToDevice, stream_h2d));
            CUDA_CHECK(cudaEventRecord(ev_h2d_stop[next_batch], stream_h2d));

            CUDA_CHECK(cudaEventRecord(h2d_done[next], stream_h2d));    
      	}

        //compute
        for (int step = start_step; step < end_step; ++step) {
            int local = step - start_step;
            //pointer to B_step inside the current device batch buffer
            data_type* d_B_step = d_B_batch[curr] + (size_t)local * k * n;

	    CUDA_CHECK(cudaEventRecord(gemm_start[step], stream_compute));   
	    CUBLAS_CHECK(
                cublasDgemm(
                    handle,
                    transa, transb,
                    m, n, (step == 0 ? k : n), //inner dim: k for step 0, n afterwards if needed
                    &alpha,
                    d_X, lda_X,
                    d_B_step, ldb,
                    &beta,
                    d_C, ldc
                )
            );
            CUDA_CHECK(cudaEventRecord(gemm_end[step], stream_compute));

            //X_i+1 overwrites X (use C as temp)
            //CUDA_CHECK(cudaMemcpyAsync(d_X, d_C, bytes_C, cudaMemcpyDeviceToDevice, stream_compute));
	    data_type* tmp = d_X;
            d_X = d_C;
            d_C = tmp;
	}
        //normalization
        int len = m * n; data_type normX = 0.0;
        // cublasDnrm2 treats d_X as a vector of length len with stride 1
        CUBLAS_CHECK( cublasDnrm2(handle,len, d_X, 1, &normX ));
        if (normX > 0.0) {
                    data_type inv_norm = 1.0 / normX;
                    // Scale X in-place: X <- inv_norm * X
                    CUBLAS_CHECK(cublasDscal(handle,len,&inv_norm,d_X, 1));
        }


    }

    //wait for all work to finish
    CUDA_CHECK(cudaStreamSynchronize(stream_compute));
    CUDA_CHECK(cudaStreamSynchronize(stream_h2d));
    
    float total_comp_ms = 0.0f;
    float total_h2d_ms  = 0.0f;

    for (int b = 0; b < num_batches; ++b) {
       float ms = 0.0f;
       CUDA_CHECK(cudaEventElapsedTime(&ms, ev_h2d_start[b], ev_h2d_stop[b]));
       total_h2d_ms += ms;
    }

    double t_gemm = 0.0;
    for (int step = 0; step < num_steps; ++step) {
        float gemm_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gemm_ms, gemm_start[step], gemm_end[step]));
        t_gemm += gemm_ms;
    }


    printf("t_gemm = %f ms\n", t_gemm);
    printf("t_h2d = %f ms\n", (1e3*t_transfers)+total_h2d_ms);

    double t_d2h = wtime();    
    //copy result d2h
    std::vector<data_type> h_X_final((size_t)m * n);
    CUDA_CHECK(cudaMemcpy(
        h_X_final.data(), d_X, bytes_C, cudaMemcpyDeviceToHost
    ));
    t_d2h = wtime() - t_d2h;
    printf("t_d2h = %f ms\n", 1e3*t_d2h);

    //sanity check
    for (int i = 0; i < 5 && i < m * n; ++i) {
        std::printf("h_X_final[%d] = %f\n", i, h_X_final[i]);
    }

    for (int b = 0; b < num_batches; ++b) {
       CUDA_CHECK(cudaEventDestroy(ev_h2d_start[b]));
       CUDA_CHECK(cudaEventDestroy(ev_h2d_stop[b]));
    }

    for (int i = 0; i < num_steps; ++i) {
       CUDA_CHECK(cudaEventDestroy(gemm_start[i]));
       CUDA_CHECK(cudaEventDestroy(gemm_end[i]));
    }

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_B_batch[0]));
    CUDA_CHECK(cudaFree(d_B_batch[1]));

    CUDA_CHECK(cudaFreeHost(h_B_all));

    CUDA_CHECK(cudaEventDestroy(h2d_done[0]));
    CUDA_CHECK(cudaEventDestroy(h2d_done[1]));

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaStreamDestroy(stream_h2d));

    CUDA_CHECK(cudaDeviceReset());
    printf("t_gpu_alloc = %f ms\n", t_gpu_alloc*1e3);
    printf("t_cpu_alloc = %f ms\n", t_cpu_alloc*1e3);
    printf("t_init = %f ms\n", 1e3*t_init);
    t_e2e = 1e3*(wtime() - t_e2e);
    printf("t_end_2_end = %f ms\n", t_e2e);
    //printf("t_gpu_alloc = %f ms\n", t_gpu_alloc*1e3);
    //printf("t_cpu_alloc = %f ms\n", t_cpu_alloc*1e3);
    //printf("t_init = %f ms\n", 1e3*t_init);

    return 0;
}

