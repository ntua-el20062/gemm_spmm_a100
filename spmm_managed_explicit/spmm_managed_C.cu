#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
#include <ctime>
#include <vector>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <random>
#include <iostream>
#include <unistd.h>
#include <cctype>
#include <cstring>
#include <algorithm>

double wtime(void)
{
    double now_time;
    struct timeval  etstart;
    struct timezone tzp;

    if (gettimeofday(&etstart, &tzp) == -1)
        perror("Error: calling gettimeofday() not successful.\n");

    now_time = ((double)etstart.tv_sec) + ((double)etstart.tv_usec) / 1000000.0;
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

#define CUDA_CHECK(call) do {                                \
    cudaError_t _e = (call);                                 \
    if (_e != cudaSuccess) {                                 \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",        \
            __FILE__, __LINE__, cudaGetErrorString(_e));      \
        std::exit(EXIT_FAILURE);                              \
    }                                                        \
} while(0)

#define CUSPARSE_CHECK(call) do {                             \
    cusparseStatus_t _s = (call);                              \
    if (_s != CUSPARSE_STATUS_SUCCESS) {                       \
        std::fprintf(stderr, "cuSPARSE error %s:%d: %d\n",      \
            __FILE__, __LINE__, (int)_s);                      \
        std::exit(EXIT_FAILURE);                               \
    }                                                         \
} while(0)

using data_type  = double;
using index_type = int;

struct CooEntry {
    index_type r, c;
    data_type  v;
};

static bool read_matrix_market_coo(
    const char* path,
    index_type &m, index_type &k,
    std::vector<CooEntry> &coo)
{
    FILE* f = std::fopen(path, "r");
    if (!f) {
        std::perror("fopen");
        return false;
    }

    char line[512];
    if (!std::fgets(line, sizeof(line), f)) {
        std::fclose(f);
        return false;
    }

    if (std::strncmp(line, "%%MatrixMarket", 14) != 0) {
        std::fprintf(stderr, "Not a MatrixMarket file\n");
        std::fclose(f);
        return false;
    }

    char object[64], format[64], field[64], symmetry[64];
    if (std::sscanf(line, "%%%%MatrixMarket %63s %63s %63s %63s", object, format, field, symmetry) != 4) {
        std::fprintf(stderr, "Bad MatrixMarket header\n");
        std::fclose(f);
        return false;
    }

    auto lower = [](char* s){
        for (; *s; ++s) *s = (char)std::tolower(*s);
    };
    lower(object); lower(format); lower(field); lower(symmetry);

    if (std::strcmp(object, "matrix") != 0 || std::strcmp(format, "coordinate") != 0) {
        std::fprintf(stderr, "Only 'matrix coordinate' supported\n");
        std::fclose(f);
        return false;
    }

    const bool is_pattern = (std::strcmp(field, "pattern") == 0);
    const bool is_real    = (std::strcmp(field, "real") == 0);
    const bool is_integer = (std::strcmp(field, "integer") == 0);
    if (!(is_pattern || is_real || is_integer)) {
        std::fprintf(stderr, "Only real/integer/pattern supported\n");
        std::fclose(f);
        return false;
    }

    const bool is_symmetric = (std::strcmp(symmetry, "symmetric") == 0);

    do {
        if (!std::fgets(line, sizeof(line), f)) {
            std::fclose(f);
            return false;
        }
    } while (line[0] == '%');

    long mm=0, kk=0, nnz=0;
    if (std::sscanf(line, "%ld %ld %ld", &mm, &kk, &nnz) != 3) {
        std::fprintf(stderr, "Bad size line\n");
        std::fclose(f);
        return false;
    }

    m = (index_type)mm;
    k = (index_type)kk;

    coo.clear();
    coo.reserve((size_t)nnz * (is_symmetric ? 2u : 1u));

    for (long t = 0; t < nnz; ++t) {
        long r1=0, c1=0;
        double v = 1.0;

        if (is_pattern) {
            if (std::fscanf(f, "%ld %ld", &r1, &c1) != 2) break;
            v = 1.0;
        } else if (is_integer) {
            long iv=0;
            if (std::fscanf(f, "%ld %ld %ld", &r1, &c1, &iv) != 3) break;
            v = (double)iv;
        } else { //real
            if (std::fscanf(f, "%ld %ld %lf", &r1, &c1, &v) != 3) break;
        }

        index_type r = (index_type)(r1 - 1);
        index_type c = (index_type)(c1 - 1);

        coo.push_back({r, c, (data_type)v});

        if (is_symmetric && r != c) {
            coo.push_back({c, r, (data_type)v});
        }
    }

    std::fclose(f);

    std::sort(coo.begin(), coo.end(), [](const CooEntry& a, const CooEntry& b){
        if (a.r != b.r) return a.r < b.r;
        return a.c < b.c;
    });

    return true;
}

static void coo_to_csr(
    index_type m, index_type k,
    const std::vector<CooEntry> &coo,
    std::vector<index_type> &rowPtr,
    std::vector<index_type> &colInd,
    std::vector<data_type>  &vals)
{
    (void)k;
    const size_t nnz = coo.size();
    rowPtr.assign((size_t)m + 1, 0);
    colInd.resize(nnz);
    vals.resize(nnz);

    for (size_t i = 0; i < nnz; ++i) {
        rowPtr[(size_t)coo[i].r + 1]++;
    }
    for (index_type i = 0; i < m; ++i) {
        rowPtr[(size_t)i + 1] += rowPtr[(size_t)i];
    }

    std::vector<index_type> next(rowPtr.begin(), rowPtr.end());
    for (size_t i = 0; i < nnz; ++i) {
        index_type r = coo[i].r;
        index_type dst = next[(size_t)r]++;
        colInd[(size_t)dst] = coo[i].c;
        vals[(size_t)dst]   = coo[i].v;
    }
}

int main(int argc, char** argv) {
    std::printf("MANAGED VERSION (cuSPARSE SpMM, A from .mtx, B=Identity)\n");

    if (argc < 2) {
        std::printf("Usage: %s A.mtx [ITERS]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* mtx_path = argv[1];
    int iters = 1250;

    index_type n=0, k=0;
    if (argc >= 3) {
      n = static_cast<index_type>(std::atoi(argv[2]));
    }
    
    index_type m=0;
    std::vector<CooEntry> coo;
    if (!read_matrix_market_coo(mtx_path, m, k, coo)) {
        std::fprintf(stderr, "Failed to read MatrixMarket file: %s\n", mtx_path);
        return EXIT_FAILURE;
    }

    std::vector<index_type> h_rowPtr, h_colInd;
    std::vector<data_type>  h_vals;
    coo_to_csr(m, k, coo, h_rowPtr, h_colInd, h_vals);

    const int64_t nnz = (int64_t)h_vals.size();

    std::printf("Loaded A: m=%d, k=%d, nnz=%lld. Using B=I (k=%d), so n=%d\n",
                (int)m, (int)k, (long long)nnz, (int)k, (int)n);
    std::printf("iters=%d, spikes at 250/500/750/1000 (25/50/75/100%%)\n", iters);

    double t_end2end = wtime();

    MemLogger memlog;
    memlog.open("memlog_managed_C.csv", t_end2end);
    memlog.sample(wtime());

    cudaStream_t stream = nullptr;
    cusparseHandle_t handle = nullptr;

    CUSPARSE_CHECK(cusparseCreate(&handle));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSPARSE_CHECK(cusparseSetStream(handle, stream));

    memlog.sample(wtime()); // after init

    index_type *d_rowPtr = nullptr, *d_colInd = nullptr;
    data_type  *d_vals   = nullptr;
    data_type  *d_B      = nullptr;
    data_type  *d_C      = nullptr;

    const size_t bytesRowPtr = (size_t)(m + 1) * sizeof(index_type);
    const size_t bytesColInd = (size_t)nnz * sizeof(index_type);
    const size_t bytesVals   = (size_t)nnz * sizeof(data_type);
    const size_t bytesB      = (size_t)k * (size_t)n * sizeof(data_type);
    const size_t bytesC      = (size_t)m * (size_t)n * sizeof(data_type);

    double t_cuda_alloc = wtime();
    CUDA_CHECK(cudaMallocManaged(&d_rowPtr, bytesRowPtr));
    memlog.sample(wtime());
    CUDA_CHECK(cudaMallocManaged(&d_colInd, bytesColInd));
    memlog.sample(wtime());
    CUDA_CHECK(cudaMallocManaged(&d_vals, bytesVals));
    memlog.sample(wtime());
    CUDA_CHECK(cudaMallocManaged(&d_B, bytesB));
    memlog.sample(wtime());
    CUDA_CHECK(cudaMallocManaged(&d_C, bytesC));
    t_cuda_alloc = 1e3*(wtime() - t_cuda_alloc);

    memlog.sample(wtime()); // after UM alloc

    std::memcpy(d_rowPtr, h_rowPtr.data(), bytesRowPtr);
    memlog.sample(wtime());
    std::memcpy(d_colInd, h_colInd.data(), bytesColInd);
    memlog.sample(wtime());
    std::memcpy(d_vals,   h_vals.data(),   bytesVals);
    memlog.sample(wtime());

    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < k; ++row) {
            d_B[(size_t)row + (size_t)col * (size_t)k] = (row == col) ? 1.0 : 0.0; // column-major
        }
    }
    memlog.sample(wtime());
    for (size_t i = 0; i < (size_t)m * (size_t)n; ++i) d_C[i] = 0.0;
    memlog.sample(wtime());

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    const int64_t ldB = k;
    const int64_t ldC = m;

    CUSPARSE_CHECK(cusparseCreateCsr(
        &matA,
        (int64_t)m, (int64_t)k, (int64_t)nnz,
        d_rowPtr, d_colInd, d_vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_64F));

    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matB, (int64_t)k, (int64_t)n, (int64_t)ldB,
        d_B, CUDA_R_64F, CUSPARSE_ORDER_COL));

    CUSPARSE_CHECK(cusparseCreateDnMat(
        &matC, (int64_t)m, (int64_t)n, (int64_t)ldC,
        d_C, CUDA_R_64F, CUSPARSE_ORDER_COL));

    memlog.sample(wtime());

    const data_type alpha = 1.0;
    const data_type beta  = 0.0;

    size_t bufferSize = 0;
    void*  dBuffer = nullptr;

    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        matB,
        &beta,
        matC,
        CUDA_R_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));

    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));
    memlog.sample(wtime()); // after workspace alloc

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    const int spike_period = 250;
    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> dist(0, (size_t)m*(size_t)n - 1);

    double t_compute_total = 0.0;

    for (int it = 0; it < iters; ++it) {
        memlog.sample(wtime());

        if (it > 0 && (it % spike_period == 0) && it <= 1000) {
            int block = it / spike_period;        // 1..4
            double frac = 0.25 * block;           // 0.25,0.5,0.75,1.0
            size_t read_elems = (size_t)(frac * (double)nnz);
            if (read_elems > (size_t)nnz) read_elems = (size_t)nnz;

            volatile double sink = 0.0;
            for (size_t i = 0; i < read_elems; ++i) {
		    size_t idx = dist(rng);
                    //sink += d_vals[idx];          // touch values
                    //sink += (double)d_colInd[idx];
		    d_C[idx] += 0.01;
		    if (i % 10 == 0) memlog.sample(wtime());
            }
            (void)sink;
        }

        CUDA_CHECK(cudaEventRecord(ev0, stream));
        memlog.sample(wtime());

        CUSPARSE_CHECK(cusparseSpMM(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            matB,
            &beta,
            matC,
            CUDA_R_64F,
            CUSPARSE_SPMM_ALG_DEFAULT,
            dBuffer));

        memlog.sample(wtime());
        CUDA_CHECK(cudaEventRecord(ev1, stream));
        CUDA_CHECK(cudaEventSynchronize(ev1));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        std::cout << "iter: " << it << ", t_iter: " << ms << " ms" << std::endl;
        t_compute_total += ms;
    }

    memlog.sample(wtime()); // after loop

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    const double t_compute_avg = t_compute_total / (double)iters;
    const double t_end2end_s = wtime() - t_end2end;

    std::printf("t_cuda_alloc     = %.3f ms\n", t_cuda_alloc);
    std::printf("t_compute_total  = %.3f ms\n", t_compute_total);
    std::printf("t_compute_avg    = %.3f ms\n", t_compute_avg);
    std::printf("t_end2end        = %.3f ms\n", 1e3 * t_end2end_s);
    std::printf("==================\n");

    // Cleanup
    CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matB));
    CUSPARSE_CHECK(cusparseDestroyDnMat(matC));
    CUSPARSE_CHECK(cusparseDestroy(handle));

    CUDA_CHECK(cudaFree(dBuffer));

    CUDA_CHECK(cudaFree(d_rowPtr));
    CUDA_CHECK(cudaFree(d_colInd));
    CUDA_CHECK(cudaFree(d_vals));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    memlog.sample(wtime()); // after free
    memlog.close();

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}

