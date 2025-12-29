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
        } else { // real
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

// Helper: time an async memcpy on a stream using CUDA events
static float timed_memcpy_async(void* dst, const void* src, size_t bytes,
                                cudaMemcpyKind kind, cudaStream_t stream)
{
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a, stream));
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, kind, stream));
    CUDA_CHECK(cudaEventRecord(b, stream));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return ms;
}

int main(int argc, char** argv) {
    std::printf("EXPLICIT H/D VERSION (cuSPARSE SpMM, A from .mtx, B=Identity)\n");

    if (argc < 2) {
        std::printf("Usage: %s A.mtx [n] [ITERS]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* mtx_path = argv[1];

    index_type n = 0;      // number of columns of B/C
    int iters = 1250;

    if (argc >= 3) {
        n = static_cast<index_type>(std::atoi(argv[2]));
    }
    if (argc >= 4) {
        iters = std::atoi(argv[3]);
    }

    index_type m=0, k=0;
    std::vector<CooEntry> coo;
    if (!read_matrix_market_coo(mtx_path, m, k, coo)) {
        std::fprintf(stderr, "Failed to read MatrixMarket file: %s\n", mtx_path);
        return EXIT_FAILURE;
    }

    if (n <= 0) {
        // If user didn't pass n, default to square-ish: n = k (classic B=I)
        n = k;
    }

    std::vector<index_type> h_rowPtr, h_colInd;
    std::vector<data_type>  h_vals;
    coo_to_csr(m, k, coo, h_rowPtr, h_colInd, h_vals);

    const int64_t nnz = (int64_t)h_vals.size();

    std::printf("Loaded A: m=%d, k=%d, nnz=%lld. Using B=I (k=%d), n=%d\n",
                (int)m, (int)k, (long long)nnz, (int)k, (int)n);
    std::printf("iters=%d, spikes at 250/500/750/1000 (25/50/75/100%%)\n", iters);

    double t_end2end = wtime();

    MemLogger memlog;
    memlog.open("memlog_explicit_spmm_A.csv", t_end2end);
    memlog.sample(wtime());

    cudaStream_t stream = nullptr;
    cusparseHandle_t handle = nullptr;

    CUSPARSE_CHECK(cusparseCreate(&handle));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSPARSE_CHECK(cusparseSetStream(handle, stream));

    memlog.sample(wtime()); // after init

    // --- Host dense matrices (column-major to match your original code) ---
    // If you want faster transfers: consider pinned memory via cudaMallocHost for these.
    std::vector<data_type> h_B((size_t)k * (size_t)n, 0.0);
    std::vector<data_type> h_C((size_t)m * (size_t)n, 0.0);

    // Fill B = Identity (or rectangular "identity-like": ones on min(k,n))
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < k; ++row) {
            h_B[(size_t)row + (size_t)col * (size_t)k] = (row == col) ? 1.0 : 0.0;
        }
    }

    // --- Device pointers ---
    index_type *d_rowPtr = nullptr, *d_colInd = nullptr;
    data_type  *d_vals   = nullptr;
    data_type  *d_B      = nullptr;
    data_type  *d_C      = nullptr;

    const size_t bytesRowPtr = (size_t)(m + 1) * sizeof(index_type);
    const size_t bytesColInd = (size_t)nnz * sizeof(index_type);
    const size_t bytesVals   = (size_t)nnz * sizeof(data_type);
    const size_t bytesB      = (size_t)k * (size_t)n * sizeof(data_type);
    const size_t bytesC      = (size_t)m * (size_t)n * sizeof(data_type);

    // Time GPU allocations (host wall time is fine here; allocation is synchronous)
    double t_cuda_alloc = wtime();
    CUDA_CHECK(cudaMalloc(&d_rowPtr, bytesRowPtr)); memlog.sample(wtime());
    CUDA_CHECK(cudaMalloc(&d_colInd, bytesColInd)); memlog.sample(wtime());
    CUDA_CHECK(cudaMalloc(&d_vals,   bytesVals));   memlog.sample(wtime());
    CUDA_CHECK(cudaMalloc(&d_B,      bytesB));      memlog.sample(wtime());
    CUDA_CHECK(cudaMalloc(&d_C,      bytesC));      memlog.sample(wtime());
    t_cuda_alloc = 1e3*(wtime() - t_cuda_alloc);

    memlog.sample(wtime()); // after device allocs

    // --- H2D transfers (timed with CUDA events on the same stream) ---
    memlog.sample(wtime());
    float t_h2d_rowptr_ms = timed_memcpy_async(d_rowPtr, h_rowPtr.data(), bytesRowPtr, cudaMemcpyHostToDevice, stream);
    memlog.sample(wtime());
    float t_h2d_colind_ms = timed_memcpy_async(d_colInd, h_colInd.data(), bytesColInd, cudaMemcpyHostToDevice, stream);
    memlog.sample(wtime());
    float t_h2d_vals_ms   = timed_memcpy_async(d_vals,   h_vals.data(),   bytesVals,   cudaMemcpyHostToDevice, stream);
    memlog.sample(wtime());
    float t_h2d_B_ms      = timed_memcpy_async(d_B,      h_B.data(),      bytesB,      cudaMemcpyHostToDevice, stream);
    memlog.sample(wtime());

    // Initialize d_C (either memset or copy zeros from host; memset is usually faster)
    cudaEvent_t z0, z1;
    CUDA_CHECK(cudaEventCreate(&z0));
    CUDA_CHECK(cudaEventCreate(&z1));
    CUDA_CHECK(cudaEventRecord(z0, stream));
    CUDA_CHECK(cudaMemsetAsync(d_C, 0, bytesC, stream));
    CUDA_CHECK(cudaEventRecord(z1, stream));
    CUDA_CHECK(cudaEventSynchronize(z1));
    float t_zero_C_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&t_zero_C_ms, z0, z1));
    CUDA_CHECK(cudaEventDestroy(z0));
    CUDA_CHECK(cudaEventDestroy(z1));

    memlog.sample(wtime()); // after H2D + zero

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

    // Time workspace alloc
    double t_ws_alloc = wtime();
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    t_ws_alloc = 1e3*(wtime() - t_ws_alloc);

    memlog.sample(wtime()); // after workspace alloc

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    const int spike_period = 250;
    std::mt19937 rng(12345);
    std::uniform_int_distribution<size_t> dist(0, (size_t)nnz - 1);
    const size_t rowptr_sz = (size_t)m + 1;
    std::uniform_int_distribution<size_t> dist_rowptr(0, rowptr_sz - 1);

    double t_compute_total = 0.0;

    for (int it = 0; it < iters; ++it) {
        memlog.sample(wtime());

        // NOTE: In the UM version, "spike" touched managed memory (triggering migrations/page-faults).
        // Here, d_* are device pointers; host cannot legally dereference them.
        // To preserve a similar effect (extra traffic), we do a small D2H sampling of device data.
        if (it > 0 && (it % spike_period == 0) && it <= 1000) {
            int block = it / spike_period;     // 1..4
            double frac = 0.25 * block;        // 0.25,0.5,0.75,1.0
            size_t read_elems = (size_t)(frac * (double)nnz);
            if (read_elems > (size_t)nnz) read_elems = (size_t)nnz;

            // Sample a subset of d_vals/d_colInd/d_rowPtr back to host to create bandwidth/latency "spikes".
            // We keep it light: copy single elements many times (inefficient but similar to your random touches).
            volatile double sink = 0.0;
            index_type tmp_i = 0;
            data_type  tmp_v = 0.0;

            for (size_t i = 0; i < read_elems; ++i) {
                size_t idx = dist(rng);
                size_t rp  = dist_rowptr(rng);

                // copy 1 value of each array (sync per copy due to timed helper; for spike we can do async + sync occasionally)
                CUDA_CHECK(cudaMemcpyAsync(&tmp_v, d_vals + idx, sizeof(data_type), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(&tmp_i, d_colInd + idx, sizeof(index_type), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(&tmp_i, d_rowPtr + rp, sizeof(index_type), cudaMemcpyDeviceToHost, stream));

                if (i % 256 == 0) {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    memlog.sample(wtime());
                }

                sink += (double)tmp_v + (double)tmp_i;
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
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

    memlog.sample(wtime());
    float t_d2h_C_ms = timed_memcpy_async(h_C.data(), d_C, bytesC, cudaMemcpyDeviceToHost, stream);
    memlog.sample(wtime());

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    const double t_compute_avg = t_compute_total / (double)iters;
    const double t_end2end_s = wtime() - t_end2end;

    std::printf("t_cuda_alloc     = %.3f ms\n", t_cuda_alloc);
    std::printf("t_ws_alloc       = %.3f ms\n", t_ws_alloc);

    std::printf("t_h2d_rowptr     = %.3f ms\n", t_h2d_rowptr_ms);
    std::printf("t_h2d_colind     = %.3f ms\n", t_h2d_colind_ms);
    std::printf("t_h2d_vals       = %.3f ms\n", t_h2d_vals_ms);
    std::printf("t_h2d_B          = %.3f ms\n", t_h2d_B_ms);
    std::printf("t_zero_d_C       = %.3f ms\n", t_zero_C_ms);
    std::printf("t_d2h_C          = %.3f ms\n", t_d2h_C_ms);

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

