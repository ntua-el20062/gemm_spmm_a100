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
#include <cstdio>
#include <unistd.h>

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
CSR<T> dense_to_csr(const T* C, int m, int n, int ld, double tol)
{
    CSR<T> out;
    out.m = m;
    out.n = n;

    out.row_ptr.assign(m + 1, 0);
    std::vector<int> I;
    std::vector<int> J;
    std::vector<T>   V;

    I.reserve(m * 4); // arbitrary guess; will grow if needed
    J.reserve(m * 4);
    V.reserve(m * 4);

    // scan row by row
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            // column-major: C(i,j) = C[i + j*ld]
            T val = C[(size_t)i + (size_t)j * (size_t)ld];
            if (std::fabs((double)val) > tol) {
                I.push_back(i);
                J.push_back(j);
                V.push_back(val);
            }
        }
    }

    int nnz = (int)I.size();
    out.nnz = nnz;
    out.col_idx.resize(nnz);
    out.vals.resize(nnz);

    // build row_ptr (like in read_matrix_market_to_csr)
    for (int k = 0; k < nnz; ++k) {
        out.row_ptr[I[k] + 1]++;
    }
    for (int r = 0; r < m; ++r) {
        out.row_ptr[r + 1] += out.row_ptr[r];
    }

    std::vector<int> offset = out.row_ptr;
    for (int k = 0; k < nnz; ++k) {
        int row  = I[k];
        int dest = offset[row]++;
        out.col_idx[dest] = J[k];
        out.vals[dest]    = V[k];
    }

    return out;
}

template<typename T>
bool csr_equal(const CSR<T>& A, const CSR<T>& B, double tol)
{
    if (A.m != B.m || A.n != B.n || A.nnz != B.nnz) {
        std::cerr << "CSR mismatch: dims or nnz differ\n";
        return false;
    }

    if (A.row_ptr.size() != B.row_ptr.size() ||
        A.col_idx.size() != B.col_idx.size() ||
        A.vals.size()    != B.vals.size()) {
        std::cerr << "CSR mismatch: internal array sizes differ\n";
        return false;
    }

    for (size_t i = 0; i < A.row_ptr.size(); ++i) {
        if (A.row_ptr[i] != B.row_ptr[i]) {
            std::cerr << "CSR mismatch: row_ptr[" << i << "] "
                      << A.row_ptr[i] << " vs " << B.row_ptr[i] << "\n";
            return false;
        }
    }

    for (size_t k = 0; k < A.col_idx.size(); ++k) {
        if (A.col_idx[k] != B.col_idx[k]) {
            std::cerr << "CSR mismatch: col_idx[" << k << "] "
                      << A.col_idx[k] << " vs " << B.col_idx[k] << "\n";
            return false;
        }
    }

    for (size_t k = 0; k < A.vals.size(); ++k) {
        double av = (double)A.vals[k];
        double bv = (double)B.vals[k];
        double diff = std::fabs(av - bv);
        if (diff > tol) {
            std::cerr << "CSR mismatch: vals[" << k << "] "
                      << av << " vs " << bv << " (diff=" << diff << ")\n";
            return false;
        }
    }

    return true;
}

template<typename T>
void build_csr_row_tile(
    const CSR<T>& A,
    int row0,
    int row1, // exclusive
    std::vector<int>& row_ptr_tile,
    std::vector<int>& col_idx_tile,
    std::vector<T>&   vals_tile)
{
    int m_tile = row1 - row0;
    int nz0 = A.row_ptr[row0];
    int nz1 = A.row_ptr[row1];
    int nnz_tile = nz1 - nz0;

    row_ptr_tile.resize(m_tile + 1);
    col_idx_tile.resize(nnz_tile);
    vals_tile.resize(nnz_tile);

    // row_ptr: shift by nz0 so that row_ptr_tile[0] = 0
    row_ptr_tile[0] = 0;
    for (int r = 0; r < m_tile; ++r) {
        int global_r = row0 + r;
        row_ptr_tile[r + 1] = A.row_ptr[global_r + 1] - nz0;
    }

    // col_idx and vals: contiguous slice
    std::memcpy(col_idx_tile.data(),
                A.col_idx.data() + nz0,
                sizeof(int) * nnz_tile);
    std::memcpy(vals_tile.data(),
                A.vals.data() + nz0,
                sizeof(T) * nnz_tile);
}


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


    
    //for (int j = 0; j < K; ++j) {
    //  for (int i = 0; i < n; ++i) {
    //    h_X[(size_t)i + (size_t)j * (size_t)n] = (i == j) ? (T)1 : (T)0;
    //  }
    //}

    
    /*    std::cout << "Building CSR from dense C and comparing with original CSR...\n";

        // C is m x K (with K==n), column-major, leading dimension = m
        double tol = std::is_same<T,float>::value ? 1e-4 : 1e-10;

        CSR<T> csr_from_C = dense_to_csr(h_C, m, n, m, tol);

        bool same = csr_equal(csr, csr_from_C, tol);
        if (same) {
            std::cout << "CSR self-check: PASSED (A*I == A, CSR identical)\n";
        } else {
            std::cout << "CSR self-check: FAILED\n";
        }
    */

// Convert CSR to dense column-major matrix (ld = m)
template<typename T>
std::vector<T> csr_to_dense_colmajor(const CSR<T>& A)
{
    int m = A.m;
    int n = A.n;
    std::vector<T> D((size_t)m * (size_t)n, (T)0);

    for (int i = 0; i < m; ++i) {
        int row_start = A.row_ptr[i];
        int row_end   = A.row_ptr[i + 1];
        for (int idx = row_start; idx < row_end; ++idx) {
            int j = A.col_idx[idx];   // column
            T   v = A.vals[idx];
            // column-major: D(i,j) = D[i + j*m]
            D[(size_t)i + (size_t)j * (size_t)m] = v;
        }
    }
    return D;
}

// Print dense matrix (column-major, ld = m)
template<typename T>
void print_dense_colmajor(const T* M, int m, int n, int ld, const char* name)
{
    std::cout << "\n==== " << name << " (m=" << m << ", n=" << n << ") ====\n";
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double v = (double)M[(size_t)i + (size_t)j * (size_t)ld];
            // high precision so you can see exact differences
            std::cout << std::scientific << v;
            if (j + 1 < n) std::cout << " ";
        }
        std::cout << "\n";
    }
    std::cout << "==== END " << name << " ====\n\n";
}

template<typename T>
bool csr_equal_sorted(const CSR<T>& A, const CSR<T>& B, double tol)
{
    if (A.m != B.m || A.n != B.n) {
        std::cerr << "Dimension mismatch\n";
        return false;
    }
    if (A.nnz != B.nnz) {
        std::cerr << "NNZ mismatch: " << A.nnz << " vs " << B.nnz << "\n";
        return false;
    }

    for (int r = 0; r < A.m; r++) {
        int a0 = A.row_ptr[r];
        int a1 = A.row_ptr[r+1];
        int b0 = B.row_ptr[r];
        int b1 = B.row_ptr[r+1];

        int na = a1 - a0;
        int nb = b1 - b0;

        if (na != nb) {
            std::cerr << "Row " << r << " nnz mismatch: "
                      << na << " vs " << nb << "\n";
            return false;
        }

        // Build row arrays
        std::vector<std::pair<int,T>> rowA, rowB;
        rowA.reserve(na);
        rowB.reserve(nb);

        for (int k = 0; k < na; k++) {
            rowA.emplace_back(A.col_idx[a0+k], A.vals[a0+k]);
            rowB.emplace_back(B.col_idx[b0+k], B.vals[b0+k]);
        }

        // Sort by column index
        auto cmp = [](auto &x, auto &y){ return x.first < y.first; };
        std::sort(rowA.begin(), rowA.end(), cmp);
        std::sort(rowB.begin(), rowB.end(), cmp);

        // Compare
        for (int k = 0; k < na; k++) {
            if (rowA[k].first != rowB[k].first) {
                std::cerr << "Row " << r << " col mismatch at k=" << k
                          << ": " << rowA[k].first
                          << " vs " << rowB[k].first << "\n";
                return false;
            }
            double diff = std::fabs((double)rowA[k].second -
                                    (double)rowB[k].second);
            if (diff > tol) {
                std::cerr << "Row " << r << " val mismatch at col "
                          << rowA[k].first << ": "
                          << rowA[k].second << " vs " << rowB[k].second
                          << " diff=" << diff << "\n";
                return false;
            }
        }
    }

    return true;
}

template<typename T>
void print_csr_diff(const CSR<T>& A, const CSR<T>& B, double tol)
{
    std::cout << "\n======= CSR DIFF =======\n";

    if (A.m != B.m || A.n != B.n) {
        std::cout << "Dimension mismatch: "
                  << A.m << "x" << A.n << " vs "
                  << B.m << "x" << B.n << "\n";
    }

    if (A.nnz != B.nnz) {
        std::cout << "NNZ mismatch: A.nnz=" << A.nnz
                  << " B.nnz=" << B.nnz << "\n";
    }

    // Compare row_ptr
    std::cout << "\nRow_ptr differences:\n";
    bool any_rowptr = false;
    for (int i = 0; i < (int)A.row_ptr.size(); i++) {
        if (A.row_ptr[i] != B.row_ptr[i]) {
            std::cout << " row_ptr[" << i << "] "
                      << A.row_ptr[i] << " vs "
                      << B.row_ptr[i] << "\n";
            any_rowptr = true;
        }
    }
    if (!any_rowptr) std::cout << " row_ptr identical.\n";

    // Compare column indices
    std::cout << "\nColumn index differences:\n";
    bool any_col = false;
    int min_nnz = std::min(A.nnz, B.nnz);
    for (int k = 0; k < min_nnz; k++) {
        if (A.col_idx[k] != B.col_idx[k]) {
            std::cout << " col_idx[" << k << "] "
                      << A.col_idx[k] << " vs "
                      << B.col_idx[k] << "\n";
            any_col = true;
        }
    }
    if (!any_col) std::cout << " col_idx identical (for min nnz).\n";

    // Compare values
    std::cout << "\nValue differences (diff > tol=" << tol << "):\n";
    bool any_vals = false;
    for (int k = 0; k < min_nnz; k++) {
        double av = (double)A.vals[k];
        double bv = (double)B.vals[k];
        if (std::fabs(av - bv) > tol) {
            std::cout << " vals[" << k << "] "
                      << av << " vs " << bv
                      << " (diff=" << std::fabs(av - bv) << ")\n";
            any_vals = true;
        }
    }
    if (!any_vals) std::cout << " vals identical (within tol).\n";

    std::cout << "======= END CSR DIFF =======\n\n";
}

template<typename T>
SpmmTimings run_spmm(const CSR<T> &csr, int K, int tile_rows, int tile_K, bool use_tf32_math)
{
    SpmmTimings Tm{};

    int m   = csr.m;
    int n   = csr.n;
    int nnz = csr.nnz;

    if (m == 0 || n == 0 || K == 0) return Tm;

    // ---------- Host dense matrices ----------
    double t_cpu_start = wtime();
    T *h_X, *h_C;
    CUDA_CHECK(cudaMallocHost(&h_X, sizeof(T) * n * K));
    CUDA_CHECK(cudaMallocHost(&h_C, sizeof(T) * m * K));
std::fill(h_C, h_C + (size_t)m * (size_t)K, (T)0);  // or memset to 0

    // Identity: X = I (column-major)
    for (int j = 0; j < K; ++j) {
        for (int i = 0; i < n; ++i) {
            h_X[(size_t)i + (size_t)j * (size_t)n] = (i == j) ? (T)1 : (T)0;
        }
    }

    print_memory();
    Tm.t_cpu_alloc = 1e3 * (wtime() - t_cpu_start);

    // ---------- cuSPARSE handle & streams ----------
    double t_gpu_start = wtime();

    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    cudaStream_t stream_h2d, stream_compute, stream_d2h;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_h2d, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_d2h, cudaStreamNonBlocking));
    CUSPARSE_CHECK(cusparseSetStream(handle, stream_compute));

    cudaDataType dt = getCudaType<T>();

    // ---------- Dense double buffers ----------
    int K_tile_max = tile_K;
    T* d_X_tile[2];
    T* d_C_tile[2];

    for (int b = 0; b < 2; b++) {
        CUDA_CHECK(cudaMalloc(&d_X_tile[b], sizeof(T) * n * K_tile_max));
        CUDA_CHECK(cudaMalloc(&d_C_tile[b], sizeof(T) * tile_rows * K_tile_max));
    }

    // ---------- CSR double buffers ----------
    int *d_row_ptr[2];
    int *d_col_idx[2];
    T   *d_vals[2];

    for (int b = 0; b < 2; b++) {
        CUDA_CHECK(cudaMalloc(&d_row_ptr[b], sizeof(int) * (tile_rows + 1)));
        CUDA_CHECK(cudaMalloc(&d_col_idx[b], sizeof(int) * nnz));
        CUDA_CHECK(cudaMalloc(&d_vals[b],    sizeof(T)   * nnz));
    }

    // Events to know when a CSR tile is ready
    cudaEvent_t ev_A_ready[2];
    CUDA_CHECK(cudaEventCreate(&ev_A_ready[0]));
    CUDA_CHECK(cudaEventCreate(&ev_A_ready[1]));

    Tm.t_gpu_alloc = 1e3 * (wtime() - t_gpu_start);

    // Host CSR tile buffers reused for all row tiles
    std::vector<int> h_row_ptr_tile;
    std::vector<int> h_col_idx_tile;
    std::vector<T>   h_vals_tile;

    int num_row_tiles = (m + tile_rows - 1) / tile_rows;

    // SpMM workspace (to be allocated once from a real matA)
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    bool buffer_allocated = false;

    T alpha = (T)1, beta = (T)0;

    // ---------- Prefetch CSR tile 0 into buffer 0 ----------
    if (num_row_tiles > 0) {
        int row0    = 0;
        int m_tile0 = std::min(tile_rows, m - row0);
        int nz0     = csr.row_ptr[row0];
        int nz1     = csr.row_ptr[row0 + m_tile0];
        int nnz_tile0 = nz1 - nz0;

        build_csr_row_tile(csr, row0, row0 + m_tile0,
                           h_row_ptr_tile, h_col_idx_tile, h_vals_tile);

        cudaEvent_t evs, eve;
        CUDA_CHECK(cudaEventCreate(&evs));
        CUDA_CHECK(cudaEventCreate(&eve));
        CUDA_CHECK(cudaEventRecord(evs, stream_h2d));

        CUDA_CHECK(cudaMemcpyAsync(d_row_ptr[0],
                                   h_row_ptr_tile.data(),
                                   sizeof(int) * (m_tile0 + 1),
                                   cudaMemcpyHostToDevice,
                                   stream_h2d));

        CUDA_CHECK(cudaMemcpyAsync(d_col_idx[0],
                                   h_col_idx_tile.data(),
                                   sizeof(int) * nnz_tile0,
                                   cudaMemcpyHostToDevice,
                                   stream_h2d));

        CUDA_CHECK(cudaMemcpyAsync(d_vals[0],
                                   h_vals_tile.data(),
                                   sizeof(T) * nnz_tile0,
                                   cudaMemcpyHostToDevice,
                                   stream_h2d));

        CUDA_CHECK(cudaEventRecord(eve, stream_h2d));
        CUDA_CHECK(cudaStreamSynchronize(stream_h2d));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, evs, eve));
        Tm.t_h2d_ms += ms;

        CUDA_CHECK(cudaEventDestroy(evs));
        CUDA_CHECK(cudaEventDestroy(eve));

        CUDA_CHECK(cudaEventRecord(ev_A_ready[0], stream_h2d));
    }

    // Events reused for dense tiles
    cudaEvent_t ev_X_ready[2];
    CUDA_CHECK(cudaEventCreate(&ev_X_ready[0]));
    CUDA_CHECK(cudaEventCreate(&ev_X_ready[1]));

    // ---------- Outer loop: row tiles ----------
    for (int rt = 0; rt < num_row_tiles; ++rt) {
        int row0    = rt * tile_rows;
        int m_tile  = std::min(tile_rows, m - row0);
        int nz0     = csr.row_ptr[row0];
        int nz1     = csr.row_ptr[row0 + m_tile];
        int nnz_tile = nz1 - nz0;

        int buf      = (rt & 1);          // CSR buffer used for compute
        int next_buf = ((rt + 1) & 1);    // CSR buffer for prefetch

        // Prefetch CSR tile (rt+1) into next_buf
        if (rt + 1 < num_row_tiles) {
            int row0_next    = (rt + 1) * tile_rows;
            int m_tile_next  = std::min(tile_rows, m - row0_next);
            int nz0_next     = csr.row_ptr[row0_next];
            int nz1_next     = csr.row_ptr[row0_next + m_tile_next];
            int nnz_tile_next = nz1_next - nz0_next;

            build_csr_row_tile(csr, row0_next, row0_next + m_tile_next,
                               h_row_ptr_tile, h_col_idx_tile, h_vals_tile);

            cudaEvent_t evs, eve;
            CUDA_CHECK(cudaEventCreate(&evs));
            CUDA_CHECK(cudaEventCreate(&eve));
            CUDA_CHECK(cudaEventRecord(evs, stream_h2d));

            CUDA_CHECK(cudaMemcpyAsync(d_row_ptr[next_buf],
                                       h_row_ptr_tile.data(),
                                       sizeof(int) * (m_tile_next + 1),
                                       cudaMemcpyHostToDevice,
                                       stream_h2d));

            CUDA_CHECK(cudaMemcpyAsync(d_col_idx[next_buf],
                                       h_col_idx_tile.data(),
                                       sizeof(int) * nnz_tile_next,
                                       cudaMemcpyHostToDevice,
                                       stream_h2d));

            CUDA_CHECK(cudaMemcpyAsync(d_vals[next_buf],
                                       h_vals_tile.data(),
                                       sizeof(T) * nnz_tile_next,
                                       cudaMemcpyHostToDevice,
                                       stream_h2d));

            CUDA_CHECK(cudaEventRecord(eve, stream_h2d));
            CUDA_CHECK(cudaEventRecord(ev_A_ready[next_buf], stream_h2d));
        }

        // Wait for CSR tile rt to be ready
        CUDA_CHECK(cudaStreamWaitEvent(stream_compute, ev_A_ready[buf], 0));

        // Create SpMat for this row tile
        cusparseSpMatDescr_t matA;
        CUSPARSE_CHECK(cusparseCreateCsr(&matA,
                                         m_tile, n, nnz_tile,
                                         d_row_ptr[buf],
                                         d_col_idx[buf],
                                         d_vals[buf],
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         dt));

        // --------- NEW: allocate SpMM workspace from real matA, once ---------
        if (!buffer_allocated) {
            double t_buf = wtime();

            // Use "worst-case" tile width for buffer sizing
            int this_K_buf = std::min(tile_K, K);

            cusparseDnMatDescr_t matX_tmp, matC_tmp;
            CUSPARSE_CHECK(cusparseCreateDnMat(&matX_tmp,
                                               n, this_K_buf, n,
                                               d_X_tile[0], dt, CUSPARSE_ORDER_COL));
            CUSPARSE_CHECK(cusparseCreateDnMat(&matC_tmp,
                                               m_tile, this_K_buf, m_tile,
                                               d_C_tile[0], dt, CUSPARSE_ORDER_COL));

            CUSPARSE_CHECK(cusparseSpMM_bufferSize(handle,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                   &alpha, matA, matX_tmp, &beta, matC_tmp,
                                                   dt, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
            CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));
            buffer_allocated = true;

            CUSPARSE_CHECK(cusparseDestroyDnMat(matX_tmp));
            CUSPARSE_CHECK(cusparseDestroyDnMat(matC_tmp));

            Tm.t_gpu_alloc += 1e3 * (wtime() - t_buf);
        }
        // ---------------------------------------------------------------------

        // ---------- K-tiling pipeline for this row tile ----------
        std::vector<cudaEvent_t> h2d_start, h2d_stop;
        std::vector<cudaEvent_t> d2h_start, d2h_stop;
        std::vector<cudaEvent_t> spmm_start, spmm_stop;

        // First H2D of X tile
        {
            int this_K = std::min(tile_K, K);
            int bufX = 0;

            cudaEvent_t evs, eve;
            CUDA_CHECK(cudaEventCreate(&evs));
            CUDA_CHECK(cudaEventCreate(&eve));
            CUDA_CHECK(cudaEventRecord(evs, stream_h2d));

            CUDA_CHECK(cudaMemcpyAsync(d_X_tile[bufX],
                                       h_X,
                                       sizeof(T) * n * this_K,
                                       cudaMemcpyHostToDevice,
                                       stream_h2d));

            CUDA_CHECK(cudaEventRecord(eve, stream_h2d));
            h2d_start.push_back(evs);
            h2d_stop.push_back(eve);

            CUDA_CHECK(cudaEventRecord(ev_X_ready[0], stream_h2d));
        }

        int num_K_tiles = (K + tile_K - 1) / tile_K;
        int prev_this_K = 0;

        for (int t = 0; t < num_K_tiles; t++) {

            int k0      = t * tile_K;
            int this_K  = std::min(tile_K, K - k0);

            int bufX      = (t & 1);
            int prev_bufX = ((t - 1) & 1);
            int next_bufX = ((t + 1) & 1);

            // D2H(t-1): copy C tile rows back
            if (t > 0) {
                cudaEvent_t evs, eve;
                CUDA_CHECK(cudaEventCreate(&evs));
                CUDA_CHECK(cudaEventCreate(&eve));
                CUDA_CHECK(cudaEventRecord(evs, stream_d2h));

                int prev_k0 = (t - 1) * tile_K;

                CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, spmm_stop[t - 1], 0));

                for (int j = 0; j < prev_this_K; ++j) {
                    const T* src = d_C_tile[prev_bufX] + (size_t)j * m_tile;
                    T* dst = h_C + row0 + (size_t)(prev_k0 + j) * m;
                    CUDA_CHECK(cudaMemcpyAsync(dst,
                                               src,
                                               sizeof(T) * m_tile,
                                               cudaMemcpyDeviceToHost,
                                               stream_d2h));
                }

                CUDA_CHECK(cudaEventRecord(eve, stream_d2h));
                d2h_start.push_back(evs);
                d2h_stop.push_back(eve);
            }

            // SpMM(t)
            CUDA_CHECK(cudaStreamWaitEvent(stream_compute, ev_X_ready[bufX], 0));
            cudaEvent_t evs, eve;
            CUDA_CHECK(cudaEventCreate(&evs));
            CUDA_CHECK(cudaEventCreate(&eve));
            CUDA_CHECK(cudaEventRecord(evs, stream_compute));

            CUDA_CHECK(cudaMemsetAsync(d_C_tile[bufX],
                                       0,
                                       sizeof(T) * m_tile * this_K,
                                       stream_compute));

            cusparseDnMatDescr_t matX, matC;
            CUSPARSE_CHECK(cusparseCreateDnMat(&matX,
                                               n, this_K, n,
                                               d_X_tile[bufX],
                                               dt, CUSPARSE_ORDER_COL));
            CUSPARSE_CHECK(cusparseCreateDnMat(&matC,
                                               m_tile, this_K, m_tile,
                                               d_C_tile[bufX],
                                               dt, CUSPARSE_ORDER_COL));

            CUSPARSE_CHECK(cusparseSpMM(handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA, matX, &beta, matC,
                                        dt, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

            CUDA_CHECK(cudaEventRecord(eve, stream_compute));
            spmm_start.push_back(evs);
            spmm_stop.push_back(eve);

            CUSPARSE_CHECK(cusparseDestroyDnMat(matX));
            CUSPARSE_CHECK(cusparseDestroyDnMat(matC));

            // H2D(t+1): next X tile
            if (t + 1 < num_K_tiles) {
                int next_k0     = (t + 1) * tile_K;
                int next_this_K = std::min(tile_K, K - next_k0);

                cudaEvent_t evs2, eve2;
                CUDA_CHECK(cudaEventCreate(&evs2));
                CUDA_CHECK(cudaEventCreate(&eve2));
                CUDA_CHECK(cudaEventRecord(evs2, stream_h2d));

                CUDA_CHECK(cudaMemcpyAsync(d_X_tile[next_bufX],
                                           h_X + (size_t)next_k0 * n,
                                           sizeof(T) * n * next_this_K,
                                           cudaMemcpyHostToDevice,
                                           stream_h2d));

                CUDA_CHECK(cudaEventRecord(eve2, stream_h2d));
                h2d_start.push_back(evs2);
                h2d_stop.push_back(eve2);

                CUDA_CHECK(cudaEventRecord(ev_X_ready[next_bufX], stream_h2d));
            }

            prev_this_K = this_K;
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_compute));
        CUDA_CHECK(cudaStreamSynchronize(stream_h2d));
        CUDA_CHECK(cudaStreamSynchronize(stream_d2h));

        // D2H for last K-tile of this row tile
        {
            int last = num_K_tiles - 1;
            int bufX = (last & 1);
            int this_K = prev_this_K;

            cudaEvent_t evs, eve;
            CUDA_CHECK(cudaEventCreate(&evs));
            CUDA_CHECK(cudaEventCreate(&eve));
            CUDA_CHECK(cudaEventRecord(evs, stream_d2h));

            CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, spmm_stop[last], 0));

            int k0 = last * tile_K;

            for (int j = 0; j < this_K; ++j) {
                const T* src = d_C_tile[bufX] + (size_t)j * m_tile;
                T* dst = h_C + row0 + (size_t)(k0 + j) * m;
                CUDA_CHECK(cudaMemcpyAsync(dst,
                                           src,
                                           sizeof(T) * m_tile,
                                           cudaMemcpyDeviceToHost,
                                           stream_d2h));
            }

            CUDA_CHECK(cudaEventRecord(eve, stream_d2h));
            d2h_start.push_back(evs);
            d2h_stop.push_back(eve);
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_d2h));

        // Accumulate timings
        float ms;
        for (size_t i = 0; i < h2d_start.size(); i++) {
            CUDA_CHECK(cudaEventElapsedTime(&ms, h2d_start[i], h2d_stop[i]));
            Tm.t_h2d_ms += ms;
        }
        for (size_t i = 0; i < d2h_start.size(); i++) {
            CUDA_CHECK(cudaEventElapsedTime(&ms, d2h_start[i], d2h_stop[i]));
            Tm.t_d2h_ms += ms;
        }
        for (size_t i = 0; i < spmm_start.size(); i++) {
            CUDA_CHECK(cudaEventElapsedTime(&ms, spmm_start[i], spmm_stop[i]));
            Tm.t_spmm_ms += ms;
        }

        for (auto &e : h2d_start) CUDA_CHECK(cudaEventDestroy(e));
        for (auto &e : h2d_stop)  CUDA_CHECK(cudaEventDestroy(e));
        for (auto &e : d2h_start) CUDA_CHECK(cudaEventDestroy(e));
        for (auto &e : d2h_stop)  CUDA_CHECK(cudaEventDestroy(e));
        for (auto &e : spmm_start)CUDA_CHECK(cudaEventDestroy(e));
        for (auto &e : spmm_stop) CUDA_CHECK(cudaEventDestroy(e));

        CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    }

    // ---------- Self-check ----------
    std::cout << "Building CSR from dense C and comparing with original CSR...\n";

    double tol = std::is_same<T,float>::value ? 1e-4 : 1e-8;
    CSR<T> csr_from_C = dense_to_csr(h_C, m, K, m, tol);

    std::cout << "csr.nnz      = " << csr.nnz
              << ", csr_from_C.nnz = " << csr_from_C.nnz << "\n";

    bool same = csr_equal_sorted(csr, csr_from_C, tol);
    if (same) {
        std::cout << "CSR self-check: PASSED (A*I == A, CSR identical)\n";
 //       print_csr_diff(csr, csr_from_C, tol);

    } else {
        std::cout << "CSR self-check: FAILED\n";
   // print_csr_diff(csr, csr_from_C, tol);

    }

    // ---------- Cleanup ----------
    CUDA_CHECK(cudaEventDestroy(ev_X_ready[0]));
    CUDA_CHECK(cudaEventDestroy(ev_X_ready[1]));
    CUDA_CHECK(cudaEventDestroy(ev_A_ready[0]));
    CUDA_CHECK(cudaEventDestroy(ev_A_ready[1]));

    for (int b = 0; b < 2; b++) {
        CUDA_CHECK(cudaFree(d_X_tile[b]));
        CUDA_CHECK(cudaFree(d_C_tile[b]));
        CUDA_CHECK(cudaFree(d_row_ptr[b]));
        CUDA_CHECK(cudaFree(d_col_idx[b]));
        CUDA_CHECK(cudaFree(d_vals[b]));
    }

    if (dBuffer) {
        CUDA_CHECK(cudaFree(dBuffer));
    }

    CUSPARSE_CHECK(cusparseDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream_h2d));
    CUDA_CHECK(cudaStreamDestroy(stream_compute));
    CUDA_CHECK(cudaStreamDestroy(stream_d2h));

    CUDA_CHECK(cudaFreeHost(h_X));
    CUDA_CHECK(cudaFreeHost(h_C));

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
    int K         = (argc >= 3) ? std::atoi(argv[2]) : 64;
    int tile_rows = (argc >= 4) ? std::atoi(argv[3]) : 164;
    int tile_K    = (argc >= 5) ? std::atoi(argv[4]) : 64;
    std::string prec_str = (argc >= 6) ? argv[5] : "fp32";
    PrecMode mode = parse_precision(prec_str);

    double t0 = wtime();

    if (mode == PrecMode::FP64) {
        using T = double;
        CSR<T> csr = read_matrix_market_to_csr<T>(filename);
        SpmmTimings t = run_spmm<T>(csr, K, tile_rows, tile_K, /*use_tf32_math=*/false);
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
        SpmmTimings t = run_spmm<T>(csr, K,tile_rows, tile_K, use_tf32_math);
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

