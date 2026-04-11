#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <ctime>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std::chrono;
using MatrixType = std::vector<std::vector<double>>;

MatrixType load_matrix(const std::string& path, int& dimension) {
    std::ifstream inp(path);
    inp >> dimension;
    MatrixType mat(dimension, std::vector<double>(dimension));
    for (int r = 0; r < dimension; ++r)
        for (int c = 0; c < dimension; ++c)
            inp >> mat[r][c];
    return mat;
}

void save_matrix(const std::string& path, const MatrixType& mat) {
    std::ofstream out(path);
    int n = mat.size();
    out << n << "\n";
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c)
            out << std::fixed << std::setprecision(6) << mat[r][c] << " ";
        out << "\n";
    }
}

MatrixType parallel_multiply(const MatrixType& A, const MatrixType& B, int num_threads) {
    int n = A.size();
    MatrixType C(n, std::vector<double>(n, 0.0));
    
#ifdef _OPENMP
    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double total = 0.0;
            for (int k = 0; k < n; ++k) {
                total += A[i][k] * B[k][j];
            }
            C[i][j] = total;
        }
    }
    return C;
}

MatrixType random_matrix(int n) {
    MatrixType mat(n, std::vector<double>(n));
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            mat[r][c] = (rand() % 10) + 1;
    return mat;
}

int main(int argc, char* argv[]) {
    if (argc != 5 && argc != 2) {
        std::cerr << "Usage:\n  " << argv[0] << " A.txt B.txt out.txt threads\n";
        std::cerr << "  " << argv[0] << " -s N\n";
        return 1;
    }
    
    if (std::string(argv[1]) == "-s") {
        int N = std::stoi(argv[2]);
        std::srand(std::time(nullptr));
        MatrixType X = random_matrix(N);
        MatrixType Y = random_matrix(N);
        
#ifdef _OPENMP
        int max_t = omp_get_max_threads();
        std::cout << "Max threads: " << max_t << "\n";
        
        for (int thr : {1, 2, 4, 8}) {
            if (thr > max_t) continue;
#else
        {
            int thr = 1;
#endif
            auto start = high_resolution_clock::now();
            MatrixType Z = parallel_multiply(X, Y, thr);
            auto end = high_resolution_clock::now();
            auto elapsed = duration_cast<microseconds>(end - start);
            
            long long flops = 2LL * N * N * N;
            double gflops_val = flops / (elapsed.count() / 1e6) / 1e9;
            
            std::cout << "Threads=" << thr << " Time=" << elapsed.count()
                      << " us GFLOPS=" << std::fixed << std::setprecision(2) << gflops_val << "\n";
#ifdef _OPENMP
        }
#endif
        return 0;
    }
    
    std::string pathA = argv[1], pathB = argv[2], pathOut = argv[3];
    int worker_threads = std::stoi(argv[4]);
    
    int dimA, dimB;
    MatrixType A = load_matrix(pathA, dimA);
    MatrixType B = load_matrix(pathB, dimB);
    
    if (dimA != dimB) {
        std::cerr << "Error: matrix dimensions differ\n";
        return 1;
    }
    
    auto t_start = high_resolution_clock::now();
    MatrixType C = parallel_multiply(A, B, worker_threads);
    auto t_end = high_resolution_clock::now();
    auto runtime = duration_cast<microseconds>(t_end - t_start);
    
    save_matrix(pathOut, C);
    
    long long total_flops = 2LL * dimA * dimA * dimA;
    std::cout << "Matrix size: " << dimA << "x" << dimA << "\n";
    std::cout << "Thread count: " << worker_threads << "\n";
    std::cout << "Runtime: " << runtime.count() << " us ("
              << runtime.count() / 1000.0 << " ms)\n";
    std::cout << "Arithmetic ops: " << total_flops << "\n";
    
    return 0;
}