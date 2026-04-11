#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <omp.h>

using namespace std::chrono;
using Matrix = std::vector<std::vector<double>>;

Matrix create_random_matrix(int n) {
    Matrix mat(n, std::vector<double>(n));
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col)
            mat[row][col] = (rand() % 10) + 1;
    return mat;
}

Matrix matmul_serial(const Matrix& A, const Matrix& B) {
    int n = A.size();
    Matrix C(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

Matrix matmul_parallel(const Matrix& A, const Matrix& B, int num_threads) {
    int n = A.size();
    Matrix C(n, std::vector<double>(n, 0.0));
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double accum = 0.0;
            for (int k = 0; k < n; ++k) {
                accum += A[i][k] * B[k][j];
            }
            C[i][j] = accum;
        }
    }
    return C;
}

int main() {
    std::vector<int> dims = {200, 400, 800, 1200, 1600, 2000};
    std::vector<int> thread_counts = {1, 2, 4, 8};
    int hardware_threads = omp_get_max_threads();
    
    std::cout << "Available threads: " << hardware_threads << "\n";
    
    std::ofstream out("perf_stats.csv");
    out << "Dimension,Threads,Time_us,Time_ms,FlopCount,GFLOPS,Speedup,Efficiency\n";
    
    for (int N : dims) {
        std::cout << "\n=== Dimension " << N << "x" << N << " ===\n";
        
        std::srand(42);
        Matrix A = create_random_matrix(N);
        Matrix B = create_random_matrix(N);
        
        long long total_flops = 2LL * N * N * N;
        double serial_time = 0.0;
        
        auto t0 = high_resolution_clock::now();
        Matrix C_serial = matmul_serial(A, B);
        auto t1 = high_resolution_clock::now();
        serial_time = duration_cast<microseconds>(t1 - t0).count();
        std::cout << "  Serial: " << serial_time / 1000.0 << " ms\n";
        
        for (int nth : thread_counts) {
            if (nth > hardware_threads) continue;
            
            const int repeat = 3;
            double total_dur = 0.0;
            
            for (int r = 0; r < repeat; ++r) {
                auto start = high_resolution_clock::now();
                Matrix C_par = matmul_parallel(A, B, nth);
                auto finish = high_resolution_clock::now();
                total_dur += duration_cast<microseconds>(finish - start).count();
            }
            
            double avg_dur = total_dur / repeat;
            double speedup_val = serial_time / avg_dur;
            double efficiency_val = speedup_val / nth;
            double gflops_val = total_flops / (avg_dur / 1e6) / 1e9;
            
            std::cout << "  Threads=" << nth << ": " << avg_dur / 1000.0
                      << " ms, speedup=" << std::fixed << std::setprecision(2) << speedup_val << "x\n";
            
            out << N << "," << nth << "," << avg_dur << "," << avg_dur/1000.0 << ","
                << total_flops << "," << gflops_val << "," << speedup_val << "," << efficiency_val << "\n";
        }
    }
    
    out.close();
    std::cout << "\nSaved: perf_stats.csv\n";
    return 0;
}