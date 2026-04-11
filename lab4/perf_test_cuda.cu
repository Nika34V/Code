#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

using namespace std;

#define TILE_DIM 16

vector<vector<double>> create_random_matrix(int n) {
    vector<vector<double>> mat(n, vector<double>(n));
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            mat[r][c] = (rand() % 10) + 1;
    return mat;
}

vector<vector<double>> cpu_multiply(const vector<vector<double>>& U, const vector<vector<double>>& V) {
    int n = U.size();
    vector<vector<double>> W(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                W[i][j] += U[i][k] * V[k][j];
    return W;
}

__global__ void matmul_kernel(double* A, double* B, double* C, int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < dim && col < dim) {
        double accum = 0.0;
        for (int k = 0; k < dim; ++k) {
            accum += A[row * dim + k] * B[k * dim + col];
        }
        C[row * dim + col] = accum;
    }
}

int main() {
    vector<int> dimensions = {200, 400, 800, 1200, 1600, 2000};
    vector<int> tile_configs = {8, 16, 32};
    
    cout << "Dimension,TileSize,CPU_Time_ms,GPU_Time_ms,Speedup" << endl;
    
    for (int N : dimensions) {
        srand(42);
        auto A = create_random_matrix(N);
        auto B = create_random_matrix(N);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        auto C_cpu = cpu_multiply(A, B);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cpu_dur = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
        
        for (int ts : tile_configs) {
            size_t mem_sz = N * N * sizeof(double);
            double *devA, *devB, *devC;
            cudaMalloc(&devA, mem_sz);
            cudaMalloc(&devB, mem_sz);
            cudaMalloc(&devC, mem_sz);
            
            vector<double> flatA(N * N);
            vector<double> flatB(N * N);
            for (int r = 0; r < N; ++r) {
                for (int c = 0; c < N; ++c) {
                    flatA[r * N + c] = A[r][c];
                    flatB[r * N + c] = B[r][c];
                }
            }
            
            cudaMemcpy(devA, flatA.data(), mem_sz, cudaMemcpyHostToDevice);
            cudaMemcpy(devB, flatB.data(), mem_sz, cudaMemcpyHostToDevice);
            
            dim3 threads_per_block(ts, ts);
            dim3 grid_blocks((N + ts - 1) / ts, (N + ts - 1) / ts);
            
            cudaEvent_t ev_start, ev_stop;
            cudaEventCreate(&ev_start);
            cudaEventCreate(&ev_stop);
            
            cudaEventRecord(ev_start);
            matmul_kernel<<<grid_blocks, threads_per_block>>>(devA, devB, devC, N);
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            
            float gpu_dur_ms;
            cudaEventElapsedTime(&gpu_dur_ms, ev_start, ev_stop);
            
            cudaFree(devA);
            cudaFree(devB);
            cudaFree(devC);
            
            double spdup = cpu_dur / gpu_dur_ms;
            
            cout << N << "," << ts << "," << cpu_dur << "," << gpu_dur_ms << "," << spdup << endl;
        }
    }
    
    return 0;
}