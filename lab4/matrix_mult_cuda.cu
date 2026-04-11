#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

using namespace std;
using namespace chrono;

#define TILE_DIM 16

using MatrixType = vector<vector<double>>;

MatrixType read_matrix_file(const string& path, int& dim) {
    ifstream src(path);
    src >> dim;
    MatrixType mat(dim, vector<double>(dim));
    for (int r = 0; r < dim; ++r)
        for (int c = 0; c < dim; ++c)
            src >> mat[r][c];
    return mat;
}

void write_matrix_file(const string& path, const MatrixType& mat) {
    ofstream dst(path);
    int n = mat.size();
    dst << n << "\n";
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c)
            dst << fixed << setprecision(6) << mat[r][c] << " ";
        dst << "\n";
    }
}

void append_result(const string& path, int dim, double cpu_ms, double gpu_ms, double spdup, int tile_sz) {
    ofstream out(path, ios::app);
    out << dim << "," << tile_sz << "," << cpu_ms << "," << gpu_ms << "," << spdup << "\n";
    out.close();
}

MatrixType make_random_square(int n) {
    MatrixType mat(n, vector<double>(n));
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            mat[r][c] = (rand() % 10) + 1;
    return mat;
}

MatrixType serial_product(const MatrixType& P, const MatrixType& Q) {
    int n = P.size();
    MatrixType R(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                R[i][j] += P[i][k] * Q[k][j];
    return R;
}

__global__ void cuda_matmul(double* A, double* B, double* C, int dim) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < dim && col < dim) {
        double total = 0.0;
        for (int k = 0; k < dim; ++k) {
            total += A[row * dim + k] * B[k * dim + col];
        }
        C[row * dim + col] = total;
    }
}

MatrixType gpu_product(const MatrixType& P, const MatrixType& Q, int tile_sz) {
    int n = P.size();
    MatrixType R(n, vector<double>(n, 0.0));
    
    size_t mem_bytes = n * n * sizeof(double);
    
    double *devP, *devQ, *devR;
    cudaMalloc(&devP, mem_bytes);
    cudaMalloc(&devQ, mem_bytes);
    cudaMalloc(&devR, mem_bytes);
    
    vector<double> flatP(n * n);
    vector<double> flatQ(n * n);
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            flatP[r * n + c] = P[r][c];
            flatQ[r * n + c] = Q[r][c];
        }
    }
    
    cudaMemcpy(devP, flatP.data(), mem_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devQ, flatQ.data(), mem_bytes, cudaMemcpyHostToDevice);
    
    dim3 threads_per_block(tile_sz, tile_sz);
    dim3 grid_blocks((n + tile_sz - 1) / tile_sz, (n + tile_sz - 1) / tile_sz);
    
    cuda_matmul<<<grid_blocks, threads_per_block>>>(devP, devQ, devR, n);
    cudaDeviceSynchronize();
    
    vector<double> flatR(n * n);
    cudaMemcpy(flatR.data(), devR, mem_bytes, cudaMemcpyDeviceToHost);
    
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            R[r][c] = flatR[r * n + c];
    
    cudaFree(devP);
    cudaFree(devQ);
    cudaFree(devR);
    
    return R;
}

int main(int argc, char* argv[]) {
    if (argc == 5) {
        string pathA = argv[1], pathB = argv[2], pathOut = argv[3];
        int tile_dim = stoi(argv[4]);
        
        int N;
        MatrixType A = read_matrix_file(pathA, N);
        MatrixType B = read_matrix_file(pathB, N);
        
        auto t_start = high_resolution_clock::now();
        MatrixType C = gpu_product(A, B, tile_dim);
        auto t_end = high_resolution_clock::now();
        double gpu_dur = duration_cast<microseconds>(t_end - t_start).count() / 1000.0;
        
        write_matrix_file(pathOut, C);
        
        cout << "Matrix dimension: " << N << "x" << N << endl;
        cout << "Tile configuration: " << tile_dim << "x" << tile_dim << endl;
        cout << "GPU runtime: " << gpu_dur << " ms" << endl;
        
        return 0;
    }
    
    if (argc == 3 && string(argv[1]) == "--bench") {
        int N = stoi(argv[2]);
        srand(time(nullptr));
        
        MatrixType A = make_random_square(N);
        MatrixType B = make_random_square(N);
        
        cout << "\n>>> CUDA Performance Test <<<" << endl;
        cout << "Dimension: " << N << "x" << N << endl;
        
        auto t0 = high_resolution_clock::now();
        MatrixType C_cpu = serial_product(A, B);
        auto t1 = high_resolution_clock::now();
        double cpu_dur = duration_cast<microseconds>(t1 - t0).count() / 1000.0;
        
        vector<int> tile_options = {8, 16, 32};
        
        cout << "\nCPU baseline: " << cpu_dur << " ms" << endl;
        cout << "\nTileSize | GPU Time (ms) | Speedup" << endl;
        cout << "---------|---------------|---------" << endl;
        
        for (int ts : tile_options) {
            cudaEvent_t ev_start, ev_stop;
            cudaEventCreate(&ev_start);
            cudaEventCreate(&ev_stop);
            
            size_t mem_bytes = N * N * sizeof(double);
            double *devA, *devB, *devC;
            cudaMalloc(&devA, mem_bytes);
            cudaMalloc(&devB, mem_bytes);
            cudaMalloc(&devC, mem_bytes);
            
            vector<double> flatA(N * N);
            vector<double> flatB(N * N);
            for (int r = 0; r < N; ++r) {
                for (int c = 0; c < N; ++c) {
                    flatA[r * N + c] = A[r][c];
                    flatB[r * N + c] = B[r][c];
                }
            }
            
            cudaMemcpy(devA, flatA.data(), mem_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(devB, flatB.data(), mem_bytes, cudaMemcpyHostToDevice);
            
            dim3 threads_per_block(ts, ts);
            dim3 grid_blocks((N + ts - 1) / ts, (N + ts - 1) / ts);
            
            cudaEventRecord(ev_start);
            cuda_matmul<<<grid_blocks, threads_per_block>>>(devA, devB, devC, N);
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            
            float gpu_dur_ms;
            cudaEventElapsedTime(&gpu_dur_ms, ev_start, ev_stop);
            
            cudaFree(devA);
            cudaFree(devB);
            cudaFree(devC);
            
            double spdup = cpu_dur / gpu_dur_ms;
            
            cout << "   " << ts << "x" << ts << "     |     " << gpu_dur_ms << "     |   " << spdup << "x" << endl;
            
            if (N <= 100 && ts == 16) {
                write_matrix_file("verify_A.txt", A);
                write_matrix_file("verify_B.txt", B);
                
                vector<double> flatC(N * N);
                cudaMemcpy(flatC.data(), devC, mem_bytes, cudaMemcpyDeviceToHost);
                MatrixType C_gpu(N, vector<double>(N));
                for (int r = 0; r < N; ++r)
                    for (int c = 0; c < N; ++c)
                        C_gpu[r][c] = flatC[r * N + c];
                write_matrix_file("verify_C.txt", C_gpu);
            }
        }
        
        return 0;
    }
    
    cerr << "Invocation:" << endl;
    cerr << "  " << argv[0] << " --bench N" << endl;
    cerr << "  " << argv[0] << " A.txt B.txt out.txt tile_size" << endl;
    return 1;
}