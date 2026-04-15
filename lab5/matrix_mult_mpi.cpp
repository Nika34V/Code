#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

using MatrixType = std::vector<std::vector<double>>;

MatrixType load_matrix_file(const std::string& path, int& dim) {
    std::ifstream src(path);
    src >> dim;
    MatrixType mat(dim, std::vector<double>(dim));
    for (int r = 0; r < dim; ++r)
        for (int c = 0; c < dim; ++c)
            src >> mat[r][c];
    return mat;
}

void save_matrix_file(const std::string& path, const MatrixType& mat) {
    std::ofstream dst(path);
    int n = mat.size();
    dst << n << "\n";
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c)
            dst << std::fixed << std::setprecision(6) << mat[r][c] << " ";
        dst << "\n";
    }
}

MatrixType make_random_square(int n) {
    MatrixType mat(n, std::vector<double>(n));
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c)
            mat[r][c] = (rand() % 10) + 1;
    return mat;
}

MatrixType sequential_product(const MatrixType& P, const MatrixType& Q) {
    int n = P.size();
    MatrixType R(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                R[i][j] += P[i][k] * Q[k][j];
    return R;
}

MatrixType mpi_product(const MatrixType& P, const MatrixType& Q, int my_id, int total_procs) {
    int n = P.size();
    int chunk_sz = n / total_procs;
    int leftover = n % total_procs;
    int my_chunk = chunk_sz + (my_id < leftover ? 1 : 0);
    
    int row_shift = 0;
    for (int p = 0; p < my_id; ++p)
        row_shift += chunk_sz + (p < leftover ? 1 : 0);
    
    MatrixType local_P(my_chunk, std::vector<double>(n));
    for (int r = 0; r < my_chunk; ++r)
        for (int c = 0; c < n; ++c)
            local_P[r][c] = P[row_shift + r][c];
    
    MatrixType local_R(my_chunk, std::vector<double>(n, 0.0));
    for (int r = 0; r < my_chunk; ++r) {
        for (int c = 0; c < n; ++c) {
            double acc = 0.0;
            for (int k = 0; k < n; ++k) {
                acc += local_P[r][k] * Q[k][c];
            }
            local_R[r][c] = acc;
        }
    }
    
    MatrixType final_mat(n, std::vector<double>(n));
    
    if (my_id == 0) {
        for (int r = 0; r < my_chunk; ++r)
            for (int c = 0; c < n; ++c)
                final_mat[row_shift + r][c] = local_R[r][c];
        
        int next_row = row_shift + my_chunk;
        for (int p = 1; p < total_procs; ++p) {
            int p_chunk = chunk_sz + (p < leftover ? 1 : 0);
            std::vector<double> inbox(p_chunk * n);
            MPI_Recv(inbox.data(), p_chunk * n, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            for (int r = 0; r < p_chunk; ++r)
                for (int c = 0; c < n; ++c)
                    final_mat[next_row + r][c] = inbox[r * n + c];
            
            next_row += p_chunk;
        }
    } else {
        std::vector<double> outbox(my_chunk * n);
        for (int r = 0; r < my_chunk; ++r)
            for (int c = 0; c < n; ++c)
                outbox[r * n + c] = local_R[r][c];
        
        MPI_Send(outbox.data(), my_chunk * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    return final_mat;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int my_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    if (argc != 3 && argc != 4) {
        if (my_rank == 0) {
            std::cerr << "Invocation:\n  " << argv[0] << " --test N\n";
            std::cerr << "  " << argv[0] << " A.txt B.txt out.txt\n";
        }
        MPI_Finalize();
        return 1;
    }
    
    if (std::string(argv[1]) == "--test") {
        int N = std::stoi(argv[2]);
        std::srand(std::time(nullptr) + my_rank);
        
        if (my_rank == 0) {
            std::cout << "\n>>> MPI Multiplication Test <<<\n";
            std::cout << "Dimension: " << N << "x" << N << "\n";
            std::cout << "Processes: " << world_size << "\n";
        }
        
        MatrixType X = make_random_square(N);
        MatrixType Y = make_random_square(N);
        
        double t0 = MPI_Wtime();
        MatrixType Z = mpi_product(X, Y, my_rank, world_size);
        double t1 = MPI_Wtime();
        
        if (my_rank == 0) {
            long long flops = 2LL * N * N * N;
            double elapsed = t1 - t0;
            double gflops_val = flops / elapsed / 1e9;
            
            std::cout << "Execution time: " << elapsed * 1000.0 << " ms\n";
            std::cout << "GFLOPS: " << std::fixed << std::setprecision(2) << gflops_val << "\n";
            std::cout << "Operations: " << flops << "\n";
            
            if (N <= 100) {
                MatrixType Z_ref = sequential_product(X, Y);
                double max_err = 0.0;
                for (int r = 0; r < N; ++r)
                    for (int c = 0; c < N; ++c)
                        max_err = std::max(max_err, std::abs(Z[r][c] - Z_ref[r][c]));
                
                if (max_err < 1e-9)
                    std::cout << "Verification: OK\n";
                else
                    std::cout << "Verification: FAILED (error=" << max_err << ")\n";
            }
        }
        
        MPI_Finalize();
        return 0;
    }
    
    std::string pathA = argv[1], pathB = argv[2], pathOut = argv[3];
    
    int dim;
    MatrixType A = load_matrix_file(pathA, dim);
    MatrixType B = load_matrix_file(pathB, dim);
    
    double t_start = MPI_Wtime();
    MatrixType C = mpi_product(A, B, my_rank, world_size);
    double t_end = MPI_Wtime();
    
    if (my_rank == 0) {
        save_matrix_file(pathOut, C);
        
        long long flop_cnt = 2LL * dim * dim * dim;
        std::cout << "Matrix size: " << dim << "x" << dim << "\n";
        std::cout << "Process count: " << world_size << "\n";
        std::cout << "Runtime: " << (t_end - t_start) * 1000.0 << " ms\n";
        std::cout << "FLOPs: " << flop_cnt << "\n";
    }
    
    MPI_Finalize();
    return 0;
}