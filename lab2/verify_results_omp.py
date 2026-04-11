import numpy as np
import sys

def load_square_matrix(path):
    with open(path, 'r') as f:
        dim = int(f.readline().strip())
        rows = []
        for _ in range(dim):
            row = [float(x) for x in f.readline().split()]
            rows.append(row)
        return np.array(rows)

def run_check():
    if len(sys.argv) != 4:
        print("Usage: check_result.py mat1.txt mat2.txt result.txt")
        return 1
    
    X = load_square_matrix(sys.argv[1])
    Y = load_square_matrix(sys.argv[2])
    Z_computed = load_square_matrix(sys.argv[3])
    Z_expected = X @ Y
    
    if np.allclose(Z_computed, Z_expected, atol=1e-10):
        print("CHECK: SUCCESS")
        return 0
    else:
        max_deviation = np.max(np.abs(Z_computed - Z_expected))
        print(f"CHECK: FAILED (max deviation: {max_deviation:.6e})")
        return 1

if __name__ == "__main__":
    sys.exit(run_check())