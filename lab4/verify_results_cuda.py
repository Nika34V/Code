import numpy as np
import sys

def load_square_matrix(path):
    with open(path, 'r') as src:
        dim = int(src.readline().strip())
        rows = []
        for _ in range(dim):
            row_vals = [float(x) for x in src.readline().split()]
            rows.append(row_vals)
        return np.array(rows)

def run_verification():
    if len(sys.argv) != 4:
        print("Usage: check_cuda_result.py matA.txt matB.txt matC.txt")
        return 1
    
    X = load_square_matrix(sys.argv[1])
    Y = load_square_matrix(sys.argv[2])
    Z_actual = load_square_matrix(sys.argv[3])
    Z_expected = np.dot(X, Y)
    
    print("\n=== VERIFICATION REPORT ===")
    print(f"Dimension: {X.shape[0]}x{X.shape[1]}")
    
    if np.allclose(Z_actual, Z_expected, atol=1e-8):
        print("OUTCOME: SUCCESS")
        return 0
    else:
        max_err = np.max(np.abs(Z_actual - Z_expected))
        print(f"OUTCOME: FAILURE (max discrepancy: {max_err:.6e})")
        return 1

if __name__ == "__main__":
    sys.exit(run_verification())