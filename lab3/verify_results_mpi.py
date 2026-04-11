import numpy as np
import sys

def fetch_matrix(path):
    with open(path, 'r') as src:
        dim = int(src.readline().strip())
        data = []
        for _ in range(dim):
            row = [float(val) for val in src.readline().split()]
            data.append(row)
        return np.array(data)

def validate():
    if len(sys.argv) != 4:
        print("Usage: check_mpi_result.py matA.txt matB.txt matC.txt")
        return 1
    
    X = fetch_matrix(sys.argv[1])
    Y = fetch_matrix(sys.argv[2])
    Z_actual = fetch_matrix(sys.argv[3])
    Z_expected = X @ Y
    
    if np.allclose(Z_actual, Z_expected, atol=1e-10):
        print("VALIDATION: SUCCESS")
        return 0
    else:
        err = np.max(np.abs(Z_actual - Z_expected))
        print(f"VALIDATION: FAILURE (max error: {err:.6e})")
        return 1

if __name__ == "__main__":
    sys.exit(validate())