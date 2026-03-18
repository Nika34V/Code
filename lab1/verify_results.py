import numpy as np
import sys

def load_matrix(path):
    """Загрузка матрицы из текстового файла"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        size = int(lines[0])
        raw_data = []
        
        for i in range(size):
            row = list(map(float, lines[i+1].split()))
            raw_data.append(row)
        
        return np.array(raw_data)
    
    except FileNotFoundError:
        print(f"Файл не обнаружен: {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Ошибка при чтении {path}: {e}")
        sys.exit(1)

def compare_matrices(a_path, b_path, c_path, tolerance=1e-10):
    """Сравнение результатов с эталоном NumPy"""
    
    print("\n" + "="*50)
    print("         ПРОВЕРКА КОРРЕКТНОСТИ")
    print("="*50)
    
    # Загрузка данных
    M1 = load_matrix(a_path)
    M2 = load_matrix(b_path)
    M_result = load_matrix(c_path)
    
    print(f"\n Размерности:")
    print(f"   Матрица A: {M1.shape[0]}x{M1.shape[1]}")
    print(f"   Матрица B: {M2.shape[0]}x{M2.shape[1]}")
    print(f"   Матрица C: {M_result.shape[0]}x{M_result.shape[1]}")
    
    # Эталонное умножение
    M_expected = np.dot(M1, M2)
    
    # Анализ расхождений
    diff = np.abs(M_result - M_expected)
    max_error = np.max(diff)
    avg_error = np.mean(diff)
    
    print(f"\n Анализ погрешностей:")
    print(f"    Максимальное отклонение: {max_error:.2e}")
    print(f"    Среднее отклонение: {avg_error:.2e}")
    
    # Вердикт
    if max_error < tolerance:
        print(f"\n РЕЗУЛЬТАТ КОРРЕКТЕН (погрешность < {tolerance})")
        return True
    else:
        print(f"\n РЕЗУЛЬТАТ НЕКОРРЕКТЕН (погрешность > {tolerance})")
        
        # Демонстрация фрагментов
        print("\n Фрагменты результата (первые 3x3):")
        print("   C++:")
        print(M_result[:3, :3])
        print("   NumPy:")
        print(M_expected[:3, :3])
        
        return False

# ============================================================
def main():
    if len(sys.argv) != 4:
        print("\n" + "="*50)
        print("         ИНСТРУКЦИЯ ПО ЗАПУСКУ")
        print("="*50)
        print(f"\n  python {sys.argv[0]} A.txt B.txt C.txt\n")
        print("  Где:")
        print("    A.txt - первый сомножитель")
        print("    B.txt - второй сомножитель")
        print("    C.txt - результат умножения\n")
        sys.exit(1)
    
    a_file = sys.argv[1]
    b_file = sys.argv[2]
    c_file = sys.argv[3]
    
    success = compare_matrices(a_file, b_file, c_file)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()