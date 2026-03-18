#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <string>
#include <clocale>
#include <locale>

#ifdef _WIN32
#include <windows.h>
#endif

using namespace std;
using namespace chrono;

// загрузка матрицы из файла
vector<vector<double>> fetchMatrix(const string& filename, int& dimension) {
    ifstream reader(filename);
    if (!reader) {
        cerr << "Файл не найден: " << filename << endl;
        exit(1);
    }
    
    reader >> dimension;
    vector<vector<double>> data(dimension, vector<double>(dimension));
    
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            reader >> data[i][j];
        }
    }
    
    reader.close();
    return data;
}

// сохранение на диск
void saveMatrix(const string& filename, const vector<vector<double>>& data) {
    ofstream writer(filename);
    if (!writer) {
        cerr << "Не удалось создать файл: " << filename << endl;
        exit(1);
    }
    
    size_t n = data.size();
    writer << n << "\n";
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            writer << fixed << setprecision(6) << data[i][j] << " ";
        }
        writer << "\n";
    }
    
    writer.close();
}

// формирование статистики
void createStats(const string& filename, int n, long long microseconds) {
    ofstream stats(filename);
    if (!stats) {
        cerr << "Не удалось создать файл статистики: " << filename << endl;
        exit(1);
    }
    
    double milliseconds = microseconds / 1000.0;
    long long ops = 2LL * n * n * n;
    double mflops = ops / (microseconds / 1e6) / 1e6;
    
    stats << "========================================\n";
    stats << "                  Отчёт                 \n";
    stats << "========================================\n\n";
    stats << "Параметры:\n";
    stats << "  Размерность: " << n << " x " << n << "\n";
    stats << "  Элементов в матрице: " << n * n << "\n";
    stats << "  Всего элементов: " << 3 * n * n << "\n\n";
    stats << "Временные характеристики:\n";
    stats << microseconds << " мкс\n";
    stats << fixed << setprecision(3) << milliseconds << " мс\n";
    stats << fixed << setprecision(6) << milliseconds / 1000.0 << " с\n\n";
    stats << "Производительность:\n";
    stats << fixed << setprecision(2) << mflops << " MFLOPS\n";
    stats << "  Операций: " << ops << "\n";
    stats << "========================================\n";
    
    stats.close();
}

// умножение двух матриц
vector<vector<double>> multiply(const vector<vector<double>>& A, 
                                 const vector<vector<double>>& B) {
    size_t n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return C;
}

//генерация случайной матрицы
vector<vector<double>> generateRandom(int dimension) {
    vector<vector<double>> matrix(dimension, vector<double>(dimension));
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            matrix[i][j] = rand() % 10;
        }
    }
    return matrix;
}

// основная функция
int main(int argc, char* argv[]) {
    #ifdef _WIN32
        SetConsoleOutputCP(1251);
        SetConsoleCP(1251);
        setlocale(LC_ALL, "Russian");
    #else
        setlocale(LC_ALL, "ru_RU.UTF-8");
    #endif
    
    string path1, path2, path3, reportPath;
    int dim = 0;
    bool randomMode = false;
    
    // Разбор параметров командной строки
    if (argc == 1) {
        randomMode = true;
        dim = 5;
        reportPath = "results.txt";
        cout << "\n Режим: генерация случайных данных (5x5)\n";
        
    } else if (argc == 5) {
        path1 = argv[1];
        path2 = argv[2];
        path3 = argv[3];
        reportPath = argv[4];
        cout << "\n Режим: обработка файлов\n";
        
    } else if (argc == 4 && string(argv[1]) == "-r") {
        randomMode = true;
        dim = stoi(argv[2]);
        reportPath = argv[3];
        cout << "\n Режим: генерация случайных данных (" << dim << "x" << dim << ")\n";
        
    } else {
        cerr << "\nИСПОЛЬЗОВАНИЕ\n";
        cerr << "  " << argv[0] << "                    # тестовый запуск (5x5)\n";
        cerr << "  " << argv[0] << " -r N file.txt      # случайные матрицы NxN\n";
        cerr << "  " << argv[0] << " A.txt B.txt C.txt R.txt  # умножение из файлов\n\n";
        return 1;
    }
    
    vector<vector<double>> A, B;
    
    if (randomMode) {
        srand(static_cast<unsigned int>(time(nullptr)));
        A = generateRandom(dim);
        B = generateRandom(dim);
        
        saveMatrix("matrix_A.txt", A);
        saveMatrix("matrix_B.txt", B);
        cout << "  Созданы файлы: matrix_A.txt, matrix_B.txt\n";
    } else {
        int dimA, dimB;
        A = fetchMatrix(path1, dimA);
        B = fetchMatrix(path2, dimB);
        
        if (dimA != dimB) {
            cerr << "Размеры матриц не совпадают!\n";
            return 1;
        }
        dim = dimA;
    }
    
    cout << "  Выполняется умножение... " << flush;
    
    auto start = high_resolution_clock::now();
    vector<vector<double>> C = multiply(A, B);
    auto end = high_resolution_clock::now();
    
    auto duration = duration_cast<microseconds>(end - start);
    
    cout << "готово\n";
    
    if (randomMode) {
        saveMatrix("matrix_C.txt", C);
        cout << "  Результат: matrix_C.txt\n";
    } else {
        saveMatrix(path3, C);
        cout << "  Результат: " << path3 << "\n";
    }
    
    createStats(reportPath, dim, duration.count());
    cout << "  Отчет: " << reportPath << "\n";
    
    cout << "\n----------------------------------------\n";
    cout << "  Время счета: " << duration.count() << " мкс";
    cout << " (" << fixed << setprecision(3) << duration.count() / 1000.0 << " мс)\n";
    cout << "----------------------------------------\n\n";
    
    return 0;
}