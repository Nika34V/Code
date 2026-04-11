@echo off
echo === CUDA Matrix Multiplication Benchmark ===
echo.

if not exist build mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

cd ..

for %%s in (200 400 800 1200 1600 2000) do (
    echo.
    echo Testing size %%s x %%s
    build\Release\matrix_mult_cuda.exe -t %%s
)

echo.
echo Benchmark complete. Results above.