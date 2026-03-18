@echo off
echo =========================================
echo   Quantum Player - Build Script (Windows)
echo =========================================

where cmake >nul 2>&1 || (echo ERROR: cmake not found. Download from cmake.org && exit /b 1)

if not exist build mkdir build
cd build

cmake .. -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 (
    echo.
    echo TIP: Set Qt path:
    echo   cmake .. -DCMAKE_PREFIX_PATH="C:\Qt\6.x.x\msvc2022_64"
    exit /b 1
)

cmake --build . --config Release

echo.
echo Build complete! Run: build\Release\QuantumPlayer.exe
