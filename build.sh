#!/bin/bash
set -e
echo "========================================="
echo "  Quantum Player — Build Script (Linux/macOS)"
echo "========================================="

# Check cmake
if ! command -v cmake &>/dev/null; then
    echo "ERROR: cmake not found. Install cmake 3.16+"
    exit 1
fi

# Check Qt
QT_PREFIX=""
if command -v qmake6 &>/dev/null; then
    QT_PREFIX=$(qmake6 -query QT_INSTALL_PREFIX)
elif command -v qmake &>/dev/null; then
    QT_PREFIX=$(qmake -query QT_INSTALL_PREFIX)
fi

mkdir -p build
cd build

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release"
if [ -n "$QT_PREFIX" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_PREFIX_PATH=$QT_PREFIX"
fi

cmake .. $CMAKE_ARGS
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "✓ Build complete! Run: ./build/QuantumPlayer"
