#!/bin/bash

# Compile DSA algorithms C code
echo "🔧 Compiling DSA algorithms..."

# Compile as shared library for Python integration
gcc -shared -fPIC -o dsa_algorithms.so dsa_algorithms.c -lm

# Compile as standalone executable for testing
gcc -o dsa_test dsa_algorithms.c -lm

echo "✅ Compilation complete!"
echo "   - Shared library: dsa_algorithms.so"
echo "   - Test executable: dsa_test"

# Make executable
chmod +x dsa_test

echo "🧪 Running DSA tests..."
./dsa_test