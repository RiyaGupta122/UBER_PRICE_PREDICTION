#!/usr/bin/env python3
"""
Uber Price Optimization System - Main Runner
Complete Python implementation with C DSA integration
"""

import os
import sys
import subprocess
import threading
import time
from pathlib import Path

def compile_c_algorithms():
    """Compile C algorithms for DSA operations"""
    print("🔧 Compiling C algorithms...")
    
    try:
        # Compile shared library for Python integration
        result = subprocess.run([
            'gcc', '-shared', '-fPIC', '-o', 'dsa_algorithms.so', 
            'dsa_algorithms.c', '-lm'
        ], cwd='backend', capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ C algorithms compiled successfully!")
        else:
            print(f"❌ Compilation failed: {result.stderr}")
            return False
            
        # Compile test executable
        subprocess.run([
            'gcc', '-o', 'dsa_test', 'dsa_algorithms.c', '-lm'
        ], cwd='backend', check=True)
        
        # Make executable
        subprocess.run(['chmod', '+x', 'backend/dsa_test'], check=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error compiling C code: {e}")
        return False
    except FileNotFoundError:
        print("❌ GCC compiler not found. Please install build tools.")
        return False

def install_python_dependencies():
    """Install required Python packages"""
    print("📦 Installing Python dependencies...")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'backend/requirements.txt'
        ], check=True)
        print("✅ Python dependencies installed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def run_data_generation():
    """Generate synthetic dataset"""
    print("📊 Generating synthetic dataset...")
    
    try:
        subprocess.run([
            sys.executable, 'backend/data_generator.py'
        ], check=True)
        print("✅ Dataset generated successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Data generation failed: {e}")
        return False

def test_dsa_algorithms():
    """Test C DSA implementations"""
    print("🧪 Testing DSA algorithms...")
    
    try:
        # Test C implementation
        result = subprocess.run(['./backend/dsa_test'], capture_output=True, text=True)
        print("C Algorithm Test Results:")
        print(result.stdout)
        
        # Test Python-C integration
        subprocess.run([sys.executable, 'backend/python_c_integration.py'], check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ DSA testing failed: {e}")
        return False

def run_flask_server():
    """Run the Flask API server"""
    print("🚀 Starting Flask API server...")
    
    try:
        subprocess.run([sys.executable, 'backend/api_server.py'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Flask server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Flask server error: {e}")

def run_frontend_dev_server():
    """Run the React development server"""
    print("🌐 Starting React development server...")
    
    try:
        subprocess.run(['npm', 'run', 'dev'], check=True)
    except KeyboardInterrupt:
        print("\n🛑 React server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ React server error: {e}")

def main():
    """Main system runner"""
    print("🚗 Uber Price Optimization System")
    print("=" * 50)
    print("Python Backend + C DSA Implementation")
    print("=" * 50)
    
    # Step 1: Compile C algorithms
    if not compile_c_algorithms():
        print("❌ Failed to compile C algorithms. Exiting...")
        return
    
    # Step 2: Install Python dependencies
    if not install_python_dependencies():
        print("❌ Failed to install Python dependencies. Exiting...")
        return
    
    # Step 3: Generate dataset
    if not run_data_generation():
        print("❌ Failed to generate dataset. Continuing anyway...")
    
    # Step 4: Test DSA algorithms
    if not test_dsa_algorithms():
        print("❌ DSA tests failed. Continuing anyway...")
    
    # Step 5: Start servers
    print("\n🚀 Starting application servers...")
    print("Choose an option:")
    print("1. Run Flask API server only")
    print("2. Run React frontend only") 
    print("3. Run both servers (recommended)")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        run_flask_server()
    elif choice == '2':
        run_frontend_dev_server()
    elif choice == '3':
        # Run both servers in separate threads
        flask_thread = threading.Thread(target=run_flask_server, daemon=True)
        react_thread = threading.Thread(target=run_frontend_dev_server, daemon=True)
        
        flask_thread.start()
        time.sleep(2)  # Give Flask time to start
        react_thread.start()
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down servers...")
    elif choice == '4':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice. Exiting...")

if __name__ == "__main__":
    main()