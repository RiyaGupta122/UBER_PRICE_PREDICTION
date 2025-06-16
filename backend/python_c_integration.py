"""
Python-C Integration for DSA Algorithms
Uses ctypes to call optimized C functions from Python
"""

import ctypes
import os
from typing import List, Tuple
import numpy as np

class DSAAlgorithms:
    """Python wrapper for C-implemented DSA algorithms"""
    
    def __init__(self):
        # Load the compiled C library
        lib_path = os.path.join(os.path.dirname(__file__), 'dsa_algorithms.so')
        if os.path.exists(lib_path):
            self.lib = ctypes.CDLL(lib_path)
            self._setup_function_signatures()
        else:
            print("Warning: C library not found. Compile with compile_dsa.sh first.")
            self.lib = None
    
    def _setup_function_signatures(self):
        """Setup C function signatures for proper type checking"""
        
        # Haversine distance function
        self.lib.haversine_distance.argtypes = [
            ctypes.c_double, ctypes.c_double, 
            ctypes.c_double, ctypes.c_double
        ]
        self.lib.haversine_distance.restype = ctypes.c_double
        
        # Dynamic pricing function
        self.lib.calculate_dynamic_pricing.argtypes = [
            ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double
        ]
        
        # Define PricingResult structure
        class PricingResult(ctypes.Structure):
            _fields_ = [
                ("base_price", ctypes.c_double),
                ("surge_multiplier", ctypes.c_double),
                ("demand_factor", ctypes.c_double),
                ("supply_factor", ctypes.c_double),
                ("final_price", ctypes.c_double)
            ]
        
        self.lib.calculate_dynamic_pricing.restype = PricingResult
        self.PricingResult = PricingResult
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance using optimized C implementation"""
        if self.lib is None:
            # Fallback to Python implementation
            return self._python_haversine(lat1, lon1, lat2, lon2)
        
        return self.lib.haversine_distance(lat1, lon1, lat2, lon2)
    
    def calculate_dynamic_price(self, base_price: float, demand: float, supply: float,
                              weather: float, traffic: float, time_factor: float) -> dict:
        """Calculate dynamic pricing using C implementation"""
        if self.lib is None:
            # Fallback to Python implementation
            return self._python_dynamic_pricing(base_price, demand, supply, weather, traffic, time_factor)
        
        result = self.lib.calculate_dynamic_pricing(
            base_price, demand, supply, weather, traffic, time_factor
        )
        
        return {
            'base_price': result.base_price,
            'surge_multiplier': result.surge_multiplier,
            'demand_factor': result.demand_factor,
            'supply_factor': result.supply_factor,
            'final_price': result.final_price
        }
    
    def _python_haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Python fallback for Haversine calculation"""
        import math
        
        R = 6371.0  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _python_dynamic_pricing(self, base_price: float, demand: float, supply: float,
                               weather: float, traffic: float, time_factor: float) -> dict:
        """Python fallback for dynamic pricing"""
        import math
        
        # Supply-demand ratio
        supply_demand_ratio = demand / supply
        surge_multiplier = max(1.0, supply_demand_ratio)
        
        # Apply exponential surge for high demand
        if surge_multiplier > 2.0:
            surge_multiplier = 2.0 + math.pow(surge_multiplier - 2.0, 1.5)
        
        # Environmental factors
        environmental_multiplier = weather * time_factor * traffic
        
        # Final price calculation
        final_price = base_price * surge_multiplier * environmental_multiplier
        
        # Price elasticity adjustment
        max_surge = 5.0
        final_price = min(final_price, base_price * max_surge)
        
        return {
            'base_price': base_price,
            'surge_multiplier': min(surge_multiplier, max_surge),
            'demand_factor': demand,
            'supply_factor': supply,
            'final_price': final_price
        }

# Performance benchmarking
def benchmark_algorithms():
    """Benchmark C vs Python implementations"""
    import time
    
    dsa = DSAAlgorithms()
    
    # Test coordinates (Mumbai locations)
    coords = [
        (19.0760, 72.8777),  # Mumbai
        (18.9690, 72.8205),  # Mumbai Central
        (19.0596, 72.8656),  # BKC
        (19.0896, 72.8656),  # Airport
    ]
    
    print("🏃‍♂️ Performance Benchmark: C vs Python")
    print("=" * 50)
    
    # Distance calculation benchmark
    iterations = 10000
    
    # C implementation
    start_time = time.time()
    for _ in range(iterations):
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dsa.calculate_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
    c_time = time.time() - start_time
    
    # Python implementation
    start_time = time.time()
    for _ in range(iterations):
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dsa._python_haversine(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
    python_time = time.time() - start_time
    
    print(f"Distance Calculation ({iterations * 6} operations):")
    print(f"  C Implementation:      {c_time:.4f} seconds")
    print(f"  Python Implementation: {python_time:.4f} seconds")
    print(f"  Speedup:              {python_time / c_time:.2f}x")
    print()
    
    # Dynamic pricing benchmark
    start_time = time.time()
    for _ in range(iterations):
        dsa.calculate_dynamic_price(100.0, 2.5, 1.2, 1.3, 1.5, 1.2)
    c_pricing_time = time.time() - start_time
    
    start_time = time.time()
    for _ in range(iterations):
        dsa._python_dynamic_pricing(100.0, 2.5, 1.2, 1.3, 1.5, 1.2)
    python_pricing_time = time.time() - start_time
    
    print(f"Dynamic Pricing ({iterations} operations):")
    print(f"  C Implementation:      {c_pricing_time:.4f} seconds")
    print(f"  Python Implementation: {python_pricing_time:.4f} seconds")
    print(f"  Speedup:              {python_pricing_time / c_pricing_time:.2f}x")
    print("=" * 50)

if __name__ == "__main__":
    benchmark_algorithms()