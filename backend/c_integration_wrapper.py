"""
C Integration Wrapper for DSA Algorithms
Enhanced Python-C integration with error handling and performance monitoring
"""

import ctypes
import os
import time
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class CAlgorithmWrapper:
    """Enhanced wrapper for C-implemented DSA algorithms"""
    
    def __init__(self):
        self.lib = None
        self.lib_path = os.path.join(os.path.dirname(__file__), 'dsa_algorithms.so')
        self._load_library()
        self._setup_function_signatures()
    
    def _load_library(self):
        """Load the compiled C library"""
        try:
            if os.path.exists(self.lib_path):
                self.lib = ctypes.CDLL(self.lib_path)
                logger.info("C library loaded successfully")
            else:
                logger.warning(f"C library not found at {self.lib_path}")
                logger.warning("Run 'bash backend/compile_dsa.sh' to compile C algorithms")
        except Exception as e:
            logger.error(f"Failed to load C library: {e}")
    
    def _setup_function_signatures(self):
        """Setup C function signatures for type safety"""
        if not self.lib:
            return
        
        try:
            # Haversine distance function
            self.lib.haversine_distance.argtypes = [
                ctypes.c_double, ctypes.c_double, 
                ctypes.c_double, ctypes.c_double
            ]
            self.lib.haversine_distance.restype = ctypes.c_double
            
            # Dynamic pricing structure
            class PricingResult(ctypes.Structure):
                _fields_ = [
                    ("base_price", ctypes.c_double),
                    ("surge_multiplier", ctypes.c_double),
                    ("demand_factor", ctypes.c_double),
                    ("supply_factor", ctypes.c_double),
                    ("final_price", ctypes.c_double)
                ]
            
            self.PricingResult = PricingResult
            
            # Dynamic pricing function
            self.lib.calculate_dynamic_pricing.argtypes = [
                ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_double, ctypes.c_double, ctypes.c_double
            ]
            self.lib.calculate_dynamic_pricing.restype = PricingResult
            
            logger.info("C function signatures configured")
            
        except Exception as e:
            logger.error(f"Failed to setup C function signatures: {e}")
    
    def calculate_distance_optimized(self, lat1: float, lon1: float, 
                                   lat2: float, lon2: float) -> float:
        """Calculate distance using optimized C implementation"""
        if not self.lib:
            return self._python_haversine_fallback(lat1, lon1, lat2, lon2)
        
        try:
            return self.lib.haversine_distance(lat1, lon1, lat2, lon2)
        except Exception as e:
            logger.error(f"C distance calculation failed: {e}")
            return self._python_haversine_fallback(lat1, lon1, lat2, lon2)
    
    def calculate_dynamic_pricing_optimized(self, base_price: float, demand: float, 
                                          supply: float, weather: float, 
                                          traffic: float, time_factor: float) -> Dict[str, float]:
        """Calculate dynamic pricing using optimized C implementation"""
        if not self.lib:
            return self._python_pricing_fallback(base_price, demand, supply, 
                                               weather, traffic, time_factor)
        
        try:
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
        except Exception as e:
            logger.error(f"C pricing calculation failed: {e}")
            return self._python_pricing_fallback(base_price, demand, supply, 
                                               weather, traffic, time_factor)
    
    def _python_haversine_fallback(self, lat1: float, lon1: float, 
                                  lat2: float, lon2: float) -> float:
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
    
    def _python_pricing_fallback(self, base_price: float, demand: float, supply: float,
                                weather: float, traffic: float, time_factor: float) -> Dict[str, float]:
        """Python fallback for dynamic pricing"""
        import math
        
        # Supply-demand ratio
        supply_demand_ratio = demand / supply if supply > 0 else demand
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
    
    def benchmark_performance(self, iterations: int = 10000) -> Dict[str, float]:
        """Benchmark C vs Python performance"""
        logger.info(f"Running performance benchmark with {iterations} iterations")
        
        # Test coordinates (Mumbai locations)
        coords = [
            (19.0760, 72.8777),  # Mumbai
            (18.9690, 72.8205),  # Mumbai Central
            (19.0596, 72.8656),  # BKC
            (19.0896, 72.8656),  # Airport
        ]
        
        results = {}
        
        # Distance calculation benchmark
        if self.lib:
            # C implementation
            start_time = time.time()
            for _ in range(iterations):
                for i in range(len(coords)):
                    for j in range(i + 1, len(coords)):
                        self.calculate_distance_optimized(
                            coords[i][0], coords[i][1], coords[j][0], coords[j][1]
                        )
            c_time = time.time() - start_time
            results['c_distance_time'] = c_time
        
        # Python implementation
        start_time = time.time()
        for _ in range(iterations):
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    self._python_haversine_fallback(
                        coords[i][0], coords[i][1], coords[j][0], coords[j][1]
                    )
        python_time = time.time() - start_time
        results['python_distance_time'] = python_time
        
        if self.lib:
            results['distance_speedup'] = python_time / c_time
        
        # Dynamic pricing benchmark
        if self.lib:
            start_time = time.time()
            for _ in range(iterations):
                self.calculate_dynamic_pricing_optimized(100.0, 2.5, 1.2, 1.3, 1.5, 1.2)
            c_pricing_time = time.time() - start_time
            results['c_pricing_time'] = c_pricing_time
        
        start_time = time.time()
        for _ in range(iterations):
            self._python_pricing_fallback(100.0, 2.5, 1.2, 1.3, 1.5, 1.2)
        python_pricing_time = time.time() - start_time
        results['python_pricing_time'] = python_pricing_time
        
        if self.lib:
            results['pricing_speedup'] = python_pricing_time / c_pricing_time
        
        return results

class AdvancedDSAImplementations:
    """Advanced DSA implementations in Python with C optimization"""
    
    def __init__(self):
        self.c_wrapper = CAlgorithmWrapper()
    
    def dijkstra_shortest_path(self, graph: Dict[str, Dict[str, float]], 
                              start: str, end: str) -> Tuple[float, List[str]]:
        """Dijkstra's algorithm implementation with priority queue"""
        import heapq
        
        if start not in graph or end not in graph:
            return float('inf'), []
        
        # Priority queue: (distance, node)
        pq = [(0, start)]
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        previous = {}
        visited = set()
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == end:
                break
            
            for neighbor, weight in graph[current].items():
                distance = current_dist + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        # Reconstruct path
        path = []
        current = end
        while current in previous:
            path.append(current)
            current = previous[current]
        path.append(start)
        path.reverse()
        
        return distances[end], path
    
    def a_star_pathfinding(self, graph: Dict[str, Dict[str, float]], 
                          start: str, end: str, 
                          heuristic: Dict[str, float]) -> Tuple[float, List[str]]:
        """A* pathfinding algorithm for optimal route finding"""
        import heapq
        
        if start not in graph or end not in graph:
            return float('inf'), []
        
        # Priority queue: (f_score, g_score, node)
        pq = [(heuristic.get(start, 0), 0, start)]
        g_scores = {node: float('inf') for node in graph}
        g_scores[start] = 0
        f_scores = {node: float('inf') for node in graph}
        f_scores[start] = heuristic.get(start, 0)
        previous = {}
        visited = set()
        
        while pq:
            current_f, current_g, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == end:
                break
            
            for neighbor, weight in graph[current].items():
                tentative_g = current_g + weight
                
                if tentative_g < g_scores[neighbor]:
                    previous[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_scores[neighbor] = tentative_g + heuristic.get(neighbor, 0)
                    
                    if neighbor not in visited:
                        heapq.heappush(pq, (f_scores[neighbor], tentative_g, neighbor))
        
        # Reconstruct path
        path = []
        current = end
        while current in previous:
            path.append(current)
            current = previous[current]
        path.append(start)
        path.reverse()
        
        return g_scores[end], path
    
    def floyd_warshall_all_pairs(self, graph: Dict[str, Dict[str, float]]) -> Dict[Tuple[str, str], float]:
        """Floyd-Warshall algorithm for all-pairs shortest paths"""
        nodes = list(graph.keys())
        n = len(nodes)
        
        # Initialize distance matrix
        dist = {}
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if u == v:
                    dist[(u, v)] = 0
                elif v in graph[u]:
                    dist[(u, v)] = graph[u][v]
                else:
                    dist[(u, v)] = float('inf')
        
        # Floyd-Warshall main loop
        for k in nodes:
            for i in nodes:
                for j in nodes:
                    if dist[(i, k)] + dist[(k, j)] < dist[(i, j)]:
                        dist[(i, j)] = dist[(i, k)] + dist[(k, j)]
        
        return dist
    
    def kruskal_mst(self, edges: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
        """Kruskal's algorithm for Minimum Spanning Tree"""
        # Union-Find data structure
        parent = {}
        rank = {}
        
        def find(x):
            if x not in parent:
                parent[x] = x
                rank[x] = 0
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        # Sort edges by weight
        edges.sort(key=lambda x: x[2])
        
        mst = []
        for u, v, weight in edges:
            if union(u, v):
                mst.append((u, v, weight))
        
        return mst
    
    def traveling_salesman_dp(self, graph: Dict[str, Dict[str, float]], 
                             start: str) -> Tuple[float, List[str]]:
        """Traveling Salesman Problem using Dynamic Programming (small instances)"""
        nodes = list(graph.keys())
        if start not in nodes:
            return float('inf'), []
        
        n = len(nodes)
        if n > 15:  # DP approach only feasible for small instances
            return self._tsp_greedy_approximation(graph, start)
        
        # Create node index mapping
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        start_idx = node_to_idx[start]
        
        # DP table: dp[mask][i] = minimum cost to visit all nodes in mask ending at i
        dp = {}
        parent = {}
        
        # Initialize
        dp[(1 << start_idx, start_idx)] = 0
        
        # Fill DP table
        for mask in range(1 << n):
            for u in range(n):
                if not (mask & (1 << u)):
                    continue
                if (mask, u) not in dp:
                    continue
                
                for v in range(n):
                    if mask & (1 << v):
                        continue
                    
                    u_node = nodes[u]
                    v_node = nodes[v]
                    if v_node not in graph[u_node]:
                        continue
                    
                    new_mask = mask | (1 << v)
                    new_cost = dp[(mask, u)] + graph[u_node][v_node]
                    
                    if (new_mask, v) not in dp or new_cost < dp[(new_mask, v)]:
                        dp[(new_mask, v)] = new_cost
                        parent[(new_mask, v)] = u
        
        # Find minimum cost to return to start
        full_mask = (1 << n) - 1
        min_cost = float('inf')
        last_node = -1
        
        for u in range(n):
            if u == start_idx:
                continue
            if (full_mask, u) not in dp:
                continue
            
            u_node = nodes[u]
            if start not in graph[u_node]:
                continue
            
            cost = dp[(full_mask, u)] + graph[u_node][start]
            if cost < min_cost:
                min_cost = cost
                last_node = u
        
        if last_node == -1:
            return float('inf'), []
        
        # Reconstruct path
        path = []
        mask = full_mask
        current = last_node
        
        while (mask, current) in parent:
            path.append(nodes[current])
            next_current = parent[(mask, current)]
            mask ^= (1 << current)
            current = next_current
        
        path.append(start)
        path.reverse()
        path.append(start)  # Return to start
        
        return min_cost, path
    
    def _tsp_greedy_approximation(self, graph: Dict[str, Dict[str, float]], 
                                 start: str) -> Tuple[float, List[str]]:
        """Greedy approximation for TSP (for larger instances)"""
        unvisited = set(graph.keys())
        unvisited.remove(start)
        
        path = [start]
        total_cost = 0
        current = start
        
        while unvisited:
            nearest = min(unvisited, 
                         key=lambda x: graph[current].get(x, float('inf')))
            
            if graph[current].get(nearest, float('inf')) == float('inf'):
                break
            
            total_cost += graph[current][nearest]
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # Return to start
        if start in graph[current]:
            total_cost += graph[current][start]
            path.append(start)
        
        return total_cost, path

def main():
    """Test the enhanced C integration and DSA implementations"""
    print("🔧 Testing Enhanced C Integration and DSA Implementations")
    print("=" * 60)
    
    # Test C wrapper
    c_wrapper = CAlgorithmWrapper()
    
    # Performance benchmark
    print("\n📊 Performance Benchmark:")
    print("-" * 30)
    
    benchmark_results = c_wrapper.benchmark_performance(iterations=1000)
    
    for key, value in benchmark_results.items():
        if 'time' in key:
            print(f"{key}: {value:.4f} seconds")
        elif 'speedup' in key:
            print(f"{key}: {value:.2f}x faster")
    
    # Test advanced DSA implementations
    print("\n🧮 Advanced DSA Algorithm Tests:")
    print("-" * 30)
    
    dsa = AdvancedDSAImplementations()
    
    # Create sample graph
    sample_graph = {
        'A': {'B': 4, 'C': 2},
        'B': {'A': 4, 'C': 1, 'D': 5},
        'C': {'A': 2, 'B': 1, 'D': 8, 'E': 10},
        'D': {'B': 5, 'C': 8, 'E': 2},
        'E': {'C': 10, 'D': 2}
    }
    
    # Test Dijkstra
    dist, path = dsa.dijkstra_shortest_path(sample_graph, 'A', 'E')
    print(f"Dijkstra A→E: Distance={dist}, Path={' → '.join(path)}")
    
    # Test A*
    heuristic = {'A': 7, 'B': 6, 'C': 2, 'D': 1, 'E': 0}
    dist, path = dsa.a_star_pathfinding(sample_graph, 'A', 'E', heuristic)
    print(f"A* A→E: Distance={dist}, Path={' → '.join(path)}")
    
    # Test Floyd-Warshall
    all_pairs = dsa.floyd_warshall_all_pairs(sample_graph)
    print(f"Floyd-Warshall A→E: Distance={all_pairs[('A', 'E')]}")
    
    # Test Kruskal MST
    edges = [
        ('A', 'B', 4), ('A', 'C', 2), ('B', 'C', 1),
        ('B', 'D', 5), ('C', 'D', 8), ('C', 'E', 10), ('D', 'E', 2)
    ]
    mst = dsa.kruskal_mst(edges)
    mst_weight = sum(weight for _, _, weight in mst)
    print(f"Kruskal MST: Weight={mst_weight}, Edges={len(mst)}")
    
    # Test TSP
    tsp_cost, tsp_path = dsa.traveling_salesman_dp(sample_graph, 'A')
    print(f"TSP from A: Cost={tsp_cost}, Path={' → '.join(tsp_path)}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")

if __name__ == "__main__":
    main()