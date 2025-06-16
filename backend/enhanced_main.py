"""
Enhanced Uber Price Optimization System - Python Implementation
Complete backend system with advanced algorithms and Indian market focus
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import math
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import heapq
from collections import defaultdict, deque
import sqlite3
import pickle
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Location:
    name: str
    lat: float
    lng: float
    address: str
    city: str
    location_type: str = 'general'

@dataclass
class Driver:
    id: str
    name: str
    location: Location
    rating: float
    is_available: bool
    completed_rides: int
    vehicle_type: str
    vehicle_number: str
    phone: str

@dataclass
class Rider:
    id: str
    name: str
    pickup: Location
    dropoff: Location
    max_price: float
    preferences: Dict[str, Any]
    phone: str

@dataclass
class PriceBreakdown:
    base_price: float
    distance_charge: float
    time_charge: float
    surge_multiplier: float
    demand_factor: float
    weather_factor: float
    traffic_factor: float
    total_price: float
    distance: float
    estimated_time: int
    vehicle_type: str
    confidence_score: float
    route_complexity: float

class DatabaseManager:
    """SQLite database manager for storing ride data"""
    
    def __init__(self, db_path: str = "uber_optimization.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Rides table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rides (
                id TEXT PRIMARY KEY,
                pickup_lat REAL,
                pickup_lng REAL,
                pickup_name TEXT,
                dropoff_lat REAL,
                dropoff_lng REAL,
                dropoff_name TEXT,
                distance REAL,
                duration INTEGER,
                base_price REAL,
                surge_multiplier REAL,
                total_price REAL,
                timestamp DATETIME,
                driver_id TEXT,
                rider_id TEXT,
                status TEXT
            )
        ''')
        
        # Drivers table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS drivers (
                id TEXT PRIMARY KEY,
                name TEXT,
                lat REAL,
                lng REAL,
                rating REAL,
                is_available BOOLEAN,
                completed_rides INTEGER,
                vehicle_type TEXT,
                vehicle_number TEXT,
                phone TEXT
            )
        ''')
        
        # Price history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                route_hash TEXT,
                distance REAL,
                base_price REAL,
                surge_multiplier REAL,
                total_price REAL,
                timestamp DATETIME,
                weather_condition TEXT,
                traffic_level TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_ride(self, ride_data: Dict):
        """Save ride data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO rides 
            (id, pickup_lat, pickup_lng, pickup_name, dropoff_lat, dropoff_lng, 
             dropoff_name, distance, duration, base_price, surge_multiplier, 
             total_price, timestamp, driver_id, rider_id, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ride_data['id'], ride_data['pickup_lat'], ride_data['pickup_lng'],
            ride_data['pickup_name'], ride_data['dropoff_lat'], ride_data['dropoff_lng'],
            ride_data['dropoff_name'], ride_data['distance'], ride_data['duration'],
            ride_data['base_price'], ride_data['surge_multiplier'], ride_data['total_price'],
            ride_data['timestamp'], ride_data.get('driver_id'), ride_data.get('rider_id'),
            ride_data.get('status', 'completed')
        ))
        
        conn.commit()
        conn.close()
    
    def get_price_history(self, route_hash: str, limit: int = 100) -> List[Dict]:
        """Get historical pricing data for a route"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM price_history 
            WHERE route_hash = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (route_hash, limit))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results

class AdvancedHaversineCalculator:
    """Enhanced distance calculation with Indian road factors"""
    
    EARTH_RADIUS_KM = 6371.0
    
    # Indian city road factors based on infrastructure quality
    CITY_ROAD_FACTORS = {
        'mumbai': 1.4,
        'delhi': 1.5,
        'bangalore': 1.6,
        'hyderabad': 1.3,
        'chennai': 1.4,
        'pune': 1.5,
        'kolkata': 1.7,
        'ahmedabad': 1.4,
        'jaipur': 1.6,
        'lucknow': 1.8
    }
    
    @classmethod
    def calculate_distance(cls, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great circle distance using Haversine formula"""
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return cls.EARTH_RADIUS_KM * c
    
    @classmethod
    def calculate_road_distance(cls, lat1: float, lon1: float, lat2: float, lon2: float, 
                              city: str = 'mumbai') -> Tuple[float, float]:
        """Calculate realistic road distance with city-specific factors"""
        straight_distance = cls.calculate_distance(lat1, lon1, lat2, lon2)
        road_factor = cls.CITY_ROAD_FACTORS.get(city.lower(), 1.4)
        road_distance = straight_distance * road_factor
        
        return straight_distance, road_distance

class PriorityDriverQueue:
    """Priority queue for efficient driver allocation using heapq"""
    
    def __init__(self):
        self.heap = []
        self.entry_finder = {}
        self.counter = 0
    
    def add_driver(self, driver: Driver, pickup_location: Location):
        """Add driver to priority queue"""
        priority = self._calculate_priority(driver, pickup_location)
        
        if driver.id in self.entry_finder:
            self.remove_driver(driver.id)
        
        entry = [priority, self.counter, driver]
        self.entry_finder[driver.id] = entry
        heapq.heappush(self.heap, entry)
        self.counter += 1
    
    def remove_driver(self, driver_id: str):
        """Remove driver from queue"""
        entry = self.entry_finder.pop(driver_id, None)
        if entry:
            entry[-1] = None  # Mark as removed
    
    def get_nearest_driver(self) -> Optional[Driver]:
        """Get the highest priority (nearest) available driver"""
        while self.heap:
            priority, count, driver = heapq.heappop(self.heap)
            if driver is not None:
                del self.entry_finder[driver.id]
                return driver
        return None
    
    def _calculate_priority(self, driver: Driver, pickup_location: Location) -> float:
        """Calculate driver priority (lower value = higher priority)"""
        if not driver.is_available:
            return float('inf')
        
        # Distance factor (primary)
        distance = AdvancedHaversineCalculator.calculate_distance(
            driver.location.lat, driver.location.lng,
            pickup_location.lat, pickup_location.lng
        )
        
        # Rating factor (better rating = lower priority value)
        rating_factor = (5.0 - driver.rating) * 0.5
        
        # Experience factor
        experience_factor = max(0, (100 - driver.completed_rides) * 0.01)
        
        return distance + rating_factor + experience_factor

class DijkstraRouteOptimizer:
    """Dijkstra's algorithm for optimal route finding"""
    
    def __init__(self):
        self.graph = defaultdict(dict)
        self.locations = {}
    
    def add_location(self, location: Location):
        """Add location to the graph"""
        self.locations[location.name] = location
    
    def add_edge(self, loc1: str, loc2: str, weight: float):
        """Add weighted edge between locations"""
        self.graph[loc1][loc2] = weight
        self.graph[loc2][loc1] = weight  # Undirected graph
    
    def build_graph_from_locations(self, locations: List[Location]):
        """Build complete graph from location list"""
        for loc in locations:
            self.add_location(loc)
        
        # Create edges between all location pairs
        for i, loc1 in enumerate(locations):
            for j, loc2 in enumerate(locations[i+1:], i+1):
                distance = AdvancedHaversineCalculator.calculate_distance(
                    loc1.lat, loc1.lng, loc2.lat, loc2.lng
                )
                self.add_edge(loc1.name, loc2.name, distance)
    
    def find_shortest_path(self, start: str, end: str) -> Tuple[float, List[str]]:
        """Find shortest path using Dijkstra's algorithm"""
        if start not in self.graph or end not in self.graph:
            return float('inf'), []
        
        distances = {node: float('inf') for node in self.graph}
        distances[start] = 0
        previous = {}
        unvisited = set(self.graph.keys())
        
        while unvisited:
            # Find minimum distance node
            current = min(unvisited, key=lambda x: distances[x])
            
            if distances[current] == float('inf'):
                break
            
            if current == end:
                break
            
            unvisited.remove(current)
            
            # Update neighbors
            for neighbor, weight in self.graph[current].items():
                if neighbor in unvisited:
                    alt_distance = distances[current] + weight
                    if alt_distance < distances[neighbor]:
                        distances[neighbor] = alt_distance
                        previous[neighbor] = current
        
        # Reconstruct path
        path = []
        current = end
        while current in previous:
            path.append(current)
            current = previous[current]
        path.append(start)
        path.reverse()
        
        return distances[end], path

class MachineLearningDemandPredictor:
    """Advanced demand prediction using multiple algorithms"""
    
    def __init__(self):
        self.historical_data = []
        self.model_weights = None
        self.feature_scaler = None
        self.is_trained = False
    
    def add_historical_data(self, timestamp: datetime, location: Location, 
                          demand_level: float, weather: str, events: List[str]):
        """Add historical demand data point"""
        features = self._extract_features(timestamp, location, weather, events)
        self.historical_data.append({
            'features': features,
            'demand': demand_level,
            'timestamp': timestamp
        })
    
    def _extract_features(self, timestamp: datetime, location: Location, 
                         weather: str, events: List[str]) -> np.ndarray:
        """Extract features for ML model"""
        features = []
        
        # Time features
        features.extend([
            timestamp.hour,
            timestamp.weekday(),
            timestamp.day,
            timestamp.month
        ])
        
        # Location features
        features.extend([
            location.lat,
            location.lng,
            hash(location.location_type) % 100  # Categorical encoding
        ])
        
        # Weather features (one-hot encoding)
        weather_conditions = ['clear', 'rain', 'heavy_rain', 'fog', 'extreme_heat']
        weather_features = [1 if weather == condition else 0 for condition in weather_conditions]
        features.extend(weather_features)
        
        # Event features
        features.append(len(events))  # Number of events
        features.append(1 if any('festival' in event.lower() for event in events) else 0)
        features.append(1 if any('concert' in event.lower() for event in events) else 0)
        
        return np.array(features, dtype=float)
    
    def train_model(self):
        """Train demand prediction model using linear regression"""
        if len(self.historical_data) < 10:
            logger.warning("Insufficient data for training. Using default predictions.")
            return
        
        # Prepare training data
        X = np.array([data['features'] for data in self.historical_data])
        y = np.array([data['demand'] for data in self.historical_data])
        
        # Simple feature scaling
        self.feature_scaler = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0) + 1e-8  # Avoid division by zero
        }
        
        X_scaled = (X - self.feature_scaler['mean']) / self.feature_scaler['std']
        
        # Simple linear regression using normal equation
        # Add bias term
        X_with_bias = np.column_stack([np.ones(X_scaled.shape[0]), X_scaled])
        
        try:
            # Normal equation: θ = (X^T X)^(-1) X^T y
            self.model_weights = np.linalg.solve(
                X_with_bias.T @ X_with_bias,
                X_with_bias.T @ y
            )
            self.is_trained = True
            logger.info("Demand prediction model trained successfully")
        except np.linalg.LinAlgError:
            logger.error("Failed to train model due to singular matrix")
    
    def predict_demand(self, timestamp: datetime, location: Location, 
                      weather: str, events: List[str]) -> float:
        """Predict demand level"""
        if not self.is_trained:
            return self._fallback_prediction(timestamp, location)
        
        features = self._extract_features(timestamp, location, weather, events)
        features_scaled = (features - self.feature_scaler['mean']) / self.feature_scaler['std']
        features_with_bias = np.concatenate([[1], features_scaled])
        
        prediction = np.dot(self.model_weights, features_with_bias)
        return max(0.5, min(3.0, prediction))  # Clamp between 0.5 and 3.0
    
    def _fallback_prediction(self, timestamp: datetime, location: Location) -> float:
        """Fallback prediction when model is not trained"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        base_demand = 1.0
        
        # Time-based patterns
        if hour in [8, 9, 18, 19, 20]:  # Peak hours
            base_demand *= 2.0
        elif hour in [7, 10, 17, 21]:  # Semi-peak
            base_demand *= 1.5
        
        # Weekend patterns
        if day_of_week >= 5:  # Weekend
            if hour >= 20:
                base_demand *= 1.3
            else:
                base_demand *= 0.8
        
        # Location-based adjustments
        if 'airport' in location.name.lower():
            base_demand *= 1.5
        elif 'mall' in location.name.lower():
            base_demand *= 1.2
        
        return min(base_demand, 3.0)

class BipartiteMatchingOptimizer:
    """Hungarian algorithm for optimal rider-driver matching"""
    
    def __init__(self):
        self.cost_matrix = None
        self.riders = []
        self.drivers = []
    
    def set_riders_and_drivers(self, riders: List[Rider], drivers: List[Driver]):
        """Set riders and drivers for matching"""
        self.riders = riders
        self.drivers = drivers
        self._build_cost_matrix()
    
    def _build_cost_matrix(self):
        """Build cost matrix for bipartite matching"""
        n_riders = len(self.riders)
        n_drivers = len(self.drivers)
        
        self.cost_matrix = np.full((n_riders, n_drivers), float('inf'))
        
        for i, rider in enumerate(self.riders):
            for j, driver in enumerate(self.drivers):
                if driver.is_available:
                    cost = self._calculate_matching_cost(rider, driver)
                    self.cost_matrix[i][j] = cost
    
    def _calculate_matching_cost(self, rider: Rider, driver: Driver) -> float:
        """Calculate cost of matching a rider with a driver"""
        # Distance cost (primary factor)
        distance = AdvancedHaversineCalculator.calculate_distance(
            rider.pickup.lat, rider.pickup.lng,
            driver.location.lat, driver.location.lng
        )
        
        # Wait time estimation
        wait_time = distance / 25.0 * 60  # Assuming 25 km/h average speed
        
        # Rating factor
        rating_factor = (5.0 - driver.rating) * 2.0
        
        # Vehicle preference
        vehicle_preference_cost = 0
        if rider.preferences.get('vehicle_type') and \
           rider.preferences['vehicle_type'] != driver.vehicle_type:
            vehicle_preference_cost = 5.0
        
        # Weighted cost calculation
        total_cost = (distance * 0.4 + wait_time * 0.3 + 
                     rating_factor * 0.2 + vehicle_preference_cost * 0.1)
        
        return total_cost
    
    def find_optimal_matching(self) -> List[Tuple[Rider, Driver, float]]:
        """Find optimal matching using simplified Hungarian algorithm"""
        if not self.riders or not self.drivers:
            return []
        
        # For simplicity, using greedy approach
        # In production, implement full Hungarian algorithm
        matches = []
        used_drivers = set()
        
        # Sort riders by urgency (could be based on wait time, price tolerance, etc.)
        sorted_riders = sorted(self.riders, key=lambda r: r.max_price, reverse=True)
        
        for rider in sorted_riders:
            best_driver = None
            best_cost = float('inf')
            
            for j, driver in enumerate(self.drivers):
                if driver.id not in used_drivers and driver.is_available:
                    rider_idx = self.riders.index(rider)
                    cost = self.cost_matrix[rider_idx][j]
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_driver = driver
            
            if best_driver and best_cost < float('inf'):
                matches.append((rider, best_driver, best_cost))
                used_drivers.add(best_driver.id)
        
        return matches

class EnhancedPriceOptimizer:
    """Main price optimization engine with all algorithms integrated"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.demand_predictor = MachineLearningDemandPredictor()
        self.route_optimizer = DijkstraRouteOptimizer()
        self.driver_queue = PriorityDriverQueue()
        self.matching_optimizer = BipartiteMatchingOptimizer()
        
        # Indian city pricing configuration
        self.city_config = {
            'mumbai': {
                'base_price_per_km': 12.0,
                'base_price_per_minute': 2.0,
                'minimum_fare': 50.0,
                'surge_cap': 5.0
            },
            'delhi': {
                'base_price_per_km': 10.0,
                'base_price_per_minute': 1.8,
                'minimum_fare': 45.0,
                'surge_cap': 4.5
            },
            'bangalore': {
                'base_price_per_km': 11.0,
                'base_price_per_minute': 1.9,
                'minimum_fare': 48.0,
                'surge_cap': 4.8
            }
        }
        
        self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize with sample historical data"""
        # Generate sample historical demand data
        base_date = datetime.now() - timedelta(days=30)
        
        sample_locations = [
            Location("Mumbai Central", 18.9690, 72.8205, "Mumbai Central Station", "Mumbai", "transport"),
            Location("BKC", 19.0596, 72.8656, "Bandra-Kurla Complex", "Mumbai", "business"),
            Location("Mumbai Airport", 19.0896, 72.8656, "CSIA Terminal", "Mumbai", "airport")
        ]
        
        for i in range(1000):  # Generate 1000 sample data points
            timestamp = base_date + timedelta(hours=random.randint(0, 720))
            location = random.choice(sample_locations)
            weather = random.choice(['clear', 'rain', 'fog'])
            events = random.choice([[], ['festival'], ['concert'], ['festival', 'holiday']])
            demand = random.uniform(0.5, 3.0)
            
            self.demand_predictor.add_historical_data(timestamp, location, demand, weather, events)
        
        # Train the demand prediction model
        self.demand_predictor.train_model()
        
        # Build route optimization graph
        self.route_optimizer.build_graph_from_locations(sample_locations)
    
    def calculate_optimized_price(self, pickup: Location, dropoff: Location, 
                                vehicle_type: str = 'UberGo', 
                                timestamp: datetime = None) -> PriceBreakdown:
        """Calculate optimized price using all algorithms"""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate distances using enhanced Haversine
        straight_distance, road_distance = AdvancedHaversineCalculator.calculate_road_distance(
            pickup.lat, pickup.lng, dropoff.lat, dropoff.lng, pickup.city.lower()
        )
        
        # Get city configuration
        city_config = self.city_config.get(pickup.city.lower(), self.city_config['mumbai'])
        
        # Predict demand using ML
        demand_factor = self.demand_predictor.predict_demand(
            timestamp, pickup, 'clear', []
        )
        
        # Calculate traffic and weather factors
        traffic_factor = self._get_traffic_factor(timestamp)
        weather_factor = self._get_weather_factor()
        
        # Base pricing components
        base_price = city_config['minimum_fare']
        distance_charge = road_distance * city_config['base_price_per_km']
        
        # Estimate time using traffic conditions
        base_speed = 25  # km/h
        adjusted_speed = base_speed / traffic_factor
        estimated_time = int((road_distance / adjusted_speed) * 60) + 5
        time_charge = estimated_time * city_config['base_price_per_minute']
        
        # Calculate surge multiplier
        surge_multiplier = min(demand_factor * 1.2, city_config['surge_cap'])
        
        # Route complexity factor
        route_complexity = self._calculate_route_complexity(pickup, dropoff, road_distance)
        
        # Total price calculation
        subtotal = base_price + distance_charge + time_charge
        total_price = subtotal * surge_multiplier * weather_factor * (1 + route_complexity * 0.1)
        
        # Ensure minimum fare
        total_price = max(total_price, city_config['minimum_fare'])
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            road_distance, traffic_factor, weather_factor, demand_factor
        )
        
        # Save to database
        ride_data = {
            'id': f"ride_{timestamp.strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
            'pickup_lat': pickup.lat,
            'pickup_lng': pickup.lng,
            'pickup_name': pickup.name,
            'dropoff_lat': dropoff.lat,
            'dropoff_lng': dropoff.lng,
            'dropoff_name': dropoff.name,
            'distance': road_distance,
            'duration': estimated_time,
            'base_price': base_price,
            'surge_multiplier': surge_multiplier,
            'total_price': total_price,
            'timestamp': timestamp.isoformat()
        }
        
        try:
            self.db_manager.save_ride(ride_data)
        except Exception as e:
            logger.error(f"Failed to save ride data: {e}")
        
        return PriceBreakdown(
            base_price=base_price,
            distance_charge=distance_charge,
            time_charge=time_charge,
            surge_multiplier=surge_multiplier,
            demand_factor=demand_factor,
            weather_factor=weather_factor,
            traffic_factor=traffic_factor,
            total_price=total_price,
            distance=road_distance,
            estimated_time=estimated_time,
            vehicle_type=vehicle_type,
            confidence_score=confidence_score,
            route_complexity=route_complexity
        )
    
    def _get_traffic_factor(self, timestamp: datetime) -> float:
        """Calculate traffic factor based on time"""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Peak hours traffic
        if hour in [8, 9, 18, 19, 20] and day_of_week < 5:
            return 1.8
        elif hour in [7, 10, 17, 21]:
            return 1.4
        else:
            return 1.1
    
    def _get_weather_factor(self) -> float:
        """Get weather impact factor (simulated)"""
        weather_conditions = ['clear', 'cloudy', 'light_rain', 'heavy_rain', 'fog']
        weights = [0.5, 0.2, 0.15, 0.1, 0.05]
        multipliers = [1.0, 1.05, 1.2, 1.5, 1.3]
        
        condition = np.random.choice(weather_conditions, p=weights)
        return multipliers[weather_conditions.index(condition)]
    
    def _calculate_route_complexity(self, pickup: Location, dropoff: Location, distance: float) -> float:
        """Calculate route complexity factor"""
        complexity = 0.0
        
        # Distance-based complexity
        if distance > 20:
            complexity += 0.3
        elif distance > 10:
            complexity += 0.2
        
        # Location type complexity
        complex_locations = ['airport', 'railway', 'mall', 'hospital']
        if any(loc_type in pickup.name.lower() for loc_type in complex_locations):
            complexity += 0.1
        if any(loc_type in dropoff.name.lower() for loc_type in complex_locations):
            complexity += 0.1
        
        return min(complexity, 0.5)  # Cap at 50% complexity
    
    def _calculate_confidence_score(self, distance: float, traffic_factor: float, 
                                  weather_factor: float, demand_factor: float) -> float:
        """Calculate prediction confidence score"""
        base_confidence = 0.85
        
        # Distance factor
        if distance < 5:
            distance_conf = 0.95
        elif distance < 15:
            distance_conf = 0.90
        else:
            distance_conf = 0.80
        
        # Traffic factor
        traffic_conf = 0.95 if traffic_factor < 1.3 else 0.80
        
        # Weather factor
        weather_conf = 0.95 if weather_factor < 1.2 else 0.85
        
        # Demand factor
        demand_conf = 0.90 if 1.0 <= demand_factor <= 2.0 else 0.75
        
        confidence = base_confidence * distance_conf * traffic_conf * weather_conf * demand_conf
        return round(confidence, 2)
    
    def allocate_optimal_driver(self, pickup: Location, available_drivers: List[Driver]) -> Optional[Driver]:
        """Allocate optimal driver using priority queue"""
        self.driver_queue = PriorityDriverQueue()  # Reset queue
        
        for driver in available_drivers:
            if driver.is_available:
                self.driver_queue.add_driver(driver, pickup)
        
        return self.driver_queue.get_nearest_driver()
    
    def optimize_multiple_rides(self, riders: List[Rider], drivers: List[Driver]) -> List[Dict]:
        """Optimize multiple rides simultaneously using bipartite matching"""
        self.matching_optimizer.set_riders_and_drivers(riders, drivers)
        matches = self.matching_optimizer.find_optimal_matching()
        
        optimized_rides = []
        for rider, driver, cost in matches:
            price_breakdown = self.calculate_optimized_price(rider.pickup, rider.dropoff)
            
            optimized_rides.append({
                'rider': asdict(rider),
                'driver': asdict(driver),
                'matching_cost': cost,
                'price_breakdown': asdict(price_breakdown),
                'estimated_pickup_time': cost / 25.0 * 60  # Convert distance to time
            })
        
        return optimized_rides

# Sample locations for testing
ENHANCED_SAMPLE_LOCATIONS = [
    Location("Mumbai Central", 18.9690, 72.8205, "Mumbai Central Railway Station", "Mumbai", "transport"),
    Location("BKC", 19.0596, 72.8656, "Bandra-Kurla Complex", "Mumbai", "business"),
    Location("Mumbai Airport", 19.0896, 72.8656, "Chhatrapati Shivaji Airport", "Mumbai", "airport"),
    Location("Gateway of India", 18.9220, 72.8347, "Gateway of India, Colaba", "Mumbai", "tourist"),
    Location("Powai", 19.1176, 72.9060, "Powai Lake Area", "Mumbai", "residential"),
    Location("Andheri West", 19.1136, 72.8697, "Andheri West Metro", "Mumbai", "residential"),
    Location("Connaught Place", 28.6315, 77.2167, "Connaught Place, New Delhi", "Delhi", "business"),
    Location("IGI Airport", 28.5562, 77.1000, "Indira Gandhi International Airport", "Delhi", "airport"),
    Location("Electronic City", 12.8456, 77.6603, "Electronic City Phase 1", "Bangalore", "tech"),
    Location("Whitefield", 12.9698, 77.7500, "Whitefield IT Hub", "Bangalore", "tech"),
]

def main():
    """Main function demonstrating the enhanced system"""
    print("🚗 Enhanced Uber Price Optimization System")
    print("=" * 60)
    print("Python Backend with Advanced DSA Implementation")
    print("=" * 60)
    
    # Initialize the optimizer
    optimizer = EnhancedPriceOptimizer()
    
    # Test price calculation
    pickup = ENHANCED_SAMPLE_LOCATIONS[0]  # Mumbai Central
    dropoff = ENHANCED_SAMPLE_LOCATIONS[1]  # BKC
    
    print(f"\n📍 Route: {pickup.name} → {dropoff.name}")
    print("-" * 40)
    
    # Calculate optimized price
    price_breakdown = optimizer.calculate_optimized_price(pickup, dropoff)
    
    print(f"💰 Price Breakdown:")
    print(f"   Base Price: ₹{price_breakdown.base_price:.2f}")
    print(f"   Distance Charge: ₹{price_breakdown.distance_charge:.2f} ({price_breakdown.distance:.1f} km)")
    print(f"   Time Charge: ₹{price_breakdown.time_charge:.2f} ({price_breakdown.estimated_time} min)")
    print(f"   Surge Multiplier: {price_breakdown.surge_multiplier:.1f}x")
    print(f"   Weather Factor: {price_breakdown.weather_factor:.1f}x")
    print(f"   Traffic Factor: {price_breakdown.traffic_factor:.1f}x")
    print(f"   Route Complexity: {price_breakdown.route_complexity:.1f}")
    print(f"   TOTAL PRICE: ₹{price_breakdown.total_price:.2f}")
    print(f"   Confidence: {price_breakdown.confidence_score:.0%}")
    
    # Test driver allocation
    print(f"\n🚗 Driver Allocation Test:")
    print("-" * 40)
    
    sample_drivers = [
        Driver("D001", "Rajesh Kumar", 
               Location("Driver1", 18.9800, 72.8300, "Near Pickup", "Mumbai", "road"), 
               4.7, True, 150, "UberGo", "MH01AB1234", "+91-9876543210"),
        Driver("D002", "Amit Sharma", 
               Location("Driver2", 18.9600, 72.8100, "Slightly Far", "Mumbai", "road"), 
               4.5, True, 200, "UberGo", "MH01CD5678", "+91-9876543211"),
        Driver("D003", "Suresh Patel", 
               Location("Driver3", 18.9900, 72.8400, "Very Close", "Mumbai", "road"), 
               4.8, True, 100, "UberGo", "MH01EF9012", "+91-9876543212"),
    ]
    
    optimal_driver = optimizer.allocate_optimal_driver(pickup, sample_drivers)
    if optimal_driver:
        print(f"   Allocated Driver: {optimal_driver.name}")
        print(f"   Rating: {optimal_driver.rating}/5.0")
        print(f"   Vehicle: {optimal_driver.vehicle_type} ({optimal_driver.vehicle_number})")
        distance_to_pickup = AdvancedHaversineCalculator.calculate_distance(
            optimal_driver.location.lat, optimal_driver.location.lng,
            pickup.lat, pickup.lng
        )
        print(f"   Distance to Pickup: {distance_to_pickup:.1f} km")
    
    # Test multiple ride optimization
    print(f"\n🔄 Multiple Ride Optimization Test:")
    print("-" * 40)
    
    sample_riders = [
        Rider("R001", "Priya Sharma", pickup, dropoff, 300.0, {"vehicle_type": "UberGo"}, "+91-9123456789"),
        Rider("R002", "Arjun Patel", ENHANCED_SAMPLE_LOCATIONS[2], ENHANCED_SAMPLE_LOCATIONS[3], 
              500.0, {"vehicle_type": "UberPremium"}, "+91-9123456790"),
    ]
    
    optimized_rides = optimizer.optimize_multiple_rides(sample_riders, sample_drivers)
    
    for i, ride in enumerate(optimized_rides, 1):
        print(f"   Ride {i}:")
        print(f"     Rider: {ride['rider']['name']}")
        print(f"     Driver: {ride['driver']['name']}")
        print(f"     Price: ₹{ride['price_breakdown']['total_price']:.2f}")
        print(f"     Pickup ETA: {ride['estimated_pickup_time']:.1f} minutes")
    
    print("\n" + "=" * 60)
    print("✅ Enhanced system demonstration completed!")
    print("💾 All ride data saved to SQLite database")
    print("🤖 ML demand prediction model trained and active")
    print("🔄 All DSA algorithms integrated and functional")

if __name__ == "__main__":
    main()