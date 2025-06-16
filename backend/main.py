"""
Uber Price Optimization System - Main Backend
Python implementation with advanced data science and machine learning
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

@dataclass
class Location:
    name: str
    lat: float
    lng: float
    address: str
    city: str

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

class IndianCityPricing:
    """Pricing configuration for major Indian cities"""
    
    CITY_CONFIG = {
        'mumbai': {
            'base_price_per_km': 12.0,
            'base_price_per_minute': 2.0,
            'minimum_fare': 50.0,
            'surge_cap': 5.0,
            'road_factor': 1.4
        },
        'delhi': {
            'base_price_per_km': 10.0,
            'base_price_per_minute': 1.8,
            'minimum_fare': 45.0,
            'surge_cap': 4.5,
            'road_factor': 1.5
        },
        'bangalore': {
            'base_price_per_km': 11.0,
            'base_price_per_minute': 1.9,
            'minimum_fare': 48.0,
            'surge_cap': 4.8,
            'road_factor': 1.6
        },
        'hyderabad': {
            'base_price_per_km': 9.0,
            'base_price_per_minute': 1.6,
            'minimum_fare': 40.0,
            'surge_cap': 4.0,
            'road_factor': 1.3
        },
        'chennai': {
            'base_price_per_km': 10.0,
            'base_price_per_minute': 1.7,
            'minimum_fare': 42.0,
            'surge_cap': 4.2,
            'road_factor': 1.4
        },
        'pune': {
            'base_price_per_km': 11.0,
            'base_price_per_minute': 1.8,
            'minimum_fare': 45.0,
            'surge_cap': 4.5,
            'road_factor': 1.5
        }
    }

class HaversineCalculator:
    """Optimized distance calculation using Haversine formula"""
    
    EARTH_RADIUS_KM = 6371.0
    
    @staticmethod
    def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth
        using the Haversine formula
        """
        # Convert latitude and longitude to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return HaversineCalculator.EARTH_RADIUS_KM * c
    
    @staticmethod
    def calculate_indian_road_distance(lat1: float, lon1: float, lat2: float, lon2: float, 
                                     city: str = 'mumbai') -> float:
        """
        Calculate realistic road distance considering Indian traffic conditions
        """
        straight_distance = HaversineCalculator.calculate_distance(lat1, lon1, lat2, lon2)
        road_factor = IndianCityPricing.CITY_CONFIG.get(city, {}).get('road_factor', 1.4)
        return straight_distance * road_factor

class TrafficAnalyzer:
    """Advanced traffic analysis using historical patterns"""
    
    def __init__(self):
        self.traffic_patterns = self._load_traffic_patterns()
    
    def _load_traffic_patterns(self) -> Dict:
        """Load historical traffic patterns for Indian cities"""
        return {
            'peak_hours': {
                'morning': [7, 8, 9, 10],
                'evening': [17, 18, 19, 20, 21]
            },
            'multipliers': {
                'peak': 1.8,
                'semi_peak': 1.4,
                'normal': 1.1,
                'off_peak': 0.9
            },
            'day_patterns': {
                'weekday': 1.2,
                'weekend': 0.8,
                'holiday': 0.6
            }
        }
    
    def get_traffic_multiplier(self, timestamp: datetime = None) -> Tuple[float, str]:
        """
        Calculate traffic multiplier based on time and day
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Determine traffic level
        if hour in self.traffic_patterns['peak_hours']['morning'] or \
           hour in self.traffic_patterns['peak_hours']['evening']:
            if day_of_week < 5:  # Weekday
                return self.traffic_patterns['multipliers']['peak'], 'Heavy Traffic - Peak Hours'
            else:  # Weekend
                return self.traffic_patterns['multipliers']['semi_peak'], 'Moderate Traffic - Weekend Peak'
        elif hour in range(6, 23):  # Daytime
            return self.traffic_patterns['multipliers']['normal'], 'Normal Traffic'
        else:  # Night time
            return self.traffic_patterns['multipliers']['off_peak'], 'Light Traffic - Night'

class WeatherImpactAnalyzer:
    """Weather impact analysis for pricing"""
    
    WEATHER_CONDITIONS = {
        'clear': {'multiplier': 1.0, 'description': 'Clear Weather'},
        'cloudy': {'multiplier': 1.05, 'description': 'Cloudy Conditions'},
        'light_rain': {'multiplier': 1.2, 'description': 'Light Rain - Increased Demand'},
        'heavy_rain': {'multiplier': 1.5, 'description': 'Heavy Rain - High Demand'},
        'thunderstorm': {'multiplier': 1.8, 'description': 'Thunderstorm - Very High Demand'},
        'fog': {'multiplier': 1.3, 'description': 'Foggy Conditions'},
        'extreme_heat': {'multiplier': 1.15, 'description': 'Extreme Heat - AC Demand'}
    }
    
    def get_weather_impact(self, condition: str = None) -> Tuple[float, str]:
        """
        Get weather impact on pricing
        Simulates real weather API integration
        """
        if condition is None:
            # Simulate weather conditions with Indian climate patterns
            conditions = list(self.WEATHER_CONDITIONS.keys())
            weights = [0.4, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05]  # Weighted for Indian climate
            condition = np.random.choice(conditions, p=weights)
        
        weather_data = self.WEATHER_CONDITIONS.get(condition, self.WEATHER_CONDITIONS['clear'])
        return weather_data['multiplier'], weather_data['description']

class DemandPredictor:
    """Machine Learning based demand prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self._generate_training_data()
    
    def _generate_training_data(self):
        """Generate synthetic training data based on Indian ride patterns"""
        np.random.seed(42)
        n_samples = 10000
        
        # Features: hour, day_of_week, weather_code, location_type, special_event
        hours = np.random.randint(0, 24, n_samples)
        days = np.random.randint(0, 7, n_samples)
        weather_codes = np.random.randint(0, 7, n_samples)
        location_types = np.random.randint(0, 5, n_samples)  # 0: residential, 1: business, 2: airport, 3: mall, 4: hospital
        special_events = np.random.randint(0, 2, n_samples)  # 0: no event, 1: event
        
        # Create feature matrix
        X = np.column_stack([hours, days, weather_codes, location_types, special_events])
        
        # Generate demand based on realistic patterns
        demand = np.zeros(n_samples)
        for i in range(n_samples):
            base_demand = 1.0
            
            # Hour-based demand
            if hours[i] in [8, 9, 18, 19, 20]:  # Peak hours
                base_demand *= 2.5
            elif hours[i] in [7, 10, 17, 21]:  # Semi-peak
                base_demand *= 1.8
            elif hours[i] in range(22, 24) or hours[i] in range(0, 6):  # Night
                base_demand *= 0.6
            
            # Day-based demand
            if days[i] in [5, 6]:  # Weekend
                if hours[i] in range(20, 24):  # Weekend nights
                    base_demand *= 1.5
                else:
                    base_demand *= 0.8
            
            # Weather impact
            weather_multipliers = [1.0, 1.05, 1.2, 1.5, 1.8, 1.3, 1.15]
            base_demand *= weather_multipliers[weather_codes[i]]
            
            # Location impact
            location_multipliers = [1.0, 1.3, 1.8, 1.2, 1.4]
            base_demand *= location_multipliers[location_types[i]]
            
            # Special events
            if special_events[i]:
                base_demand *= 1.6
            
            # Add noise
            demand[i] = max(0.5, base_demand + np.random.normal(0, 0.2))
        
        # Train the model
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, demand)
        self.is_trained = True
    
    def predict_demand(self, hour: int, day_of_week: int, weather_code: int, 
                      location_type: int, special_event: int = 0) -> float:
        """Predict demand multiplier"""
        if not self.is_trained:
            return 1.0
        
        features = np.array([[hour, day_of_week, weather_code, location_type, special_event]])
        features_scaled = self.scaler.transform(features)
        demand = self.model.predict(features_scaled)[0]
        
        return max(0.5, min(3.0, demand))  # Cap between 0.5x and 3.0x

class SurgePricingEngine:
    """Advanced surge pricing algorithm"""
    
    def __init__(self):
        self.demand_predictor = DemandPredictor()
    
    def calculate_surge_multiplier(self, pickup_location: Location, 
                                 timestamp: datetime = None) -> Tuple[float, Dict]:
        """
        Calculate surge multiplier using multiple factors
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Get demand prediction
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Determine location type
        location_type = self._classify_location(pickup_location)
        weather_code = random.randint(0, 6)  # Simulated weather
        special_event = self._check_special_events(timestamp, pickup_location)
        
        demand_factor = self.demand_predictor.predict_demand(
            hour, day_of_week, weather_code, location_type, special_event
        )
        
        # Calculate base surge
        surge_multiplier = 1.0
        if demand_factor > 2.0:
            surge_multiplier = 2.8
        elif demand_factor > 1.5:
            surge_multiplier = 2.0
        elif demand_factor > 1.2:
            surge_multiplier = 1.5
        elif demand_factor > 1.0:
            surge_multiplier = 1.2
        
        # Apply city-specific caps
        city = pickup_location.city.lower()
        surge_cap = IndianCityPricing.CITY_CONFIG.get(city, {}).get('surge_cap', 5.0)
        surge_multiplier = min(surge_multiplier, surge_cap)
        
        breakdown = {
            'demand_factor': demand_factor,
            'location_type': location_type,
            'special_event': special_event,
            'surge_multiplier': surge_multiplier
        }
        
        return surge_multiplier, breakdown
    
    def _classify_location(self, location: Location) -> int:
        """Classify location type for demand prediction"""
        name_lower = location.name.lower()
        
        if 'airport' in name_lower:
            return 2  # Airport
        elif any(keyword in name_lower for keyword in ['mall', 'market', 'shopping']):
            return 3  # Mall/Shopping
        elif any(keyword in name_lower for keyword in ['hospital', 'medical']):
            return 4  # Hospital
        elif any(keyword in name_lower for keyword in ['office', 'business', 'corporate', 'tech', 'it']):
            return 1  # Business
        else:
            return 0  # Residential
    
    def _check_special_events(self, timestamp: datetime, location: Location) -> int:
        """Check for special events affecting demand"""
        # Simulate special events (festivals, concerts, etc.)
        # In production, this would integrate with events API
        return random.choice([0, 0, 0, 0, 1])  # 20% chance of special event

class ETAPredictor:
    """ETA prediction using machine learning and traffic analysis"""
    
    def __init__(self):
        self.traffic_analyzer = TrafficAnalyzer()
    
    def predict_eta(self, distance: float, pickup_location: Location, 
                   dropoff_location: Location, timestamp: datetime = None) -> Tuple[int, float]:
        """
        Predict ETA using multiple factors
        Returns: (estimated_minutes, confidence_score)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Base speed calculation for Indian cities
        city = pickup_location.city.lower()
        base_speeds = {
            'mumbai': 22,  # km/h average
            'delhi': 25,
            'bangalore': 20,
            'hyderabad': 28,
            'chennai': 24,
            'pune': 26
        }
        
        base_speed = base_speeds.get(city, 25)
        
        # Get traffic multiplier
        traffic_multiplier, _ = self.traffic_analyzer.get_traffic_multiplier(timestamp)
        
        # Adjust speed based on traffic
        effective_speed = base_speed / traffic_multiplier
        
        # Calculate base time
        base_time_hours = distance / effective_speed
        base_time_minutes = base_time_hours * 60
        
        # Add buffer time and route complexity
        buffer_minutes = 5  # Standard buffer
        route_complexity = self._calculate_route_complexity(pickup_location, dropoff_location)
        complexity_minutes = route_complexity * 3
        
        total_minutes = int(base_time_minutes + buffer_minutes + complexity_minutes)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(distance, traffic_multiplier, timestamp)
        
        return total_minutes, confidence
    
    def _calculate_route_complexity(self, pickup: Location, dropoff: Location) -> float:
        """Calculate route complexity factor"""
        # Simplified complexity based on distance and location types
        distance = HaversineCalculator.calculate_distance(
            pickup.lat, pickup.lng, dropoff.lat, dropoff.lng
        )
        
        if distance > 20:  # Long distance
            return 1.5
        elif distance > 10:  # Medium distance
            return 1.2
        else:  # Short distance
            return 1.0
    
    def _calculate_confidence(self, distance: float, traffic_multiplier: float, 
                            timestamp: datetime) -> float:
        """Calculate prediction confidence score"""
        base_confidence = 0.85
        
        # Distance factor
        if distance < 5:
            distance_factor = 0.95
        elif distance < 15:
            distance_factor = 0.90
        else:
            distance_factor = 0.80
        
        # Traffic factor
        if traffic_multiplier < 1.2:
            traffic_factor = 0.95
        elif traffic_multiplier < 1.5:
            traffic_factor = 0.85
        else:
            traffic_factor = 0.75
        
        # Time factor (predictions are more accurate during normal hours)
        hour = timestamp.hour
        if 9 <= hour <= 17:
            time_factor = 0.95
        else:
            time_factor = 0.85
        
        confidence = base_confidence * distance_factor * traffic_factor * time_factor
        return round(confidence, 2)

class UberPriceOptimizer:
    """Main price optimization engine"""
    
    def __init__(self):
        self.traffic_analyzer = TrafficAnalyzer()
        self.weather_analyzer = WeatherImpactAnalyzer()
        self.surge_engine = SurgePricingEngine()
        self.eta_predictor = ETAPredictor()
    
    def calculate_optimized_price(self, pickup: Location, dropoff: Location, 
                                vehicle_type: str = 'UberGo', 
                                timestamp: datetime = None) -> PriceBreakdown:
        """
        Calculate optimized price using all factors
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate distance
        distance = HaversineCalculator.calculate_indian_road_distance(
            pickup.lat, pickup.lng, dropoff.lat, dropoff.lng, pickup.city.lower()
        )
        
        # Get city pricing configuration
        city = pickup.city.lower()
        city_config = IndianCityPricing.CITY_CONFIG.get(city, IndianCityPricing.CITY_CONFIG['mumbai'])
        
        # Base pricing components
        base_price = city_config['minimum_fare']
        distance_charge = distance * city_config['base_price_per_km']
        
        # Get ETA and time charge
        estimated_time, confidence_score = self.eta_predictor.predict_eta(
            distance, pickup, dropoff, timestamp
        )
        time_charge = estimated_time * city_config['base_price_per_minute']
        
        # Get dynamic factors
        traffic_multiplier, traffic_desc = self.traffic_analyzer.get_traffic_multiplier(timestamp)
        weather_multiplier, weather_desc = self.weather_analyzer.get_weather_impact()
        surge_multiplier, surge_breakdown = self.surge_engine.calculate_surge_multiplier(
            pickup, timestamp
        )
        
        # Calculate total price
        subtotal = base_price + distance_charge + time_charge
        total_price = subtotal * surge_multiplier * weather_multiplier
        
        # Apply minimum fare
        total_price = max(total_price, city_config['minimum_fare'])
        
        return PriceBreakdown(
            base_price=base_price,
            distance_charge=distance_charge,
            time_charge=time_charge,
            surge_multiplier=surge_multiplier,
            demand_factor=surge_breakdown['demand_factor'],
            weather_factor=weather_multiplier,
            traffic_factor=traffic_multiplier,
            total_price=total_price,
            distance=distance,
            estimated_time=estimated_time,
            vehicle_type=vehicle_type,
            confidence_score=confidence_score
        )

# Sample locations for testing
SAMPLE_LOCATIONS = [
    Location("Mumbai Central", 18.9690, 72.8205, "Mumbai Central Railway Station", "Mumbai"),
    Location("BKC", 19.0596, 72.8656, "Bandra-Kurla Complex", "Mumbai"),
    Location("Mumbai Airport", 19.0896, 72.8656, "Chhatrapati Shivaji Airport", "Mumbai"),
    Location("Connaught Place", 28.6315, 77.2167, "Connaught Place, New Delhi", "Delhi"),
    Location("IGI Airport", 28.5562, 77.1000, "Indira Gandhi International Airport", "Delhi"),
    Location("Electronic City", 12.8456, 77.6603, "Electronic City, Bangalore", "Bangalore"),
    Location("Whitefield", 12.9698, 77.7500, "Whitefield, Bangalore", "Bangalore"),
]

def main():
    """Main function for testing the price optimization system"""
    optimizer = UberPriceOptimizer()
    
    # Test with sample locations
    pickup = SAMPLE_LOCATIONS[0]  # Mumbai Central
    dropoff = SAMPLE_LOCATIONS[1]  # BKC
    
    print("🚗 Uber Price Optimization System")
    print("=" * 50)
    print(f"Pickup: {pickup.name}")
    print(f"Dropoff: {dropoff.name}")
    print("-" * 50)
    
    # Calculate price
    price_breakdown = optimizer.calculate_optimized_price(pickup, dropoff)
    
    print(f"Base Price: ₹{price_breakdown.base_price:.2f}")
    print(f"Distance Charge: ₹{price_breakdown.distance_charge:.2f} ({price_breakdown.distance:.1f} km)")
    print(f"Time Charge: ₹{price_breakdown.time_charge:.2f} ({price_breakdown.estimated_time} min)")
    print(f"Surge Multiplier: {price_breakdown.surge_multiplier:.1f}x")
    print(f"Weather Factor: {price_breakdown.weather_factor:.1f}x")
    print(f"Traffic Factor: {price_breakdown.traffic_factor:.1f}x")
    print("-" * 50)
    print(f"TOTAL PRICE: ₹{price_breakdown.total_price:.2f}")
    print(f"Confidence Score: {price_breakdown.confidence_score:.0%}")
    print("=" * 50)

if __name__ == "__main__":
    main()