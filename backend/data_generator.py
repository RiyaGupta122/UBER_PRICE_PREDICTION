"""
Synthetic Data Generator for Uber Price Optimization
Generates realistic Indian ride-sharing data for training and testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random
from typing import List, Dict
import os

class IndianRideDataGenerator:
    """Generate synthetic ride data based on Indian market patterns"""
    
    def __init__(self):
        self.cities = {
            'Mumbai': {
                'center': (19.0760, 72.8777),
                'radius': 0.5,
                'base_price': 12,
                'population_density': 'very_high'
            },
            'Delhi': {
                'center': (28.7041, 77.1025),
                'radius': 0.6,
                'base_price': 10,
                'population_density': 'very_high'
            },
            'Bangalore': {
                'center': (12.9716, 77.5946),
                'radius': 0.4,
                'base_price': 11,
                'population_density': 'high'
            },
            'Hyderabad': {
                'center': (17.3850, 78.4867),
                'radius': 0.4,
                'base_price': 9,
                'population_density': 'high'
            },
            'Chennai': {
                'center': (13.0827, 80.2707),
                'radius': 0.4,
                'base_price': 10,
                'population_density': 'high'
            },
            'Pune': {
                'center': (18.5204, 73.8567),
                'radius': 0.3,
                'base_price': 11,
                'population_density': 'medium'
            }
        }
        
        self.location_types = [
            'residential', 'business', 'airport', 'mall', 'hospital', 
            'railway_station', 'metro_station', 'university', 'tech_park'
        ]
        
        self.weather_conditions = [
            'clear', 'cloudy', 'light_rain', 'heavy_rain', 'fog', 'extreme_heat'
        ]
        
        self.vehicle_types = ['UberGo', 'UberPremium', 'UberXL', 'UberAuto']
    
    def generate_location(self, city: str) -> Dict:
        """Generate a random location within a city"""
        city_data = self.cities[city]
        center_lat, center_lng = city_data['center']
        radius = city_data['radius']
        
        # Generate random coordinates within city radius
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(0, radius)
        
        lat = center_lat + distance * np.cos(angle)
        lng = center_lng + distance * np.sin(angle)
        
        location_type = random.choice(self.location_types)
        
        return {
            'lat': lat,
            'lng': lng,
            'city': city,
            'location_type': location_type
        }
    
    def generate_time_patterns(self, start_date: datetime, num_days: int) -> List[datetime]:
        """Generate realistic time patterns for rides"""
        timestamps = []
        
        for day in range(num_days):
            current_date = start_date + timedelta(days=day)
            day_of_week = current_date.weekday()
            
            # Different patterns for weekdays vs weekends
            if day_of_week < 5:  # Weekday
                peak_hours = [8, 9, 18, 19, 20]  # Morning and evening peaks
                rides_per_hour = {
                    **{hour: random.randint(50, 80) for hour in peak_hours},
                    **{hour: random.randint(20, 40) for hour in range(24) if hour not in peak_hours}
                }
            else:  # Weekend
                peak_hours = [11, 12, 20, 21, 22]  # Brunch and night peaks
                rides_per_hour = {
                    **{hour: random.randint(40, 60) for hour in peak_hours},
                    **{hour: random.randint(15, 30) for hour in range(24) if hour not in peak_hours}
                }
            
            # Generate timestamps for each hour
            for hour in range(24):
                num_rides = rides_per_hour.get(hour, 20)
                for _ in range(num_rides):
                    minute = random.randint(0, 59)
                    second = random.randint(0, 59)
                    timestamp = current_date.replace(hour=hour, minute=minute, second=second)
                    timestamps.append(timestamp)
        
        return timestamps
    
    def calculate_realistic_price(self, pickup: Dict, dropoff: Dict, 
                                timestamp: datetime, weather: str) -> Dict:
        """Calculate realistic price based on multiple factors"""
        from main import HaversineCalculator, IndianCityPricing
        
        # Calculate distance
        distance = HaversineCalculator.calculate_indian_road_distance(
            pickup['lat'], pickup['lng'], dropoff['lat'], dropoff['lng'], 
            pickup['city'].lower()
        )
        
        # Get city pricing
        city_config = IndianCityPricing.CITY_CONFIG.get(
            pickup['city'].lower(), 
            IndianCityPricing.CITY_CONFIG['mumbai']
        )
        
        # Base price calculation
        base_price = city_config['minimum_fare']
        distance_charge = distance * city_config['base_price_per_km']
        
        # Time-based factors
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Traffic multiplier
        if hour in [8, 9, 18, 19, 20] and day_of_week < 5:
            traffic_multiplier = 1.8
        elif hour in [7, 10, 17, 21]:
            traffic_multiplier = 1.4
        else:
            traffic_multiplier = 1.1
        
        # Weather multiplier
        weather_multipliers = {
            'clear': 1.0, 'cloudy': 1.05, 'light_rain': 1.2,
            'heavy_rain': 1.5, 'fog': 1.3, 'extreme_heat': 1.15
        }
        weather_multiplier = weather_multipliers.get(weather, 1.0)
        
        # Demand multiplier based on location and time
        demand_multiplier = 1.0
        if pickup['location_type'] in ['airport', 'railway_station']:
            demand_multiplier *= 1.3
        if hour in [8, 9, 18, 19, 20]:
            demand_multiplier *= 1.5
        
        # Calculate estimated time
        base_speed = 25  # km/h
        adjusted_speed = base_speed / traffic_multiplier
        estimated_time = int((distance / adjusted_speed) * 60) + 5
        
        time_charge = estimated_time * city_config['base_price_per_minute']
        
        # Total price calculation
        subtotal = base_price + distance_charge + time_charge
        surge_multiplier = min(demand_multiplier, 3.0)
        total_price = subtotal * surge_multiplier * weather_multiplier
        
        return {
            'base_price': base_price,
            'distance_charge': distance_charge,
            'time_charge': time_charge,
            'distance': distance,
            'estimated_time': estimated_time,
            'surge_multiplier': surge_multiplier,
            'weather_multiplier': weather_multiplier,
            'traffic_multiplier': traffic_multiplier,
            'total_price': max(total_price, city_config['minimum_fare'])
        }
    
    def generate_ride_dataset(self, num_rides: int = 10000, 
                            start_date: datetime = None) -> pd.DataFrame:
        """Generate comprehensive ride dataset"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        
        print(f"🔄 Generating {num_rides} synthetic rides...")
        
        rides = []
        timestamps = self.generate_time_patterns(start_date, 30)
        
        for i in range(num_rides):
            # Select random timestamp
            timestamp = random.choice(timestamps)
            
            # Select random city
            city = random.choice(list(self.cities.keys()))
            
            # Generate pickup and dropoff locations
            pickup = self.generate_location(city)
            dropoff = self.generate_location(city)
            
            # Ensure minimum distance
            while HaversineCalculator.calculate_distance(
                pickup['lat'], pickup['lng'], dropoff['lat'], dropoff['lng']
            ) < 0.5:  # Minimum 0.5 km
                dropoff = self.generate_location(city)
            
            # Random weather
            weather = random.choice(self.weather_conditions)
            
            # Random vehicle type
            vehicle_type = random.choice(self.vehicle_types)
            
            # Calculate price
            price_data = self.calculate_realistic_price(pickup, dropoff, timestamp, weather)
            
            # Create ride record
            ride = {
                'ride_id': f'R{i+1:06d}',
                'timestamp': timestamp,
                'pickup_lat': pickup['lat'],
                'pickup_lng': pickup['lng'],
                'pickup_location_type': pickup['location_type'],
                'dropoff_lat': dropoff['lat'],
                'dropoff_lng': dropoff['lng'],
                'dropoff_location_type': dropoff['location_type'],
                'city': city,
                'distance': price_data['distance'],
                'estimated_time': price_data['estimated_time'],
                'vehicle_type': vehicle_type,
                'weather': weather,
                'hour': timestamp.hour,
                'day_of_week': timestamp.weekday(),
                'is_weekend': timestamp.weekday() >= 5,
                'base_price': price_data['base_price'],
                'distance_charge': price_data['distance_charge'],
                'time_charge': price_data['time_charge'],
                'surge_multiplier': price_data['surge_multiplier'],
                'weather_multiplier': price_data['weather_multiplier'],
                'traffic_multiplier': price_data['traffic_multiplier'],
                'total_price': price_data['total_price']
            }
            
            rides.append(ride)
            
            if (i + 1) % 1000 == 0:
                print(f"  Generated {i + 1}/{num_rides} rides...")
        
        df = pd.DataFrame(rides)
        print(f"✅ Dataset generated with {len(df)} rides")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = 'uber_rides_dataset.csv'):
        """Save dataset to CSV file"""
        filepath = os.path.join(os.path.dirname(__file__), filename)
        df.to_csv(filepath, index=False)
        print(f"💾 Dataset saved to {filepath}")
        
        # Generate summary statistics
        self.generate_dataset_summary(df)
    
    def generate_dataset_summary(self, df: pd.DataFrame):
        """Generate and display dataset summary"""
        print("\n📊 Dataset Summary")
        print("=" * 50)
        print(f"Total Rides: {len(df):,}")
        print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Cities: {', '.join(df['city'].unique())}")
        print(f"Average Fare: ₹{df['total_price'].mean():.2f}")
        print(f"Average Distance: {df['distance'].mean():.2f} km")
        print(f"Average Duration: {df['estimated_time'].mean():.1f} minutes")
        
        print("\n🏙️ City Distribution:")
        city_counts = df['city'].value_counts()
        for city, count in city_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {city}: {count:,} rides ({percentage:.1f}%)")
        
        print("\n🚗 Vehicle Type Distribution:")
        vehicle_counts = df['vehicle_type'].value_counts()
        for vehicle, count in vehicle_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {vehicle}: {count:,} rides ({percentage:.1f}%)")
        
        print("\n⏰ Peak Hours Analysis:")
        hourly_rides = df.groupby('hour').size().sort_values(ascending=False)
        print("  Top 5 busiest hours:")
        for hour, count in hourly_rides.head().items():
            print(f"    {hour:02d}:00 - {count:,} rides")
        
        print("\n💰 Price Analysis:")
        print(f"  Min Fare: ₹{df['total_price'].min():.2f}")
        print(f"  Max Fare: ₹{df['total_price'].max():.2f}")
        print(f"  Median Fare: ₹{df['total_price'].median():.2f}")
        print(f"  Avg Surge: {df['surge_multiplier'].mean():.2f}x")
        
        print("=" * 50)

def main():
    """Generate and save the dataset"""
    generator = IndianRideDataGenerator()
    
    # Generate dataset
    df = generator.generate_ride_dataset(num_rides=15000)
    
    # Save to file
    generator.save_dataset(df, 'indian_uber_rides_dataset.csv')
    
    # Generate additional smaller datasets for specific analysis
    print("\n🔄 Generating specialized datasets...")
    
    # Peak hours dataset
    peak_df = df[df['hour'].isin([8, 9, 18, 19, 20])].copy()
    generator.save_dataset(peak_df, 'peak_hours_rides.csv')
    
    # Weekend dataset
    weekend_df = df[df['is_weekend'] == True].copy()
    generator.save_dataset(weekend_df, 'weekend_rides.csv')
    
    # High surge dataset
    surge_df = df[df['surge_multiplier'] > 1.5].copy()
    generator.save_dataset(surge_df, 'high_surge_rides.csv')
    
    print("\n✅ All datasets generated successfully!")

if __name__ == "__main__":
    main()