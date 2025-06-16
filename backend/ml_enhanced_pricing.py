"""
Machine Learning Enhanced Pricing System
Advanced ML models for demand prediction and price optimization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class MLPrediction:
    demand_level: float
    confidence: float
    factors: Dict[str, float]
    model_used: str

@dataclass
class PriceOptimizationResult:
    base_price: float
    optimized_price: float
    demand_elasticity: float
    revenue_impact: float
    recommendation: str

class FeatureEngineering:
    """Advanced feature engineering for ride demand prediction"""
    
    @staticmethod
    def extract_time_features(timestamp: datetime) -> Dict[str, float]:
        """Extract comprehensive time-based features"""
        return {
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'day_of_month': timestamp.day,
            'month': timestamp.month,
            'quarter': (timestamp.month - 1) // 3 + 1,
            'is_weekend': float(timestamp.weekday() >= 5),
            'is_holiday': float(FeatureEngineering._is_indian_holiday(timestamp)),
            'hour_sin': np.sin(2 * np.pi * timestamp.hour / 24),
            'hour_cos': np.cos(2 * np.pi * timestamp.hour / 24),
            'day_sin': np.sin(2 * np.pi * timestamp.weekday() / 7),
            'day_cos': np.cos(2 * np.pi * timestamp.weekday() / 7),
            'month_sin': np.sin(2 * np.pi * timestamp.month / 12),
            'month_cos': np.cos(2 * np.pi * timestamp.month / 12),
        }
    
    @staticmethod
    def extract_location_features(lat: float, lng: float, location_type: str, city: str) -> Dict[str, float]:
        """Extract location-based features"""
        # Distance from city center
        city_centers = {
            'mumbai': (19.0760, 72.8777),
            'delhi': (28.7041, 77.1025),
            'bangalore': (12.9716, 77.5946),
            'hyderabad': (17.3850, 78.4867),
            'chennai': (13.0827, 80.2707),
            'pune': (18.5204, 73.8567)
        }
        
        center_lat, center_lng = city_centers.get(city.lower(), (19.0760, 72.8777))
        distance_from_center = np.sqrt((lat - center_lat)**2 + (lng - center_lng)**2)
        
        # Location type encoding
        location_types = {
            'residential': 0, 'business': 1, 'airport': 2, 'mall': 3, 
            'hospital': 4, 'transport': 5, 'tourist': 6, 'tech': 7
        }
        
        return {
            'latitude': lat,
            'longitude': lng,
            'distance_from_center': distance_from_center,
            'location_type_encoded': location_types.get(location_type, 0),
            'is_airport': float('airport' in location_type.lower()),
            'is_business': float('business' in location_type.lower()),
            'is_transport': float('transport' in location_type.lower()),
            'is_tech_hub': float('tech' in location_type.lower()),
        }
    
    @staticmethod
    def extract_weather_features(weather_condition: str, temperature: float = 25.0) -> Dict[str, float]:
        """Extract weather-based features"""
        weather_encoding = {
            'clear': [1, 0, 0, 0, 0],
            'cloudy': [0, 1, 0, 0, 0],
            'light_rain': [0, 0, 1, 0, 0],
            'heavy_rain': [0, 0, 0, 1, 0],
            'fog': [0, 0, 0, 0, 1]
        }
        
        encoding = weather_encoding.get(weather_condition, [1, 0, 0, 0, 0])
        
        return {
            'weather_clear': encoding[0],
            'weather_cloudy': encoding[1],
            'weather_light_rain': encoding[2],
            'weather_heavy_rain': encoding[3],
            'weather_fog': encoding[4],
            'temperature': temperature,
            'temperature_normalized': (temperature - 25) / 15,  # Normalize around 25°C
            'is_extreme_weather': float(weather_condition in ['heavy_rain', 'fog']),
        }
    
    @staticmethod
    def extract_event_features(events: List[str]) -> Dict[str, float]:
        """Extract event-based features"""
        return {
            'num_events': len(events),
            'has_festival': float(any('festival' in event.lower() for event in events)),
            'has_concert': float(any('concert' in event.lower() for event in events)),
            'has_sports': float(any('match' in event.lower() or 'game' in event.lower() for event in events)),
            'has_conference': float(any('conference' in event.lower() or 'summit' in event.lower() for event in events)),
            'event_impact_score': sum(FeatureEngineering._get_event_impact(event) for event in events),
        }
    
    @staticmethod
    def _is_indian_holiday(date: datetime) -> bool:
        """Check if date is an Indian holiday (simplified)"""
        # Major Indian holidays (simplified - in practice would use a proper holiday library)
        holidays_2024 = [
            (1, 26),   # Republic Day
            (3, 8),    # Holi (approximate)
            (8, 15),   # Independence Day
            (10, 2),   # Gandhi Jayanti
            (11, 12),  # Diwali (approximate)
        ]
        
        return (date.month, date.day) in holidays_2024
    
    @staticmethod
    def _get_event_impact(event: str) -> float:
        """Get impact score for an event"""
        event_lower = event.lower()
        if 'festival' in event_lower:
            return 2.0
        elif 'concert' in event_lower:
            return 1.5
        elif 'match' in event_lower or 'game' in event_lower:
            return 1.8
        elif 'conference' in event_lower:
            return 1.2
        else:
            return 0.5

class EnsembleDemandPredictor:
    """Ensemble model for demand prediction using multiple ML algorithms"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': Ridge(alpha=1.0),
        }
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.model_weights = None
        self.feature_importance = None
    
    def prepare_features(self, timestamp: datetime, lat: float, lng: float, 
                        location_type: str, city: str, weather: str, 
                        events: List[str], temperature: float = 25.0) -> np.ndarray:
        """Prepare feature vector for prediction"""
        features = {}
        
        # Extract all feature types
        features.update(FeatureEngineering.extract_time_features(timestamp))
        features.update(FeatureEngineering.extract_location_features(lat, lng, location_type, city))
        features.update(FeatureEngineering.extract_weather_features(weather, temperature))
        features.update(FeatureEngineering.extract_event_features(events))
        
        # Convert to array in consistent order
        if not self.feature_names:
            self.feature_names = sorted(features.keys())
        
        feature_vector = np.array([features[name] for name in self.feature_names])
        return feature_vector.reshape(1, -1)
    
    def generate_training_data(self, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data with realistic patterns"""
        np.random.seed(42)
        
        X_list = []
        y_list = []
        
        # Sample cities and their characteristics
        cities = ['mumbai', 'delhi', 'bangalore', 'hyderabad', 'chennai', 'pune']
        location_types = ['residential', 'business', 'airport', 'mall', 'hospital', 'transport', 'tourist', 'tech']
        weather_conditions = ['clear', 'cloudy', 'light_rain', 'heavy_rain', 'fog']
        
        for _ in range(n_samples):
            # Random timestamp within last year
            base_date = datetime.now() - timedelta(days=365)
            random_days = np.random.randint(0, 365)
            random_hours = np.random.randint(0, 24)
            timestamp = base_date + timedelta(days=random_days, hours=random_hours)
            
            # Random location
            city = np.random.choice(cities)
            location_type = np.random.choice(location_types)
            
            # City-specific coordinates (approximate ranges)
            city_coords = {
                'mumbai': (19.0760 + np.random.normal(0, 0.1), 72.8777 + np.random.normal(0, 0.1)),
                'delhi': (28.7041 + np.random.normal(0, 0.1), 77.1025 + np.random.normal(0, 0.1)),
                'bangalore': (12.9716 + np.random.normal(0, 0.1), 77.5946 + np.random.normal(0, 0.1)),
                'hyderabad': (17.3850 + np.random.normal(0, 0.1), 78.4867 + np.random.normal(0, 0.1)),
                'chennai': (13.0827 + np.random.normal(0, 0.1), 80.2707 + np.random.normal(0, 0.1)),
                'pune': (18.5204 + np.random.normal(0, 0.1), 73.8567 + np.random.normal(0, 0.1)),
            }
            lat, lng = city_coords[city]
            
            # Random weather
            weather = np.random.choice(weather_conditions, p=[0.5, 0.2, 0.15, 0.1, 0.05])
            temperature = np.random.normal(25, 8)  # Indian temperature range
            
            # Random events
            events = []
            if np.random.random() < 0.1:  # 10% chance of events
                event_types = ['festival', 'concert', 'cricket match', 'conference']
                events = [np.random.choice(event_types)]
            
            # Prepare features
            features = self.prepare_features(timestamp, lat, lng, location_type, city, weather, events, temperature)
            
            # Generate realistic demand based on features
            demand = self._generate_realistic_demand(timestamp, location_type, weather, events)
            
            X_list.append(features.flatten())
            y_list.append(demand)
        
        return np.array(X_list), np.array(y_list)
    
    def _generate_realistic_demand(self, timestamp: datetime, location_type: str, 
                                 weather: str, events: List[str]) -> float:
        """Generate realistic demand based on input features"""
        base_demand = 1.0
        
        # Time-based patterns
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Peak hours
        if hour in [8, 9, 18, 19, 20]:
            base_demand *= 2.2
        elif hour in [7, 10, 17, 21]:
            base_demand *= 1.6
        elif hour in range(22, 24) or hour in range(0, 6):
            base_demand *= 0.7
        
        # Weekend patterns
        if day_of_week >= 5:  # Weekend
            if hour >= 20:
                base_demand *= 1.4
            else:
                base_demand *= 0.8
        
        # Location type impact
        location_multipliers = {
            'airport': 1.8, 'business': 1.4, 'mall': 1.3, 'transport': 1.5,
            'hospital': 1.2, 'tourist': 1.1, 'tech': 1.3, 'residential': 1.0
        }
        base_demand *= location_multipliers.get(location_type, 1.0)
        
        # Weather impact
        weather_multipliers = {
            'clear': 1.0, 'cloudy': 1.05, 'light_rain': 1.3, 
            'heavy_rain': 1.8, 'fog': 1.4
        }
        base_demand *= weather_multipliers.get(weather, 1.0)
        
        # Events impact
        if events:
            base_demand *= (1.0 + len(events) * 0.3)
        
        # Add noise
        base_demand += np.random.normal(0, 0.1)
        
        return max(0.5, min(3.5, base_demand))
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train all models in the ensemble"""
        logger.info("Training ensemble demand prediction models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model_scores = {}
        predictions = {}
        
        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            score = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            model_scores[name] = {
                'r2_score': score,
                'mse': mse,
                'rmse': np.sqrt(mse)
            }
            predictions[name] = y_pred
            
            logger.info(f"{name} - R²: {score:.4f}, RMSE: {np.sqrt(mse):.4f}")
        
        # Calculate ensemble weights based on performance
        r2_scores = [model_scores[name]['r2_score'] for name in self.models.keys()]
        r2_scores = np.array(r2_scores)
        r2_scores = np.maximum(r2_scores, 0)  # Ensure non-negative
        
        if r2_scores.sum() > 0:
            self.model_weights = r2_scores / r2_scores.sum()
        else:
            self.model_weights = np.ones(len(self.models)) / len(self.models)
        
        # Calculate feature importance (from Random Forest)
        if hasattr(self.models['random_forest'], 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_names, 
                self.models['random_forest'].feature_importances_
            ))
        
        self.is_trained = True
        logger.info("Ensemble training completed!")
        
        return model_scores
    
    def predict_demand(self, timestamp: datetime, lat: float, lng: float, 
                      location_type: str, city: str, weather: str, 
                      events: List[str], temperature: float = 25.0) -> MLPrediction:
        """Predict demand using ensemble of models"""
        if not self.is_trained:
            logger.warning("Models not trained. Using fallback prediction.")
            return MLPrediction(
                demand_level=1.5,
                confidence=0.5,
                factors={'fallback': 1.0},
                model_used='fallback'
            )
        
        # Prepare features
        features = self.prepare_features(timestamp, lat, lng, location_type, city, weather, events, temperature)
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from all models
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(features_scaled)[0]
            predictions.append(pred)
        
        # Ensemble prediction
        ensemble_pred = np.average(predictions, weights=self.model_weights)
        
        # Calculate confidence based on prediction variance
        pred_variance = np.var(predictions)
        confidence = max(0.5, min(0.95, 1.0 - pred_variance))
        
        # Extract important factors
        factors = {}
        if self.feature_importance:
            feature_values = features.flatten()
            for i, (name, importance) in enumerate(sorted(self.feature_importance.items(), 
                                                         key=lambda x: x[1], reverse=True)[:5]):
                factors[name] = float(feature_values[self.feature_names.index(name)] * importance)
        
        return MLPrediction(
            demand_level=max(0.5, min(3.5, ensemble_pred)),
            confidence=confidence,
            factors=factors,
            model_used='ensemble'
        )

class PriceElasticityAnalyzer:
    """Analyze price elasticity and optimize pricing strategies"""
    
    def __init__(self):
        self.elasticity_model = None
        self.price_response_data = []
    
    def add_price_response_data(self, base_price: float, actual_price: float, 
                               demand_before: float, demand_after: float, 
                               timestamp: datetime):
        """Add historical price response data"""
        price_change_pct = (actual_price - base_price) / base_price
        demand_change_pct = (demand_after - demand_before) / demand_before if demand_before > 0 else 0
        
        self.price_response_data.append({
            'price_change_pct': price_change_pct,
            'demand_change_pct': demand_change_pct,
            'base_price': base_price,
            'actual_price': actual_price,
            'demand_before': demand_before,
            'demand_after': demand_after,
            'timestamp': timestamp,
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_peak': timestamp.hour in [8, 9, 18, 19, 20]
        })
    
    def calculate_price_elasticity(self, segment: str = 'overall') -> float:
        """Calculate price elasticity of demand"""
        if len(self.price_response_data) < 10:
            # Default elasticity values for Indian ride-sharing market
            default_elasticities = {
                'peak': -0.8,      # Less elastic during peak hours
                'off_peak': -1.2,  # More elastic during off-peak
                'overall': -1.0    # Average elasticity
            }
            return default_elasticities.get(segment, -1.0)
        
        df = pd.DataFrame(self.price_response_data)
        
        if segment == 'peak':
            df = df[df['is_peak'] == True]
        elif segment == 'off_peak':
            df = df[df['is_peak'] == False]
        
        if len(df) < 5:
            return -1.0
        
        # Calculate elasticity as % change in demand / % change in price
        valid_data = df[(df['price_change_pct'] != 0) & (df['demand_change_pct'].notna())]
        
        if len(valid_data) == 0:
            return -1.0
        
        elasticity = (valid_data['demand_change_pct'] / valid_data['price_change_pct']).mean()
        
        # Bound elasticity to reasonable range
        return max(-3.0, min(-0.1, elasticity))
    
    def optimize_price(self, base_price: float, current_demand: float, 
                      target_utilization: float = 0.8) -> PriceOptimizationResult:
        """Optimize price based on demand and elasticity"""
        
        # Get elasticity
        elasticity = self.calculate_price_elasticity()
        
        # Current utilization (simplified)
        current_utilization = min(1.0, current_demand / 2.0)  # Assuming max demand is 2.0
        
        # Calculate optimal price adjustment
        utilization_gap = target_utilization - current_utilization
        
        # Price adjustment based on elasticity
        # If demand is too high (utilization > target), increase price
        # If demand is too low (utilization < target), decrease price
        price_adjustment_pct = -utilization_gap / elasticity
        
        # Limit price adjustment to reasonable bounds
        price_adjustment_pct = max(-0.3, min(0.5, price_adjustment_pct))  # -30% to +50%
        
        optimized_price = base_price * (1 + price_adjustment_pct)
        
        # Estimate revenue impact
        demand_change_pct = elasticity * price_adjustment_pct
        new_demand = current_demand * (1 + demand_change_pct)
        
        current_revenue = base_price * current_demand
        new_revenue = optimized_price * new_demand
        revenue_impact = (new_revenue - current_revenue) / current_revenue
        
        # Generate recommendation
        if price_adjustment_pct > 0.1:
            recommendation = f"Increase price by {price_adjustment_pct*100:.1f}% to balance high demand"
        elif price_adjustment_pct < -0.1:
            recommendation = f"Decrease price by {abs(price_adjustment_pct)*100:.1f}% to stimulate demand"
        else:
            recommendation = "Current pricing is optimal"
        
        return PriceOptimizationResult(
            base_price=base_price,
            optimized_price=optimized_price,
            demand_elasticity=elasticity,
            revenue_impact=revenue_impact,
            recommendation=recommendation
        )

class MLEnhancedPricingSystem:
    """Complete ML-enhanced pricing system"""
    
    def __init__(self):
        self.demand_predictor = EnsembleDemandPredictor()
        self.elasticity_analyzer = PriceElasticityAnalyzer()
        self.is_initialized = False
    
    def initialize_system(self, n_training_samples: int = 10000):
        """Initialize the ML system with training data"""
        logger.info("Initializing ML-enhanced pricing system...")
        
        # Generate training data
        X, y = self.demand_predictor.generate_training_data(n_training_samples)
        
        # Train models
        model_scores = self.demand_predictor.train_models(X, y)
        
        # Generate sample price response data
        self._generate_sample_price_response_data()
        
        self.is_initialized = True
        logger.info("ML system initialization completed!")
        
        return model_scores
    
    def _generate_sample_price_response_data(self):
        """Generate sample price response data for elasticity analysis"""
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(500):  # Generate 500 sample price responses
            timestamp = base_date + timedelta(hours=np.random.randint(0, 720))
            base_price = np.random.uniform(50, 300)
            price_multiplier = np.random.uniform(0.8, 2.0)
            actual_price = base_price * price_multiplier
            
            demand_before = np.random.uniform(1.0, 2.5)
            
            # Simulate demand response to price change
            price_change = (actual_price - base_price) / base_price
            elasticity = np.random.uniform(-1.5, -0.5)  # Realistic elasticity range
            demand_change = elasticity * price_change
            demand_after = demand_before * (1 + demand_change)
            demand_after = max(0.5, demand_after)  # Ensure positive demand
            
            self.elasticity_analyzer.add_price_response_data(
                base_price, actual_price, demand_before, demand_after, timestamp
            )
    
    def get_comprehensive_pricing_recommendation(self, timestamp: datetime, lat: float, lng: float,
                                               location_type: str, city: str, weather: str,
                                               events: List[str], base_price: float,
                                               temperature: float = 25.0) -> Dict[str, Any]:
        """Get comprehensive pricing recommendation using ML"""
        
        if not self.is_initialized:
            logger.warning("System not initialized. Initializing with default parameters...")
            self.initialize_system(1000)  # Quick initialization
        
        # Predict demand
        demand_prediction = self.demand_predictor.predict_demand(
            timestamp, lat, lng, location_type, city, weather, events, temperature
        )
        
        # Optimize price
        price_optimization = self.elasticity_analyzer.optimize_price(
            base_price, demand_prediction.demand_level
        )
        
        # Calculate surge multiplier
        surge_multiplier = max(1.0, min(3.0, demand_prediction.demand_level))
        
        # Final recommended price
        ml_recommended_price = price_optimization.optimized_price * surge_multiplier
        
        return {
            'demand_prediction': {
                'level': demand_prediction.demand_level,
                'confidence': demand_prediction.confidence,
                'key_factors': demand_prediction.factors,
                'model_used': demand_prediction.model_used
            },
            'price_optimization': {
                'base_price': base_price,
                'optimized_base_price': price_optimization.optimized_price,
                'surge_multiplier': surge_multiplier,
                'final_recommended_price': ml_recommended_price,
                'elasticity': price_optimization.demand_elasticity,
                'revenue_impact': price_optimization.revenue_impact,
                'recommendation': price_optimization.recommendation
            },
            'market_insights': {
                'demand_level_category': self._categorize_demand(demand_prediction.demand_level),
                'pricing_strategy': self._get_pricing_strategy(demand_prediction.demand_level, surge_multiplier),
                'confidence_level': self._get_confidence_level(demand_prediction.confidence)
            }
        }
    
    def _categorize_demand(self, demand_level: float) -> str:
        """Categorize demand level"""
        if demand_level >= 2.5:
            return "Very High"
        elif demand_level >= 2.0:
            return "High"
        elif demand_level >= 1.5:
            return "Moderate"
        elif demand_level >= 1.0:
            return "Normal"
        else:
            return "Low"
    
    def _get_pricing_strategy(self, demand_level: float, surge_multiplier: float) -> str:
        """Get pricing strategy recommendation"""
        if surge_multiplier >= 2.5:
            return "Premium Pricing - High demand period"
        elif surge_multiplier >= 1.8:
            return "Dynamic Pricing - Moderate surge"
        elif surge_multiplier >= 1.3:
            return "Slight Premium - Above normal demand"
        else:
            return "Standard Pricing - Normal demand"
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level description"""
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.8:
            return "High"
        elif confidence >= 0.7:
            return "Moderate"
        else:
            return "Low"

def main():
    """Test the ML-enhanced pricing system"""
    print("🤖 ML-Enhanced Pricing System Test")
    print("=" * 50)
    
    # Initialize system
    ml_system = MLEnhancedPricingSystem()
    
    print("🔄 Initializing ML models...")
    model_scores = ml_system.initialize_system(5000)
    
    print("\n📊 Model Performance:")
    for model_name, scores in model_scores.items():
        print(f"  {model_name}:")
        print(f"    R² Score: {scores['r2_score']:.4f}")
        print(f"    RMSE: {scores['rmse']:.4f}")
    
    # Test pricing recommendation
    print("\n💰 Testing Pricing Recommendations:")
    print("-" * 40)
    
    test_scenarios = [
        {
            'name': 'Peak Hour - Business District',
            'timestamp': datetime.now().replace(hour=18, minute=30),
            'lat': 19.0596, 'lng': 72.8656,
            'location_type': 'business', 'city': 'mumbai',
            'weather': 'clear', 'events': [],
            'base_price': 150.0
        },
        {
            'name': 'Rainy Evening - Airport',
            'timestamp': datetime.now().replace(hour=20, minute=0),
            'lat': 19.0896, 'lng': 72.8656,
            'location_type': 'airport', 'city': 'mumbai',
            'weather': 'heavy_rain', 'events': [],
            'base_price': 200.0
        },
        {
            'name': 'Weekend Night - Entertainment District',
            'timestamp': datetime.now().replace(hour=22, minute=0),
            'lat': 19.0760, 'lng': 72.8777,
            'location_type': 'tourist', 'city': 'mumbai',
            'weather': 'clear', 'events': ['concert'],
            'base_price': 120.0
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n🎯 Scenario: {scenario['name']}")
        print("-" * 30)
        
        recommendation = ml_system.get_comprehensive_pricing_recommendation(
            scenario['timestamp'], scenario['lat'], scenario['lng'],
            scenario['location_type'], scenario['city'], scenario['weather'],
            scenario['events'], scenario['base_price']
        )
        
        demand = recommendation['demand_prediction']
        pricing = recommendation['price_optimization']
        insights = recommendation['market_insights']
        
        print(f"Demand Level: {demand['level']:.2f} ({insights['demand_level_category']})")
        print(f"Confidence: {demand['confidence']:.2f} ({insights['confidence_level']})")
        print(f"Base Price: ₹{pricing['base_price']:.2f}")
        print(f"Recommended Price: ₹{pricing['final_recommended_price']:.2f}")
        print(f"Surge Multiplier: {pricing['surge_multiplier']:.2f}x")
        print(f"Strategy: {insights['pricing_strategy']}")
        print(f"Revenue Impact: {pricing['revenue_impact']*100:+.1f}%")
        
        if demand['key_factors']:
            print("Key Factors:")
            for factor, value in list(demand['key_factors'].items())[:3]:
                print(f"  - {factor}: {value:.3f}")
    
    print("\n" + "=" * 50)
    print("✅ ML-Enhanced Pricing System test completed!")

if __name__ == "__main__":
    main()