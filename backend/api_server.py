"""
Flask API Server for Uber Price Optimization
Provides REST endpoints for the React frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import json
from main import UberPriceOptimizer, Location, SAMPLE_LOCATIONS
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize the price optimizer
optimizer = UberPriceOptimizer()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Uber Price Optimization API'
    })

@app.route('/api/locations', methods=['GET'])
def get_locations():
    """Get all available locations"""
    try:
        locations = []
        for loc in SAMPLE_LOCATIONS:
            locations.append({
                'name': loc.name,
                'coordinates': {
                    'lat': loc.lat,
                    'lng': loc.lng
                },
                'address': loc.address,
                'city': loc.city
            })
        
        return jsonify({
            'success': True,
            'locations': locations
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/calculate-price', methods=['POST'])
def calculate_price():
    """Calculate optimized price for a trip"""
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'pickup' not in data or 'dropoff' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing pickup or dropoff location'
            }), 400
        
        # Create Location objects
        pickup_data = data['pickup']
        dropoff_data = data['dropoff']
        
        pickup = Location(
            name=pickup_data['name'],
            lat=pickup_data['coordinates']['lat'],
            lng=pickup_data['coordinates']['lng'],
            address=pickup_data.get('address', ''),
            city=pickup_data.get('city', 'Mumbai')
        )
        
        dropoff = Location(
            name=dropoff_data['name'],
            lat=dropoff_data['coordinates']['lat'],
            lng=dropoff_data['coordinates']['lng'],
            address=dropoff_data.get('address', ''),
            city=dropoff_data.get('city', 'Mumbai')
        )
        
        # Get optional parameters
        vehicle_type = data.get('vehicle_type', 'UberGo')
        timestamp_str = data.get('timestamp')
        
        timestamp = None
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        
        # Calculate price
        price_breakdown = optimizer.calculate_optimized_price(
            pickup, dropoff, vehicle_type, timestamp
        )
        
        # Format response
        response = {
            'success': True,
            'price_breakdown': {
                'basePrice': price_breakdown.base_price,
                'distanceCharge': price_breakdown.distance_charge,
                'timeCharge': price_breakdown.time_charge,
                'surgeMultiplier': price_breakdown.surge_multiplier,
                'demandFactor': price_breakdown.demand_factor,
                'weatherFactor': price_breakdown.weather_factor,
                'trafficFactor': price_breakdown.traffic_factor,
                'totalPrice': price_breakdown.total_price,
                'distance': price_breakdown.distance,
                'estimatedTime': price_breakdown.estimated_time,
                'vehicleType': price_breakdown.vehicle_type,
                'confidenceScore': price_breakdown.confidence_score,
                'route': {
                    'distance': price_breakdown.distance,
                    'duration': price_breakdown.estimated_time,
                    'coordinates': [
                        {'lat': pickup.lat, 'lng': pickup.lng},
                        {'lat': dropoff.lat, 'lng': dropoff.lng}
                    ]
                }
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error calculating price: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/demand-forecast', methods=['GET'])
def get_demand_forecast():
    """Get demand forecast for different areas"""
    try:
        # Simulate demand forecast data
        forecast_data = {
            'current_time': datetime.now().isoformat(),
            'areas': [
                {
                    'name': 'Mumbai Central',
                    'demand_level': 'High',
                    'surge_multiplier': 1.8,
                    'predicted_wait_time': 5
                },
                {
                    'name': 'BKC',
                    'demand_level': 'Very High',
                    'surge_multiplier': 2.2,
                    'predicted_wait_time': 8
                },
                {
                    'name': 'Andheri',
                    'demand_level': 'Medium',
                    'surge_multiplier': 1.3,
                    'predicted_wait_time': 3
                },
                {
                    'name': 'Powai',
                    'demand_level': 'Medium',
                    'surge_multiplier': 1.4,
                    'predicted_wait_time': 4
                }
            ]
        }
        
        return jsonify({
            'success': True,
            'forecast': forecast_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data for dashboard"""
    try:
        # Simulate analytics data
        analytics_data = {
            'total_rides': 12847,
            'average_fare': 245,
            'peak_hours': '6-9 PM',
            'popular_route': 'Mumbai Central → BKC',
            'surge_areas': ['Bandra', 'Powai', 'Andheri'],
            'demand_forecast': 'High',
            'price_distribution': [
                {'range': '₹0-100', 'percentage': 25},
                {'range': '₹100-250', 'percentage': 35},
                {'range': '₹250-500', 'percentage': 25},
                {'range': '₹500+', 'percentage': 15}
            ],
            'demand_patterns': [
                {'time': '6 AM', 'demand': 20},
                {'time': '9 AM', 'demand': 85},
                {'time': '12 PM', 'demand': 60},
                {'time': '3 PM', 'demand': 45},
                {'time': '6 PM', 'demand': 95},
                {'time': '9 PM', 'demand': 70},
                {'time': '12 AM', 'demand': 25}
            ]
        }
        
        return jsonify({
            'success': True,
            'analytics': analytics_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/driver-allocation', methods=['POST'])
def allocate_drivers():
    """Simulate driver allocation using priority queue"""
    try:
        data = request.get_json()
        
        if not data or 'pickup_location' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing pickup location'
            }), 400
        
        # Simulate driver allocation response
        allocation_result = {
            'allocated_driver': {
                'id': 'D001',
                'name': 'Rajesh Kumar',
                'rating': 4.7,
                'vehicle': 'Maruti Swift',
                'license_plate': 'MH 01 AB 1234',
                'distance_to_pickup': 2.3,
                'estimated_arrival': 6
            },
            'alternative_drivers': [
                {
                    'id': 'D002',
                    'name': 'Amit Sharma',
                    'rating': 4.5,
                    'distance_to_pickup': 3.1,
                    'estimated_arrival': 8
                },
                {
                    'id': 'D003',
                    'name': 'Suresh Patel',
                    'rating': 4.8,
                    'distance_to_pickup': 4.2,
                    'estimated_arrival': 10
                }
            ]
        }
        
        return jsonify({
            'success': True,
            'allocation': allocation_result
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Starting Uber Price Optimization API Server")
    print("=" * 50)
    print("Available endpoints:")
    print("  GET  /api/health - Health check")
    print("  GET  /api/locations - Get all locations")
    print("  POST /api/calculate-price - Calculate trip price")
    print("  GET  /api/demand-forecast - Get demand forecast")
    print("  GET  /api/analytics - Get analytics data")
    print("  POST /api/driver-allocation - Allocate drivers")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)