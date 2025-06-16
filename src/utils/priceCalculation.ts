import { Location, PriceBreakdown, TrafficData, WeatherData } from '../types';

// Haversine formula for calculating distance between two coordinates
export function haversineDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371; // Earth's radius in kilometers
  const dLat = toRadians(lat2 - lat1);
  const dLon = toRadians(lon2 - lon1);
  
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRadians(lat1)) * Math.cos(toRadians(lat2)) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  
  return R * c;
}

function toRadians(degrees: number): number {
  return degrees * (Math.PI / 180);
}

// Enhanced distance calculation for Indian roads
function calculateIndianRoadDistance(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const straightLineDistance = haversineDistance(lat1, lon1, lat2, lon2);
  
  // Indian road factor - accounts for traffic, road quality, and detours
  // Based on city type and infrastructure quality
  const roadFactor = 1.4; // Metro cities average
  
  return straightLineDistance * roadFactor;
}

// Get current traffic conditions (simulated)
function getCurrentTrafficData(): TrafficData {
  const hour = new Date().getHours();
  
  // Peak hours: 8-10 AM and 6-9 PM
  if ((hour >= 8 && hour <= 10) || (hour >= 18 && hour <= 21)) {
    return {
      level: 'high',
      multiplier: 1.8,
      description: 'Heavy traffic - Peak hours'
    };
  } else if ((hour >= 7 && hour <= 11) || (hour >= 17 && hour <= 22)) {
    return {
      level: 'medium',
      multiplier: 1.3,
      description: 'Moderate traffic'
    };
  } else {
    return {
      level: 'low',
      multiplier: 1.0,
      description: 'Light traffic'
    };
  }
}

// Get current weather conditions (simulated)
function getCurrentWeatherData(): WeatherData {
  const conditions = ['clear', 'rain', 'heavy_rain', 'fog'] as const;
  const multipliers = [1.0, 1.2, 1.5, 1.3];
  const descriptions = [
    'Clear weather',
    'Light rain - Increased demand',
    'Heavy rain - High demand',
    'Foggy conditions - Reduced visibility'
  ];
  
  // Simulate weather (weighted towards clear weather)
  const random = Math.random();
  let selectedIndex = 0;
  
  if (random < 0.6) selectedIndex = 0; // 60% clear
  else if (random < 0.8) selectedIndex = 1; // 20% rain
  else if (random < 0.95) selectedIndex = 2; // 15% heavy rain
  else selectedIndex = 3; // 5% fog
  
  return {
    condition: conditions[selectedIndex],
    multiplier: multipliers[selectedIndex],
    description: descriptions[selectedIndex]
  };
}

// Calculate demand factor based on time and location
function calculateDemandFactor(pickup: Location, dropoff: Location): number {
  const hour = new Date().getHours();
  const day = new Date().getDay();
  
  // Base demand multiplier
  let demandFactor = 1.0;
  
  // Time-based demand
  if ((hour >= 8 && hour <= 10) || (hour >= 18 && hour <= 21)) {
    demandFactor *= 1.5; // Peak hours
  } else if ((hour >= 22 && hour <= 24) || (hour >= 0 && hour <= 6)) {
    demandFactor *= 1.2; // Late night/early morning
  }
  
  // Weekend factor
  if (day === 0 || day === 6) { // Sunday or Saturday
    if (hour >= 20 && hour <= 24) {
      demandFactor *= 1.3; // Weekend nights
    }
  }
  
  // Location-based demand (airports, business districts)
  const businessAreas = ['airport', 'bkc', 'electronic city', 'cyber city', 'hitec city'];
  const isBusinessArea = businessAreas.some(area => 
    pickup.name.toLowerCase().includes(area) || 
    dropoff.name.toLowerCase().includes(area)
  );
  
  if (isBusinessArea) {
    demandFactor *= 1.2;
  }
  
  return Math.min(demandFactor, 3.0); // Cap at 3x
}

// Calculate surge multiplier based on supply-demand
function calculateSurgeMultiplier(demandFactor: number, trafficData: TrafficData, weatherData: WeatherData): number {
  let surgeMultiplier = 1.0;
  
  // Demand-based surge
  if (demandFactor > 2.0) {
    surgeMultiplier = 2.5;
  } else if (demandFactor > 1.5) {
    surgeMultiplier = 1.8;
  } else if (demandFactor > 1.2) {
    surgeMultiplier = 1.4;
  }
  
  // Weather impact on surge
  if (weatherData.condition === 'heavy_rain') {
    surgeMultiplier *= 1.3;
  } else if (weatherData.condition === 'rain') {
    surgeMultiplier *= 1.15;
  }
  
  // Traffic impact (slight)
  if (trafficData.level === 'high') {
    surgeMultiplier *= 1.1;
  }
  
  return Math.min(surgeMultiplier, 5.0); // Cap at 5x
}

// Estimate travel time
function estimateTravelTime(distance: number, trafficData: TrafficData): number {
  // Base speed assumptions for Indian cities
  const baseSpeed = 25; // km/h average in cities
  const adjustedSpeed = baseSpeed / trafficData.multiplier;
  
  // Convert to minutes and add buffer
  const travelTime = (distance / adjustedSpeed) * 60;
  return Math.round(travelTime + 5); // Add 5 minutes buffer
}

// Generate route simulation
function generateRoute(pickup: Location, dropoff: Location, distance: number): any {
  // Simulate route with intermediate points
  const steps = 5;
  const coordinates = [];
  
  for (let i = 0; i <= steps; i++) {
    const ratio = i / steps;
    const lat = pickup.coordinates.lat + (dropoff.coordinates.lat - pickup.coordinates.lat) * ratio;
    const lng = pickup.coordinates.lng + (dropoff.coordinates.lng - pickup.coordinates.lng) * ratio;
    coordinates.push({ lat, lng });
  }
  
  return {
    coordinates,
    distance: distance,
    duration: estimateTravelTime(distance, getCurrentTrafficData())
  };
}

// Main price calculation function
export function calculatePrice(pickup: Location, dropoff: Location): PriceBreakdown {
  // Calculate distance
  const distance = calculateIndianRoadDistance(
    pickup.coordinates.lat,
    pickup.coordinates.lng,
    dropoff.coordinates.lat,
    dropoff.coordinates.lng
  );
  
  // Get current conditions
  const trafficData = getCurrentTrafficData();
  const weatherData = getCurrentWeatherData();
  const demandFactor = calculateDemandFactor(pickup, dropoff);
  const surgeMultiplier = calculateSurgeMultiplier(demandFactor, trafficData, weatherData);
  
  // Base pricing (Mumbai rates as example)
  const basePricePerKm = 12; // ₹12 per km
  const basePricePerMinute = 2; // ₹2 per minute
  const minimumFare = 50; // ₹50 minimum
  
  // Calculate base components
  const basePrice = minimumFare;
  const distanceCharge = distance * basePricePerKm;
  const estimatedTime = estimateTravelTime(distance, trafficData);
  const timeCharge = estimatedTime * basePricePerMinute;
  
  // Apply multipliers
  const subtotal = basePrice + distanceCharge + timeCharge;
  const totalPrice = subtotal * surgeMultiplier * weatherData.multiplier;
  
  // Generate route
  const route = generateRoute(pickup, dropoff, distance);
  
  return {
    basePrice,
    distanceCharge,
    timeCharge,
    surgeMultiplier,
    demandFactor,
    weatherFactor: weatherData.multiplier,
    trafficFactor: trafficData.multiplier,
    totalPrice,
    distance,
    estimatedTime,
    vehicleType: 'UberGo',
    route
  };
}