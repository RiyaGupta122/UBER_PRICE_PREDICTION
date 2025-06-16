export interface Location {
  name: string;
  coordinates: {
    lat: number;
    lng: number;
  };
  address: string;
}

export interface PriceBreakdown {
  basePrice: number;
  distanceCharge: number;
  timeCharge: number;
  surgeMultiplier: number;
  demandFactor: number;
  weatherFactor: number;
  trafficFactor: number;
  totalPrice: number;
  distance: number;
  estimatedTime: number;
  vehicleType: string;
  route: {
    coordinates: Array<{lat: number, lng: number}>;
    distance: number;
    duration: number;
  };
}

export interface IndianCity {
  name: string;
  coordinates: {
    lat: number;
    lng: number;
  };
  basePricePerKm: number;
  basePricePerMinute: number;
  minimumFare: number;
}

export interface TrafficData {
  level: 'low' | 'medium' | 'high';
  multiplier: number;
  description: string;
}

export interface WeatherData {
  condition: 'clear' | 'rain' | 'heavy_rain' | 'fog';
  multiplier: number;
  description: string;
}