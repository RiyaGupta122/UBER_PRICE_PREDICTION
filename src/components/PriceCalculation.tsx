import React, { useEffect, useState } from 'react';
import { Location, PriceBreakdown } from '../types';
import { calculatePrice } from '../utils/priceCalculation';
import { Car, Clock, MapPin, TrendingUp, Cloud, Navigation, IndianRupee } from 'lucide-react';

interface PriceCalculationProps {
  pickup: Location;
  dropoff: Location;
  onPriceCalculated: (breakdown: PriceBreakdown) => void;
}

const PriceCalculation: React.FC<PriceCalculationProps> = ({
  pickup,
  dropoff,
  onPriceCalculated,
}) => {
  const [breakdown, setBreakdown] = useState<PriceBreakdown | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (pickup && dropoff) {
      setLoading(true);
      // Simulate API call delay
      setTimeout(() => {
        const priceBreakdown = calculatePrice(pickup, dropoff);
        setBreakdown(priceBreakdown);
        onPriceCalculated(priceBreakdown);
        setLoading(false);
      }, 1500);
    }
  }, [pickup, dropoff, onPriceCalculated]);

  if (loading) {
    return (
      <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6">
        <div className="animate-pulse">
          <div className="flex items-center space-x-3 mb-6">
            <div className="w-10 h-10 bg-gray-200 rounded-lg"></div>
            <div className="space-y-2">
              <div className="h-4 bg-gray-200 rounded w-32"></div>
              <div className="h-3 bg-gray-200 rounded w-48"></div>
            </div>
          </div>
          <div className="space-y-4">
            <div className="h-8 bg-gray-200 rounded"></div>
            <div className="h-6 bg-gray-200 rounded"></div>
            <div className="h-6 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!breakdown) return null;

  const factors = [
    {
      icon: MapPin,
      label: 'Distance Charge',
      value: `₹${breakdown.distanceCharge.toFixed(2)}`,
      detail: `${breakdown.distance.toFixed(1)} km`,
      color: 'blue',
    },
    {
      icon: Clock,
      label: 'Time Charge',
      value: `₹${breakdown.timeCharge.toFixed(2)}`,
      detail: `${breakdown.estimatedTime} min`,
      color: 'green',
    },
    {
      icon: TrendingUp,
      label: 'Demand Surge',
      value: `${breakdown.surgeMultiplier}x`,
      detail: `${breakdown.demandFactor > 1 ? 'High' : 'Normal'} demand`,
      color: 'orange',
    },
    {
      icon: Cloud,
      label: 'Weather Factor',
      value: `${breakdown.weatherFactor}x`,
      detail: breakdown.weatherFactor > 1 ? 'Weather impact' : 'Clear weather',
      color: 'purple',
    },
  ];

  return (
    <div className="space-y-6">
      {/* Price Summary */}
      <div className="bg-gradient-to-r from-orange-500 to-green-500 rounded-2xl shadow-lg text-white p-6">
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center space-x-2 mb-2">
              <Car className="w-6 h-6" />
              <span className="text-lg font-semibold">{breakdown.vehicleType}</span>
            </div>
            <div className="flex items-center space-x-1">
              <IndianRupee className="w-8 h-8" />
              <span className="text-4xl font-bold">{breakdown.totalPrice.toFixed(0)}</span>
            </div>
            <p className="text-orange-100 mt-1">Estimated fare</p>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold">{breakdown.distance.toFixed(1)} km</div>
            <div className="text-orange-100">{breakdown.estimatedTime} minutes</div>
          </div>
        </div>
      </div>

      {/* Price Breakdown */}
      <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-500 rounded-lg flex items-center justify-center">
            <Navigation className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-xl font-semibold text-gray-900">Price Breakdown</h3>
            <p className="text-sm text-gray-600">Detailed fare calculation</p>
          </div>
        </div>

        <div className="space-y-4">
          {/* Base Price */}
          <div className="flex justify-between items-center py-3 border-b border-gray-100">
            <span className="font-medium text-gray-700">Base Price</span>
            <span className="font-semibold text-gray-900">₹{breakdown.basePrice.toFixed(2)}</span>
          </div>

          {/* Dynamic Factors */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {factors.map((factor, index) => {
              const Icon = factor.icon;
              const colorClasses = {
                blue: 'from-blue-500 to-blue-600',
                green: 'from-green-500 to-green-600',
                orange: 'from-orange-500 to-orange-600',
                purple: 'from-purple-500 to-purple-600',
              };

              return (
                <div key={index} className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center space-x-3">
                    <div className={`w-8 h-8 bg-gradient-to-r ${colorClasses[factor.color as keyof typeof colorClasses]} rounded-lg flex items-center justify-center`}>
                      <Icon className="w-4 h-4 text-white" />
                    </div>
                    <div className="flex-1">
                      <div className="flex justify-between items-center">
                        <span className="text-sm font-medium text-gray-700">{factor.label}</span>
                        <span className="font-semibold text-gray-900">{factor.value}</span>
                      </div>
                      <div className="text-xs text-gray-600 mt-1">{factor.detail}</div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Total */}
          <div className="flex justify-between items-center py-4 border-t-2 border-gray-200 mt-6">
            <span className="text-lg font-semibold text-gray-900">Total Fare</span>
            <div className="flex items-center space-x-1">
              <IndianRupee className="w-5 h-5 text-gray-900" />
              <span className="text-2xl font-bold text-gray-900">{breakdown.totalPrice.toFixed(2)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Route Information */}
      <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Route Information</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{breakdown.route.distance.toFixed(1)}</div>
            <div className="text-sm text-gray-600">Kilometers</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{breakdown.route.duration}</div>
            <div className="text-sm text-gray-600">Minutes</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">{breakdown.trafficFactor.toFixed(1)}x</div>
            <div className="text-sm text-gray-600">Traffic Factor</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">{breakdown.surgeMultiplier.toFixed(1)}x</div>
            <div className="text-sm text-gray-600">Surge Multiplier</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PriceCalculation;