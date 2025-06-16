import React from 'react';
import { PriceBreakdown } from '../types';
import { BarChart3, TrendingUp, Users, Clock, MapPin, IndianRupee } from 'lucide-react';

interface DashboardProps {
  priceBreakdown: PriceBreakdown | null;
}

const Dashboard: React.FC<DashboardProps> = ({ priceBreakdown }) => {
  const mockAnalytics = {
    totalRides: 12847,
    averageFare: 245,
    peakHours: '6-9 PM',
    popularRoute: 'Mumbai Central → BKC',
    surgeAreas: ['Bandra', 'Powai', 'Andheri'],
    demandForecast: 'High',
  };

  const priceDistribution = [
    { range: '₹0-100', percentage: 25, color: 'bg-green-500' },
    { range: '₹100-250', percentage: 35, color: 'bg-blue-500' },
    { range: '₹250-500', percentage: 25, color: 'bg-orange-500' },
    { range: '₹500+', percentage: 15, color: 'bg-red-500' },
  ];

  const demandPatterns = [
    { time: '6 AM', demand: 20 },
    { time: '9 AM', demand: 85 },
    { time: '12 PM', demand: 60 },
    { time: '3 PM', demand: 45 },
    { time: '6 PM', demand: 95 },
    { time: '9 PM', demand: 70 },
    { time: '12 AM', demand: 25 },
  ];

  return (
    <div className="space-y-6">
      {/* Analytics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Rides</p>
              <p className="text-2xl font-bold text-gray-900">{mockAnalytics.totalRides.toLocaleString()}</p>
            </div>
            <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-blue-600 rounded-lg flex items-center justify-center">
              <Users className="w-6 h-6 text-white" />
            </div>
          </div>
          <p className="text-xs text-green-600 mt-2">↗ +12% from last month</p>
        </div>

        <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Avg Fare</p>
              <p className="text-2xl font-bold text-gray-900">₹{mockAnalytics.averageFare}</p>
            </div>
            <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-green-600 rounded-lg flex items-center justify-center">
              <IndianRupee className="w-6 h-6 text-white" />
            </div>
          </div>
          <p className="text-xs text-green-600 mt-2">↗ +5% from last month</p>
        </div>

        <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Peak Hours</p>
              <p className="text-2xl font-bold text-gray-900">{mockAnalytics.peakHours}</p>
            </div>
            <div className="w-12 h-12 bg-gradient-to-r from-orange-500 to-orange-600 rounded-lg flex items-center justify-center">
              <Clock className="w-6 h-6 text-white" />
            </div>
          </div>
          <p className="text-xs text-gray-600 mt-2">Mumbai time zone</p>
        </div>

        <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Demand</p>
              <p className="text-2xl font-bold text-gray-900">{mockAnalytics.demandForecast}</p>
            </div>
            <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-purple-600 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
          </div>
          <p className="text-xs text-red-600 mt-2">↗ Peak demand period</p>
        </div>
      </div>

      {/* Current Trip Analysis (if available) */}
      {priceBreakdown && (
        <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Current Trip Analysis</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">₹{priceBreakdown.totalPrice.toFixed(0)}</div>
              <div className="text-sm text-blue-800">Total Fare</div>
            </div>
            <div className="text-center p-4 bg-gradient-to-br from-green-50 to-green-100 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{priceBreakdown.distance.toFixed(1)}km</div>
              <div className="text-sm text-green-800">Distance</div>
            </div>
            <div className="text-center p-4 bg-gradient-to-br from-orange-50 to-orange-100 rounded-lg">
              <div className="text-2xl font-bold text-orange-600">{priceBreakdown.surgeMultiplier}x</div>
              <div className="text-sm text-orange-800">Surge Factor</div>
            </div>
            <div className="text-center p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">{priceBreakdown.estimatedTime}min</div>
              <div className="text-sm text-purple-800">ETA</div>
            </div>
          </div>
        </div>
      )}

      {/* Price Distribution */}
      <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center">
            <BarChart3 className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Price Distribution</h3>
            <p className="text-sm text-gray-600">Fare ranges across all rides</p>
          </div>
        </div>

        <div className="space-y-4">
          {priceDistribution.map((range, index) => (
            <div key={index} className="flex items-center space-x-4">
              <div className="w-20 text-sm font-medium text-gray-700">{range.range}</div>
              <div className="flex-1 bg-gray-200 rounded-full h-3">
                <div
                  className={`h-3 rounded-full ${range.color}`}
                  style={{ width: `${range.percentage}%` }}
                ></div>
              </div>
              <div className="w-12 text-sm font-medium text-gray-600">{range.percentage}%</div>
            </div>
          ))}
        </div>
      </div>

      {/* Demand Patterns */}
      <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-r from-pink-500 to-red-500 rounded-lg flex items-center justify-center">
            <TrendingUp className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Demand Patterns</h3>
            <p className="text-sm text-gray-600">Hourly demand throughout the day</p>
          </div>
        </div>

        <div className="grid grid-cols-7 gap-2">
          {demandPatterns.map((pattern, index) => (
            <div key={index} className="text-center">
              <div className="bg-gray-100 rounded-lg p-2 mb-2">
                <div
                  className="bg-gradient-to-t from-orange-500 to-red-500 rounded"
                  style={{ height: `${pattern.demand}px` }}
                ></div>
              </div>
              <div className="text-xs font-medium text-gray-600">{pattern.time}</div>
              <div className="text-xs text-gray-500">{pattern.demand}%</div>
            </div>
          ))}
        </div>
      </div>

      {/* Popular Routes & Surge Areas */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6">
          <div className="flex items-center space-x-3 mb-4">
            <MapPin className="w-5 h-5 text-blue-500" />
            <h3 className="text-lg font-semibold text-gray-900">Popular Route</h3>
          </div>
          <div className="text-center py-8">
            <div className="text-2xl font-bold text-gray-900 mb-2">{mockAnalytics.popularRoute}</div>
            <div className="text-sm text-gray-600">Most frequent route today</div>
            <div className="mt-4 inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
              1,247 rides today
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6">
          <div className="flex items-center space-x-3 mb-4">
            <TrendingUp className="w-5 h-5 text-red-500" />
            <h3 className="text-lg font-semibold text-gray-900">Surge Areas</h3>
          </div>
          <div className="space-y-3">
            {mockAnalytics.surgeAreas.map((area, index) => (
              <div key={index} className="flex items-center justify-between">
                <span className="text-gray-700">{area}</span>
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
                  {(1.5 + index * 0.3).toFixed(1)}x surge
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;