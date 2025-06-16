import React, { useState } from 'react';
import LocationInput from './components/LocationInput';
import PriceCalculation from './components/PriceCalculation';
import AlgorithmExplanation from './components/AlgorithmExplanation';
import Dashboard from './components/Dashboard';
import { Location, PriceBreakdown } from './types';
import { Car, MapPin, Calculator, BookOpen } from 'lucide-react';

function App() {
  const [pickup, setPickup] = useState<Location | null>(null);
  const [dropoff, setDropoff] = useState<Location | null>(null);
  const [activeTab, setActiveTab] = useState<'calculator' | 'dashboard' | 'algorithms'>('calculator');
  const [priceBreakdown, setPriceBreakdown] = useState<PriceBreakdown | null>(null);

  const tabs = [
    { id: 'calculator', label: 'Price Calculator', icon: Calculator },
    { id: 'dashboard', label: 'Analytics Dashboard', icon: Car },
    { id: 'algorithms', label: 'DSA Concepts', icon: BookOpen },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 via-white to-green-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-orange-100">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-orange-500 to-green-500 rounded-lg flex items-center justify-center">
                <Car className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Uber Price Optimization</h1>
                <p className="text-sm text-gray-600">Advanced DSA & Data Science for Indian Markets</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 text-sm text-gray-600">
              <MapPin className="w-4 h-4" />
              <span className="font-medium">🇮🇳 Indian Pricing Model</span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as any)}
                  className={`py-4 px-1 inline-flex items-center space-x-2 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'border-orange-500 text-orange-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'calculator' && (
          <div className="space-y-8">
            <LocationInput
              pickup={pickup}
              dropoff={dropoff}
              onPickupChange={setPickup}
              onDropoffChange={setDropoff}
            />
            {pickup && dropoff && (
              <PriceCalculation
                pickup={pickup}
                dropoff={dropoff}
                onPriceCalculated={setPriceBreakdown}
              />
            )}
          </div>
        )}

        {activeTab === 'dashboard' && (
          <Dashboard priceBreakdown={priceBreakdown} />
        )}

        {activeTab === 'algorithms' && (
          <AlgorithmExplanation />
        )}
      </main>
    </div>
  );
}

export default App;