import React, { useState } from 'react';
import { MapPin, Navigation, Clock, Search } from 'lucide-react';
import { Location } from '../types';
import { indianCities, getPopularDestinations } from '../utils/locationData';

interface LocationInputProps {
  pickup: Location | null;
  dropoff: Location | null;
  onPickupChange: (location: Location) => void;
  onDropoffChange: (location: Location) => void;
}

const LocationInput: React.FC<LocationInputProps> = ({
  pickup,
  dropoff,
  onPickupChange,
  onDropoffChange,
}) => {
  const [pickupSearch, setPickupSearch] = useState('');
  const [dropoffSearch, setDropoffSearch] = useState('');
  const [showPickupSuggestions, setShowPickupSuggestions] = useState(false);
  const [showDropoffSuggestions, setShowDropoffSuggestions] = useState(false);

  const popularDestinations = getPopularDestinations();

  const filterLocations = (search: string) => {
    if (!search) return popularDestinations.slice(0, 6);
    return popularDestinations.filter(location =>
      location.name.toLowerCase().includes(search.toLowerCase()) ||
      location.address.toLowerCase().includes(search.toLowerCase())
    ).slice(0, 6);
  };

  const handleLocationSelect = (location: Location, type: 'pickup' | 'dropoff') => {
    if (type === 'pickup') {
      onPickupChange(location);
      setPickupSearch(location.name);
      setShowPickupSuggestions(false);
    } else {
      onDropoffChange(location);
      setDropoffSearch(location.name);
      setShowDropoffSuggestions(false);
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6">
      <div className="flex items-center space-x-3 mb-6">
        <div className="w-10 h-10 bg-gradient-to-r from-green-500 to-blue-500 rounded-lg flex items-center justify-center">
          <Navigation className="w-5 h-5 text-white" />
        </div>
        <div>
          <h2 className="text-xl font-semibold text-gray-900">Trip Details</h2>
          <p className="text-sm text-gray-600">Enter pickup and drop locations</p>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        {/* Pickup Location */}
        <div className="relative">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <MapPin className="w-4 h-4 inline mr-1 text-green-500" />
            Pickup Location
          </label>
          <div className="relative">
            <input
              type="text"
              value={pickupSearch}
              onChange={(e) => {
                setPickupSearch(e.target.value);
                setShowPickupSuggestions(true);
              }}
              onFocus={() => setShowPickupSuggestions(true)}
              placeholder="Search pickup location..."
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all"
            />
            <Search className="absolute right-3 top-3 w-5 h-5 text-gray-400" />
          </div>
          
          {showPickupSuggestions && (
            <div className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto">
              {filterLocations(pickupSearch).map((location, index) => (
                <button
                  key={index}
                  onClick={() => handleLocationSelect(location, 'pickup')}
                  className="w-full px-4 py-3 text-left hover:bg-gray-50 border-b border-gray-100 last:border-b-0 transition-colors"
                >
                  <div className="font-medium text-gray-900">{location.name}</div>
                  <div className="text-sm text-gray-600">{location.address}</div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Dropoff Location */}
        <div className="relative">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            <MapPin className="w-4 h-4 inline mr-1 text-red-500" />
            Drop Location
          </label>
          <div className="relative">
            <input
              type="text"
              value={dropoffSearch}
              onChange={(e) => {
                setDropoffSearch(e.target.value);
                setShowDropoffSuggestions(true);
              }}
              onFocus={() => setShowDropoffSuggestions(true)}
              placeholder="Search drop location..."
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-transparent transition-all"
            />
            <Search className="absolute right-3 top-3 w-5 h-5 text-gray-400" />
          </div>
          
          {showDropoffSuggestions && (
            <div className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-60 overflow-y-auto">
              {filterLocations(dropoffSearch).map((location, index) => (
                <button
                  key={index}
                  onClick={() => handleLocationSelect(location, 'dropoff')}
                  className="w-full px-4 py-3 text-left hover:bg-gray-50 border-b border-gray-100 last:border-b-0 transition-colors"
                >
                  <div className="font-medium text-gray-900">{location.name}</div>
                  <div className="text-sm text-gray-600">{location.address}</div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Selected Locations Display */}
      {(pickup || dropoff) && (
        <div className="mt-6 pt-6 border-t border-gray-100">
          <div className="flex items-center space-x-4">
            {pickup && (
              <div className="flex items-center space-x-2 bg-green-50 px-3 py-2 rounded-lg">
                <MapPin className="w-4 h-4 text-green-600" />
                <span className="text-sm font-medium text-green-800">{pickup.name}</span>
              </div>
            )}
            {pickup && dropoff && (
              <div className="text-gray-400">
                <Clock className="w-4 h-4" />
              </div>
            )}
            {dropoff && (
              <div className="flex items-center space-x-2 bg-red-50 px-3 py-2 rounded-lg">
                <MapPin className="w-4 h-4 text-red-600" />
                <span className="text-sm font-medium text-red-800">{dropoff.name}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default LocationInput;