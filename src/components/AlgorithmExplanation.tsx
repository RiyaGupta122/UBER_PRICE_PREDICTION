import React, { useState } from 'react';
import { 
  Code, 
  GitBranch, 
  Zap, 
  Target, 
  Clock, 
  TrendingUp, 
  Map,
  Calculator,
  Database,
  Cpu
} from 'lucide-react';

const AlgorithmExplanation: React.FC = () => {
  const [activeAlgorithm, setActiveAlgorithm] = useState('dijkstra');

  const algorithms = [
    {
      id: 'dijkstra',
      name: 'Dijkstra\'s Algorithm',
      icon: Map,
      category: 'Shortest Path',
      description: 'Finds the shortest path between pickup and drop locations',
      complexity: 'O((V + E) log V)',
      implementation: `// Dijkstra's Algorithm for shortest path
function dijkstra(graph, start, end) {
  const distances = {};
  const previous = {};
  const unvisited = new Set();
  
  // Initialize distances
  for (let node in graph) {
    distances[node] = node === start ? 0 : Infinity;
    unvisited.add(node);
  }
  
  while (unvisited.size > 0) {
    // Find minimum distance node
    let current = null;
    let minDistance = Infinity;
    
    for (let node of unvisited) {
      if (distances[node] < minDistance) {
        minDistance = distances[node];
        current = node;
      }
    }
    
    if (current === end) break;
    
    unvisited.delete(current);
    
    // Update neighbors
    for (let neighbor in graph[current]) {
      const alt = distances[current] + graph[current][neighbor];
      if (alt < distances[neighbor]) {
        distances[neighbor] = alt;
        previous[neighbor] = current;
      }
    }
  }
  
  return { distance: distances[end], path: reconstructPath(previous, end) };
}`,
      usage: 'Used to calculate the optimal route between pickup and drop locations, considering traffic and road conditions.',
      benefits: [
        'Guarantees shortest path',
        'Handles weighted graphs (traffic conditions)',
        'Efficient for sparse graphs',
        'Widely tested and reliable'
      ]
    },
    {
      id: 'dynamic-pricing',
      name: 'Dynamic Pricing Algorithm',
      icon: TrendingUp,
      category: 'Pricing Optimization',
      description: 'Calculates surge pricing based on supply-demand dynamics',
      complexity: 'O(1) - Constant time',
      implementation: `// Dynamic Pricing Algorithm
function calculateDynamicPrice(basePrice, factors) {
  const {
    demandLevel,      // 0.5 to 3.0
    supplyLevel,      // 0.5 to 2.0
    weatherCondition, // 1.0 to 1.5
    timeOfDay,        // 0.8 to 1.3
    specialEvents,    // 1.0 to 2.0
    trafficLevel      // 1.0 to 1.8
  } = factors;
  
  // Supply-Demand multiplier
  const supplyDemandRatio = demandLevel / supplyLevel;
  let surgeMultiplier = Math.max(1.0, supplyDemandRatio);
  
  // Apply exponential surge for high demand
  if (surgeMultiplier > 2.0) {
    surgeMultiplier = 2.0 + Math.pow(surgeMultiplier - 2.0, 1.5);
  }
  
  // Environmental factors
  const environmentalMultiplier = 
    weatherCondition * timeOfDay * specialEvents * trafficLevel;
  
  // Final price calculation
  const finalPrice = basePrice * surgeMultiplier * environmentalMultiplier;
  
  // Price elasticity adjustment (avoid extreme prices)
  const maxSurge = 5.0;
  const adjustedPrice = Math.min(finalPrice, basePrice * maxSurge);
  
  return {
    basePrice,
    surgeMultiplier: Math.min(surgeMultiplier, maxSurge),
    environmentalMultiplier,
    finalPrice: adjustedPrice
  };
}`,
      usage: 'Dynamically adjusts ride prices based on real-time supply-demand, weather, events, and traffic conditions.',
      benefits: [
        'Maximizes revenue during peak demand',
        'Balances supply and demand',
        'Responsive to market conditions',
        'Prevents price manipulation'
      ]
    },
    {
      id: 'heap',
      name: 'Priority Queue (Heap)',
      icon: Zap,
      category: 'Data Structure',
      description: 'Manages driver allocation and ride requests efficiently',
      complexity: 'O(log n) insert/delete',
      implementation: `// Priority Queue for Driver Allocation
class DriverQueue {
  constructor() {
    this.heap = [];
  }
  
  // Add driver with priority (distance + availability score)
  addDriver(driver) {
    const priority = this.calculatePriority(driver);
    this.heap.push({ driver, priority });
    this.heapifyUp(this.heap.length - 1);
  }
  
  // Get nearest available driver
  getNearestDriver() {
    if (this.heap.length === 0) return null;
    
    const nearest = this.heap[0];
    const last = this.heap.pop();
    
    if (this.heap.length > 0) {
      this.heap[0] = last;
      this.heapifyDown(0);
    }
    
    return nearest.driver;
  }
  
  calculatePriority(driver) {
    const {
      distanceToPickup,  // km
      rating,           // 1-5
      isAvailable,      // boolean
      vehicleType,      // efficiency score
      completedRides    // experience factor
    } = driver;
    
    if (!isAvailable) return Infinity;
    
    // Lower priority value = higher priority
    const distanceFactor = distanceToPickup;
    const ratingFactor = (5 - rating) * 0.5; // Better rating = lower value
    const experienceFactor = Math.max(0, (100 - completedRides) * 0.01);
    
    return distanceFactor + ratingFactor + experienceFactor;
  }
  
  heapifyUp(index) {
    if (index === 0) return;
    
    const parentIndex = Math.floor((index - 1) / 2);
    if (this.heap[parentIndex].priority > this.heap[index].priority) {
      [this.heap[parentIndex], this.heap[index]] = 
        [this.heap[index], this.heap[parentIndex]];
      this.heapifyUp(parentIndex);
    }
  }
  
  heapifyDown(index) {
    const leftChild = 2 * index + 1;
    const rightChild = 2 * index + 2;
    let smallest = index;
    
    if (leftChild < this.heap.length && 
        this.heap[leftChild].priority < this.heap[smallest].priority) {
      smallest = leftChild;
    }
    
    if (rightChild < this.heap.length && 
        this.heap[rightChild].priority < this.heap[smallest].priority) {
      smallest = rightChild;
    }
    
    if (smallest !== index) {
      [this.heap[index], this.heap[smallest]] = 
        [this.heap[smallest], this.heap[index]];
      this.heapifyDown(smallest);
    }
  }
}`,
      usage: 'Efficiently manages driver allocation by maintaining a priority queue of available drivers sorted by distance and rating.',
      benefits: [
        'O(log n) insertion and removal',
        'Always provides optimal driver match',
        'Handles dynamic driver availability',
        'Scales well with large driver pools'
      ]
    },
    {
      id: 'haversine',
      name: 'Haversine Formula',
      icon: Target,
      category: 'Geolocation',
      description: 'Calculates accurate distances between geographical coordinates',
      complexity: 'O(1) - Constant time',
      implementation: `// Haversine Formula for distance calculation
function haversineDistance(lat1, lon1, lat2, lon2) {
  const R = 6371; // Earth's radius in kilometers
  
  // Convert latitude and longitude to radians
  const dLat = toRadians(lat2 - lat1);
  const dLon = toRadians(lon2 - lon1);
  
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRadians(lat1)) * Math.cos(toRadians(lat2)) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  
  return R * c; // Distance in kilometers
}

function toRadians(degrees) {
  return degrees * (Math.PI / 180);
}

// Enhanced distance calculation with Indian road factors
function calculateIndianRoadDistance(lat1, lon1, lat2, lon2, cityType = 'metro') {
  const straightLineDistance = haversineDistance(lat1, lon1, lat2, lon2);
  
  // Indian road factors (accounts for traffic, road quality, detours)
  const roadFactors = {
    metro: 1.4,     // Mumbai, Delhi, Bangalore
    tier1: 1.5,     // Pune, Hyderabad, Chennai  
    tier2: 1.7,     // Indore, Bhopal, Coimbatore
    rural: 2.2      // Rural and highway connections
  };
  
  const actualDistance = straightLineDistance * roadFactors[cityType];
  
  return {
    straightLine: straightLineDistance,
    actual: actualDistance,
    factor: roadFactors[cityType]
  };
}`,
      usage: 'Calculates precise distances between pickup and drop locations using GPS coordinates, essential for accurate pricing.',
      benefits: [
        'High accuracy for geographical distances',
        'Accounts for Earth\'s curvature',
        'Optimized for Indian road conditions',
        'Fast computation for real-time usage'
      ]
    },
    {
      id: 'eta-prediction',
      name: 'ETA Prediction Algorithm',
      icon: Clock,
      category: 'Time Estimation',
      description: 'Predicts arrival time using historical data and machine learning',
      complexity: 'O(log n) for tree traversal',
      implementation: `// ETA Prediction using Decision Tree and Historical Data
class ETAPredictor {
  constructor() {
    this.historicalData = new Map();
    this.trafficPatterns = new Map();
    this.weatherImpact = new Map();
  }
  
  predictETA(route, currentConditions) {
    const {
      distance,
      startTime,
      dayOfWeek,
      weather,
      specialEvents,
      routeSegments
    } = route;
    
    let totalTime = 0;
    
    // Segment-based ETA calculation
    for (const segment of routeSegments) {
      const segmentTime = this.calculateSegmentTime(segment, currentConditions);
      totalTime += segmentTime;
    }
    
    // Apply ML corrections based on historical accuracy
    const mlCorrection = this.getMLCorrection(route, currentConditions);
    const adjustedTime = totalTime * mlCorrection;
    
    return {
      estimatedMinutes: Math.round(adjustedTime),
      confidence: this.calculateConfidence(route, currentConditions),
      breakdown: this.getTimeBreakdown(routeSegments, currentConditions)
    };
  }
  
  calculateSegmentTime(segment, conditions) {
    const baseSpeed = this.getBaseSpeed(segment.roadType);
    const trafficMultiplier = this.getTrafficMultiplier(segment, conditions);
    const weatherMultiplier = this.getWeatherMultiplier(conditions.weather);
    
    const adjustedSpeed = baseSpeed * trafficMultiplier * weatherMultiplier;
    return (segment.distance / adjustedSpeed) * 60; // Convert to minutes
  }
  
  getBaseSpeed(roadType) {
    const speedLimits = {
      highway: 80,    // km/h
      arterial: 50,   // Major city roads
      collector: 35,  // Secondary roads
      local: 25,      // Local streets
      congested: 15   // Heavy traffic areas
    };
    return speedLimits[roadType] || 30;
  }
  
  getTrafficMultiplier(segment, conditions) {
    const { timeOfDay, dayOfWeek } = conditions;
    const hour = new Date(timeOfDay).getHours();
    
    // Peak hours traffic impact
    const peakHours = [8, 9, 18, 19, 20]; // 8-9 AM, 6-8 PM
    const isPeakHour = peakHours.includes(hour);
    const isWeekend = [0, 6].includes(dayOfWeek); // Sunday, Saturday
    
    let multiplier = 1.0;
    
    if (isPeakHour && !isWeekend) {
      multiplier = 0.4; // 60% speed reduction during peak hours
    } else if (isPeakHour && isWeekend) {
      multiplier = 0.7; // 30% speed reduction on weekend peak hours
    } else if (!isPeakHour && !isWeekend) {
      multiplier = 0.8; // 20% speed reduction during normal hours
    } else {
      multiplier = 0.9; // 10% speed reduction on weekend off-peak
    }
    
    return multiplier;
  }
  
  getMLCorrection(route, conditions) {
    // Simplified ML model - in production would use trained models
    const features = this.extractFeatures(route, conditions);
    const prediction = this.neuralNetworkPredict(features);
    
    // Bound the correction to reasonable limits
    return Math.max(0.7, Math.min(1.5, prediction));
  }
  
  calculateConfidence(route, conditions) {
    const factors = {
      historicalDataPoints: this.getHistoricalDataPoints(route),
      weatherCertainty: conditions.weather === 'clear' ? 0.9 : 0.7,
      trafficPredictability: this.getTrafficPredictability(route),
      routeComplexity: route.segments.length > 5 ? 0.8 : 0.9
    };
    
    const weightedConfidence = Object.values(factors).reduce((a, b) => a * b, 1);
    return Math.round(weightedConfidence * 100);
  }
}`,
      usage: 'Provides accurate ETA predictions by analyzing historical traffic patterns, current conditions, and route complexity.',
      benefits: [
        'Machine learning enhanced accuracy',
        'Real-time condition adaptation',
        'Segment-based granular analysis',
        'Confidence scoring for reliability'
      ]
    },
    {
      id: 'matching',
      name: 'Bipartite Matching',
      icon: GitBranch,
      category: 'Optimization',
      description: 'Optimally matches riders with drivers using graph algorithms',
      complexity: 'O(V²E) Hungarian Algorithm',
      implementation: `// Bipartite Matching for Rider-Driver Assignment
class RiderDriverMatcher {
  constructor() {
    this.riders = [];
    this.drivers = [];
    this.costMatrix = [];
  }
  
  // Hungarian Algorithm for optimal assignment
  findOptimalMatching() {
    const costMatrix = this.buildCostMatrix();
    const assignment = this.hungarianAlgorithm(costMatrix);
    
    return this.createMatching(assignment);
  }
  
  buildCostMatrix() {
    const matrix = [];
    
    for (let i = 0; i < this.riders.length; i++) {
      matrix[i] = [];
      const rider = this.riders[i];
      
      for (let j = 0; j < this.drivers.length; j++) {
        const driver = this.drivers[j];
        matrix[i][j] = this.calculateMatchingCost(rider, driver);
      }
    }
    
    return matrix;
  }
  
  calculateMatchingCost(rider, driver) {
    if (!driver.isAvailable) return Infinity;
    
    const factors = {
      // Distance cost (primary factor)
      distance: this.haversineDistance(
        rider.pickup.lat, rider.pickup.lng,
        driver.location.lat, driver.location.lng
      ),
      
      // Wait time cost
      waitTime: this.estimateWaitTime(rider, driver),
      
      // Vehicle type preference
      vehiclePreference: this.getVehiclePreferenceCost(rider, driver),
      
      // Driver rating impact
      ratingFactor: (5 - driver.rating) * 2,
      
      // Special requirements
      specialRequirements: this.checkSpecialRequirements(rider, driver)
    };
    
    // Weighted cost calculation
    const weights = {
      distance: 0.4,
      waitTime: 0.3,
      vehiclePreference: 0.15,
      ratingFactor: 0.1,
      specialRequirements: 0.05
    };
    
    let totalCost = 0;
    for (const [factor, value] of Object.entries(factors)) {
      totalCost += value * weights[factor];
    }
    
    return totalCost;
  }
  
  hungarianAlgorithm(costMatrix) {
    const n = costMatrix.length;
    const m = costMatrix[0].length;
    
    // Step 1: Subtract row minimums
    for (let i = 0; i < n; i++) {
      const rowMin = Math.min(...costMatrix[i]);
      for (let j = 0; j < m; j++) {
        costMatrix[i][j] -= rowMin;
      }
    }
    
    // Step 2: Subtract column minimums
    for (let j = 0; j < m; j++) {
      const colMin = Math.min(...costMatrix.map(row => row[j]));
      for (let i = 0; i < n; i++) {
        costMatrix[i][j] -= colMin;
      }
    }
    
    // Step 3: Find optimal assignment using augmenting paths
    const assignment = this.findAugmentingPaths(costMatrix);
    
    return assignment;
  }
  
  findAugmentingPaths(matrix) {
    const n = matrix.length;
    const assignment = new Array(n).fill(-1);
    const matched = new Array(matrix[0].length).fill(false);
    
    for (let rider = 0; rider < n; rider++) {
      const visited = new Array(matrix[0].length).fill(false);
      this.dfs(matrix, rider, visited, assignment, matched);
    }
    
    return assignment;
  }
  
  dfs(matrix, rider, visited, assignment, matched) {
    for (let driver = 0; driver < matrix[0].length; driver++) {
      if (matrix[rider][driver] === 0 && !visited[driver]) {
        visited[driver] = true;
        
        if (!matched[driver] || 
            this.dfs(matrix, assignment.indexOf(driver), visited, assignment, matched)) {
          assignment[rider] = driver;
          matched[driver] = true;
          return true;
        }
      }
    }
    return false;
  }
  
  createMatching(assignment) {
    const matches = [];
    
    for (let i = 0; i < assignment.length; i++) {
      if (assignment[i] !== -1) {
        const rider = this.riders[i];
        const driver = this.drivers[assignment[i]];
        
        matches.push({
          rider: rider,
          driver: driver,
          estimatedPickupTime: this.calculatePickupTime(rider, driver),
          matchScore: this.calculateMatchScore(rider, driver)
        });
      }
    }
    
    return matches;
  }
}`,
      usage: 'Optimally assigns multiple riders to drivers simultaneously, minimizing overall wait times and maximizing system efficiency.',
      benefits: [
        'Globally optimal assignments',
        'Handles multiple constraints',
        'Maximizes system efficiency',
        'Scales to large rider/driver pools'
      ]
    }
  ];

  const selectedAlg = algorithms.find(alg => alg.id === activeAlgorithm);

  return (
    <div className="space-y-6">
      {/* Algorithm Selection */}
      <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
            <Code className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-gray-900">DSA Algorithms in Uber Optimization</h2>
            <p className="text-sm text-gray-600">Explore the core algorithms powering ride optimization</p>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {algorithms.map((algorithm) => {
            const Icon = algorithm.icon;
            return (
              <button
                key={algorithm.id}
                onClick={() => setActiveAlgorithm(algorithm.id)}
                className={`p-4 rounded-lg border-2 transition-all text-left ${
                  activeAlgorithm === algorithm.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300 bg-white'
                }`}
              >
                <Icon className={`w-6 h-6 mb-2 ${
                  activeAlgorithm === algorithm.id ? 'text-blue-600' : 'text-gray-600'
                }`} />
                <div className={`font-medium text-sm ${
                  activeAlgorithm === algorithm.id ? 'text-blue-900' : 'text-gray-900'
                }`}>
                  {algorithm.name}
                </div>
                <div className="text-xs text-gray-600 mt-1">{algorithm.category}</div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Selected Algorithm Details */}
      {selectedAlg && (
        <div className="space-y-6">
          {/* Overview */}
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6">
            <div className="flex items-start space-x-4">
              <div className="w-12 h-12 bg-gradient-to-r from-indigo-500 to-purple-500 rounded-lg flex items-center justify-center flex-shrink-0">
                <selectedAlg.icon className="w-6 h-6 text-white" />
              </div>
              <div className="flex-1">
                <div className="flex items-center space-x-3 mb-2">
                  <h3 className="text-2xl font-bold text-gray-900">{selectedAlg.name}</h3>
                  <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                    {selectedAlg.category}
                  </span>
                </div>
                <p className="text-gray-600 mb-4">{selectedAlg.description}</p>
                <div className="flex items-center space-x-6">
                  <div className="flex items-center space-x-2">
                    <Cpu className="w-4 h-4 text-gray-500" />
                    <span className="text-sm text-gray-700">
                      <strong>Complexity:</strong> {selectedAlg.complexity}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Implementation */}
          <div className="bg-white rounded-2xl shadow-lg border border-gray-100 overflow-hidden">
            <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
              <div className="flex items-center space-x-3">
                <Database className="w-5 h-5 text-gray-600" />
                <h4 className="text-lg font-semibold text-gray-900">Implementation</h4>
              </div>
            </div>
            <div className="p-0">
              <pre className="bg-gray-900 text-gray-100 p-6 text-sm overflow-x-auto">
                <code>{selectedAlg.implementation}</code>
              </pre>
            </div>
          </div>

          {/* Usage and Benefits */}
          <div className="grid md:grid-cols-2 gap-6">
            <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6">
              <div className="flex items-center space-x-3 mb-4">
                <Target className="w-5 h-5 text-green-600" />
                <h4 className="text-lg font-semibold text-gray-900">Real-World Usage</h4>
              </div>
              <p className="text-gray-700">{selectedAlg.usage}</p>
            </div>

            <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6">
              <div className="flex items-center space-x-3 mb-4">
                <Zap className="w-5 h-5 text-orange-600" />
                <h4 className="text-lg font-semibold text-gray-900">Key Benefits</h4>
              </div>
              <ul className="space-y-2">
                {selectedAlg.benefits.map((benefit, index) => (
                  <li key={index} className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-orange-400 rounded-full"></div>
                    <span className="text-gray-700">{benefit}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* System Architecture Overview */}
      <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-lg flex items-center justify-center">
            <GitBranch className="w-5 h-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900">System Architecture Integration</h3>
            <p className="text-sm text-gray-600">How these algorithms work together in the Uber ecosystem</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg p-4">
            <h4 className="font-semibold text-blue-900 mb-2">1. Request Processing</h4>
            <p className="text-sm text-blue-800">
              Haversine formula calculates distances, while priority queues manage incoming requests efficiently.
            </p>
          </div>
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-4">
            <h4 className="font-semibold text-green-900 mb-2">2. Optimal Matching</h4>
            <p className="text-sm text-green-800">
              Bipartite matching algorithms assign riders to drivers, while Dijkstra finds optimal routes.
            </p>
          </div>
          <div className="bg-gradient-to-br from-orange-50 to-red-50 rounded-lg p-4">
            <h4 className="font-semibold text-orange-900 mb-2">3. Dynamic Pricing</h4>
            <p className="text-sm text-orange-800">
              ML-powered ETA prediction combines with surge pricing algorithms for real-time fare optimization.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AlgorithmExplanation;