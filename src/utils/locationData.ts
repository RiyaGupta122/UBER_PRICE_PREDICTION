import { Location, IndianCity } from '../types';

export const indianCities: IndianCity[] = [
  {
    name: 'Mumbai',
    coordinates: { lat: 19.0760, lng: 72.8777 },
    basePricePerKm: 12,
    basePricePerMinute: 2,
    minimumFare: 50,
  },
  {
    name: 'Delhi',
    coordinates: { lat: 28.7041, lng: 77.1025 },
    basePricePerKm: 10,
    basePricePerMinute: 1.8,
    minimumFare: 45,
  },
  {
    name: 'Bangalore',
    coordinates: { lat: 12.9716, lng: 77.5946 },
    basePricePerKm: 11,
    basePricePerMinute: 1.9,
    minimumFare: 48,
  },
  {
    name: 'Hyderabad',
    coordinates: { lat: 17.3850, lng: 78.4867 },
    basePricePerKm: 9,
    basePricePerMinute: 1.6,
    minimumFare: 40,
  },
  {
    name: 'Chennai',
    coordinates: { lat: 13.0827, lng: 80.2707 },
    basePricePerKm: 10,
    basePricePerMinute: 1.7,
    minimumFare: 42,
  },
  {
    name: 'Pune',
    coordinates: { lat: 18.5204, lng: 73.8567 },
    basePricePerKm: 11,
    basePricePerMinute: 1.8,
    minimumFare: 45,
  },
];

export const getPopularDestinations = (): Location[] => {
  return [
    // Mumbai locations
    {
      name: 'Mumbai Central',
      coordinates: { lat: 18.9690, lng: 72.8205 },
      address: 'Mumbai Central Railway Station, Mumbai, Maharashtra',
    },
    {
      name: 'Bandra-Kurla Complex',
      coordinates: { lat: 19.0596, lng: 72.8656 },
      address: 'BKC, Bandra East, Mumbai, Maharashtra',
    },
    {
      name: 'Chhatrapati Shivaji Airport',
      coordinates: { lat: 19.0896, lng: 72.8656 },
      address: 'Mumbai Airport, Andheri East, Mumbai, Maharashtra',
    },
    {
      name: 'Gateway of India',
      coordinates: { lat: 18.9220, lng: 72.8347 },
      address: 'Gateway of India, Colaba, Mumbai, Maharashtra',
    },
    {
      name: 'Powai',
      coordinates: { lat: 19.1176, lng: 72.9060 },
      address: 'Powai, Mumbai, Maharashtra',
    },
    {
      name: 'Andheri West',
      coordinates: { lat: 19.1136, lng: 72.8697 },
      address: 'Andheri West, Mumbai, Maharashtra',
    },

    // Delhi locations
    {
      name: 'Connaught Place',
      coordinates: { lat: 28.6315, lng: 77.2167 },
      address: 'Connaught Place, New Delhi, Delhi',
    },
    {
      name: 'IGI Airport',
      coordinates: { lat: 28.5562, lng: 77.1000 },
      address: 'Indira Gandhi International Airport, Delhi',
    },
    {
      name: 'Gurgaon Cyber City',
      coordinates: { lat: 28.4595, lng: 77.0266 },
      address: 'Cyber City, Gurgaon, Haryana',
    },
    {
      name: 'Karol Bagh',
      coordinates: { lat: 28.6519, lng: 77.1909 },
      address: 'Karol Bagh, New Delhi, Delhi',
    },

    // Bangalore locations
    {
      name: 'Electronic City',
      coordinates: { lat: 12.8456, lng: 77.6603 },
      address: 'Electronic City, Bangalore, Karnataka',
    },
    {
      name: 'Whitefield',
      coordinates: { lat: 12.9698, lng: 77.7500 },
      address: 'Whitefield, Bangalore, Karnataka',
    },
    {
      name: 'Koramangala',
      coordinates: { lat: 12.9352, lng: 77.6245 },
      address: 'Koramangala, Bangalore, Karnataka',
    },
    {
      name: 'Bangalore Airport',
      coordinates: { lat: 13.1979, lng: 77.7068 },
      address: 'Kempegowda International Airport, Bangalore, Karnataka',
    },

    // Chennai locations
    {
      name: 'T. Nagar',
      coordinates: { lat: 13.0418, lng: 80.2341 },
      address: 'T. Nagar, Chennai, Tamil Nadu',
    },
    {
      name: 'Chennai Airport',
      coordinates: { lat: 12.9941, lng: 80.1709 },
      address: 'Chennai International Airport, Chennai, Tamil Nadu',
    },
    {
      name: 'OMR IT Corridor',
      coordinates: { lat: 12.8406, lng: 80.2270 },
      address: 'Old Mahabalipuram Road, Chennai, Tamil Nadu',
    },

    // Hyderabad locations
    {
      name: 'HITEC City',
      coordinates: { lat: 17.4435, lng: 78.3772 },
      address: 'HITEC City, Hyderabad, Telangana',
    },
    {
      name: 'Hyderabad Airport',
      coordinates: { lat: 17.2403, lng: 78.4294 },
      address: 'Rajiv Gandhi International Airport, Hyderabad, Telangana',
    },
    {
      name: 'Banjara Hills',
      coordinates: { lat: 17.4126, lng: 78.4482 },
      address: 'Banjara Hills, Hyderabad, Telangana',
    },

    // Pune locations
    {
      name: 'Hinjewadi',
      coordinates: { lat: 18.5912, lng: 73.7389 },
      address: 'Hinjewadi IT Park, Pune, Maharashtra',
    },
    {
      name: 'Pune Airport',
      coordinates: { lat: 18.5821, lng: 73.9197 },
      address: 'Pune Airport, Lohegaon, Pune, Maharashtra',
    },
    {
      name: 'Koregaon Park',
      coordinates: { lat: 18.5362, lng: 73.8980 },
      address: 'Koregaon Park, Pune, Maharashtra',
    },
  ];
};