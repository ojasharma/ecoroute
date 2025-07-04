// types/route.ts

export interface Coordinate {
  lat: number;
  lng: number;
}

// Original RouteResponse (keep for backward compatibility)
export interface RouteResponse {
  path: Coordinate[];
}

// EcoRoute API Types
export interface EcoRouteRequest {
  origin: [number, number]; // [lat, lon]
  destination: [number, number]; // [lat, lon]
  vehicle: string;
}

export interface EcoRouteStats {
  distance_km: number;
  time_minutes: number;
  time_minutes_google?: number;
  co2_kg: number;
}

export interface EcoRouteComparison {
  co2_savings_kg: number;
  co2_savings_percent: number;
  time_difference_minutes: number;
}

export interface EcoRouteResponse {
  eco_route: [number, number][]; // Array of [lat, lon] coordinates
  google_route: [number, number][]; // Array of [lat, lon] coordinates
  eco_stats: EcoRouteStats;
  google_stats: EcoRouteStats;
  comparison: EcoRouteComparison;
}

export interface Vehicle {
  id: string;
  name: string;
}

export interface VehiclesResponse {
  vehicles: Vehicle[];
}

// Map Component Props
export interface MapSelectorProps {
  onSelect: (from: Coordinate, to: Coordinate) => void;
  route?: Coordinate[] | null;
  ecoRouteData?: EcoRouteResponse | null;
  showRouteType?: 'eco' | 'google' | 'both';
  fromCoord?: Coordinate | null;
  toCoord?: Coordinate | null;
}

// API Route Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

export type RouteType = 'eco' | 'google' | 'both';