// app/api/vehicles/route.ts
import { NextResponse } from 'next/server';
import { VehiclesResponse } from '@/types/route';

const BACKEND_URL = process.env.ECOROUTE_BACKEND_URL || 'http://localhost:8000';

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/vehicles`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      // Fallback to default vehicles if backend is unavailable
      const fallbackVehicles: VehiclesResponse = {
        vehicles: [
          { id: "pass. car", name: "Passenger Car" },
          { id: "motorcycle", name: "Motorcycle" },
          { id: "LCV", name: "Light Commercial Vehicle" },
          { id: "HGV", name: "Heavy Goods Vehicle" },
          { id: "coach", name: "Coach" },
          { id: "urban bus", name: "Urban Bus" }
        ]
      };
      return NextResponse.json(fallbackVehicles);
    }

    const data: VehiclesResponse = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('Vehicles API error:', error);
    
    // Return fallback vehicles on error
    const fallbackVehicles: VehiclesResponse = {
      vehicles: [
        { id: "pass. car", name: "Passenger Car" },
        { id: "motorcycle", name: "Motorcycle" },
        { id: "LCV", name: "Light Commercial Vehicle" },
        { id: "HGV", name: "Heavy Goods Vehicle" },
        { id: "coach", name: "Coach" },
        { id: "urban bus", name: "Urban Bus" }
      ]
    };
    return NextResponse.json(fallbackVehicles);
  }
}