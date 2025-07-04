import { NextRequest, NextResponse } from 'next/server';
import { EcoRouteRequest, EcoRouteResponse } from '@/types/route';

const BACKEND_URL = process.env.ECOROUTE_BACKEND_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const body: EcoRouteRequest = await request.json();
    
    // Validate request body
    if (!body.origin || !body.destination || !body.vehicle) {
      return NextResponse.json(
        { error: 'Missing required fields: origin, destination, vehicle' },
        { status: 400 }
      );
    }

    // Validate coordinates
    const [originLat, originLng] = body.origin;
    const [destLat, destLng] = body.destination;
    
    if (!isValidCoordinate(originLat, originLng) || !isValidCoordinate(destLat, destLng)) {
      return NextResponse.json(
        { error: 'Invalid coordinates provided' },
        { status: 400 }
      );
    }

    // Call FastAPI backend
    const response = await fetch(`${BACKEND_URL}/route`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorData = await response.text();
      console.error('Backend error:', errorData);
      return NextResponse.json(
        { error: 'Failed to calculate route' },
        { status: response.status }
      );
    }

    const data: EcoRouteResponse = await response.json();
    return NextResponse.json(data);

  } catch (error) {
    console.error('EcoRoute API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

function isValidCoordinate(lat: number, lng: number): boolean {
  return (
    typeof lat === 'number' &&
    typeof lng === 'number' &&
    lat >= -90 && lat <= 90 &&
    lng >= -180 && lng <= 180
  );
}