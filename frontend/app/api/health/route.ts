import { NextResponse } from 'next/server';

const BACKEND_URL = process.env.ECOROUTE_BACKEND_URL || 'http://localhost:8000';

export async function GET() {
  try {
    const response = await fetch(`${BACKEND_URL}/`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      return NextResponse.json(
        { status: 'Backend unavailable', backend_url: BACKEND_URL },
        { status: 503 }
      );
    }

    const data = await response.json();
    return NextResponse.json({
      status: 'OK',
      backend_status: data.message,
      backend_url: BACKEND_URL
    });

  } catch (error) {
    console.error('Health check error:', error);
    return NextResponse.json(
      { status: 'Backend connection failed', backend_url: BACKEND_URL },
      { status: 503 }
    );
  }
}