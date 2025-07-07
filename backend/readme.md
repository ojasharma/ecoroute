# EcoRoute API

A FastAPI application that calculates eco-friendly driving routes by considering CO2 emissions, elevation changes, and real-time traffic data.

## Features

- Calculates both eco-friendly and fastest routes between two points
- Considers vehicle type, elevation changes, and real-time traffic
- Compares results with Google Maps routes
- Caches elevation and traffic data for better performance

## Installation

### Prerequisites

- Python 3.8+
- Google Maps API key
- TomTom API key

### Steps

1. Clone the repository:
```bash
git clone git@github.com:Team-Semicolon-27/eco-route.git
cd ecoroute-api
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file for your API keys:
```bash
echo "GOOGLE_MAPS_API_KEY=your_google_maps_key" > .env
echo "TOMTOM_API_KEY=your_tomtom_key" >> .env
```

## Running the Application

### Development
```bash
uvicorn main:app --reload
```

### Production
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Documentation

After starting the server, access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

- `POST /route` - Calculate eco-friendly and fastest routes
- `GET /vehicles` - List available vehicle types

## Configuration

You can configure the application by editing these environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_MAPS_API_KEY` | Google Maps API key | Required |
| `TOMTOM_API_KEY` | TomTom Traffic API key | Required |
| `CSV_PATH` | Path to CO2 emissions data CSV | "export-hbefa.csv" |

## Cache Files

The application creates and uses these cache files:
- `elevation_cache.json` - Cached elevation data
- `tomtom_cache.json` - Cached traffic data

## License

[MIT License](LICENSE)