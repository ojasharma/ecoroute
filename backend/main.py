import logging
import time
import os
import io # Import the io module

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.core import calculate_route as core_calculate_route
from app.core import calculate_route_streamed
from app.utils import check_api_keys, load_co2_data, load_elevation_cache, load_tomtom_cache, co2_map
from app.models import RouteRequest, RouteResponse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="EcoRoute API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize data and check API keys on startup."""
    os.environ["API_KEY_GOOGLE"] = os.getenv("API_KEY_GOOGLE", "AIzaSyBn6r80g6qTuIgaKnJpK4-oWAkWls5YtL0")
    os.environ["API_KEY_TOMTOM"] = os.getenv("API_KEY_TOMTOM", "WzqAKEeo0JHT9Z3SU05ZkgL8rqFbzHGN")

    check_api_keys()
    load_co2_data()
    load_elevation_cache()
    load_tomtom_cache()
    logger.info("Application startup complete.")


@app.get("/")
async def root():
    return {"message": "EcoRoute API is running. Visit /docs for API documentation."}


@app.get("/vehicles")
async def get_vehicles():
    """Get available vehicle types for route calculation."""
    if not co2_map:
        load_co2_data()
    return {
        "vehicles": [
            {"id": vehicle_id, "name": vehicle_id.replace('.', ' ').title()}
            for vehicle_id in co2_map.keys()
        ]
    }


def log_performance(func_name: str, start_time: float, **kwargs):
    """Log performance metrics and additional info"""
    duration = time.time() - start_time
    logger.info(f"{func_name} completed in {duration:.2f}s - {kwargs}")

def log_graph_stats(G, stage: str):
    """Log graph statistics for debugging"""
    logger.info(f"Graph stats at {stage}: {len(G.nodes)} nodes, {len(G.edges)} edges")
    
    # Sample some edge attributes for debugging
    sample_edges = list(G.edges(data=True))[:3]
    for u, v, data in sample_edges:
        logger.debug(f"Sample edge {u}->{v}: {list(data.keys())}")

@app.post("/route", response_model=RouteResponse)
async def calculate_route(request: RouteRequest):
    """
    Calculate eco-friendly and fastest routes between origin and destination for a given vehicle type.
    """
    start_time = time.time()
    logger.info(f"Received route calculation request for vehicle '{request.vehicle}' from {request.origin} to {request.destination}")

    # --- Start Log Capture ---
    log_stream = io.StringIO()
    core_logger = logging.getLogger("app.core")
    original_level = core_logger.level
    core_logger.setLevel(logging.DEBUG) # Capture all logs from DEBUG level up

    handler = logging.StreamHandler(log_stream)
    # Use a simple formatter to get clean messages for the frontend
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    core_logger.addHandler(handler)
    # --- End Log Capture Setup ---

    try:
        response_data = await core_calculate_route(request)
        logger.info(f"Route calculation completed successfully in {time.time() - start_time:.2f}s.")
        return response_data
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Validation error in route calculation: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.critical(f"An unhandled critical error occurred during route calculation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: An unexpected error occurred. Please try again later. ({str(e)})")
    finally:
        # --- Finalize and Add Logs ---
        # This block runs whether the request succeeds or fails
        core_logger.removeHandler(handler)
        core_logger.setLevel(original_level) # Restore original logger level
        
        captured_logs = log_stream.getvalue().splitlines()
        log_stream.close()
        
        # If response_data was created, add logs to it
        if 'response_data' in locals() and response_data:
            response_data.logs = captured_logs
        # --- End Finalize ---

@app.get("/route/stream")
async def stream_route(origin: str, destination: str, vehicle: str):
    """
    Calculates routes and streams logs and results back to the client using SSE.
    
    Query Params:
    - origin: "lat,lon"
    - destination: "lat,lon"
    - vehicle: "pass. car"
    """
    try:
        # Parse and validate coordinates from query strings
        origin_coords = tuple(map(float, origin.split(',')))
        destination_coords = tuple(map(float, destination.split(',')))
        
        if not all(-90 <= lat <= 90 and -180 <= lon <= 180 for lat, lon in [origin_coords, destination_coords]):
            raise ValueError("Invalid coordinates.")

        request = RouteRequest(
            origin=origin_coords,
            destination=destination_coords,
            vehicle=vehicle
        )
        
        # Return a streaming response that calls our async generator
        return StreamingResponse(calculate_route_streamed(request), media_type="text/event-stream")

    except ValueError as e:
        # This error is for invalid query parameter format
        raise HTTPException(status_code=400, detail=f"Invalid request parameters: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in route calculation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    # For local development, you can set API keys directly or via environment variables
    # These will be picked up by os.getenv in startup_event
    os.environ["API_KEY_GOOGLE"] = "AIzaSyBn6r80g6qTuIgaKnJpK4-oWAkWls5YtL0"
    os.environ["API_KEY_TOMTOM"] = "WzqAKEeo0JHT9Z3SU05ZkgL8rqFbzHGN"
    uvicorn.run(app, host="0.0.0.0", port=8000)