import logging
import time
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core import calculate_route as core_calculate_route
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

# Configuration - These will be loaded into core/utils as needed
CSV_PATH = "export-hbefa.csv" # This file should be present in the root of your application
# API_KEY_GOOGLE = "AIzaSyBn6r80g6qTuIgaKnJpK4-oWAkWls5YtL0" # Handled by env vars in main
# API_KEY_TOMTOM="WzqAKEeo0JHT9Z3SU05ZkgL8rqFbzHGN" # Handled by env vars in main


@app.on_event("startup")
async def startup_event():
    """Initialize data and check API keys on startup."""
    # Set API keys from environment variables for other modules to access
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
        load_co2_data() # Ensure it's loaded if not already
    return {
        "vehicles": [
            {"id": vehicle_id, "name": vehicle_id.replace('.', ' ').title()}
            for vehicle_id in co2_map.keys()
        ]
    }


@app.post("/route", response_model=RouteResponse)
async def calculate_route(request: RouteRequest):
    """
    Calculate eco-friendly and fastest routes between origin and destination for a given vehicle type.
    """
    start_time = time.time()
    logger.info(f"Received route calculation request for vehicle '{request.vehicle}' from {request.origin} to {request.destination}")

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


if __name__ == "__main__":
    import uvicorn
    # For local development, you can set API keys directly or via environment variables
    # These will be picked up by os.getenv in startup_event
    os.environ["API_KEY_GOOGLE"] = "AIzaSyBn6r80g6qTuIgaKnJpK4-oWAkWls5YtL0"
    os.environ["API_KEY_TOMTOM"] = "WzqAKEeo0JHT9Z3SU05ZkgL8rqFbzHGN"
    uvicorn.run(app, host="0.0.0.0", port=8000)