import os
import json
import logging
import time
import math
import random
import pickle
import hashlib
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import requests
import networkx as nx
from pathlib import Path
from geopy.distance import geodesic

logger = logging.getLogger(__name__)

# Global variables for caching and data
# The 'global' keyword here is for clarity, but the assignment below initializes them
co2_map = {}
elev_cache = {}
tomtom_cache = {}

# Configuration paths
CURRENT_DIR = Path(__file__).resolve().parent
CSV_PATH = CURRENT_DIR.parent / "export-hbefa.csv" # CORRECTED PATH
ELEVATION_CACHE_PATH = CURRENT_DIR.parent / "elevation_cache.json"
TOMTOM_CACHE_PATH = CURRENT_DIR.parent / "tomtom_cache.json"

# Global cache directory
CACHE_DIR = Path("graph_cache")
CACHE_DIR.mkdir(exist_ok=True)


def load_co2_data():
    """Load CO2 emission factors from CSV"""
    global co2_map
    if not co2_map: # This check is still useful to prevent re-loading if already populated
        try:
            df = pd.read_csv(CSV_PATH, header=1)
            logger.info(f"DF after reading CSV (shape): {df.shape}")
            logger.info(f"DF columns: {df.columns.tolist()}")

            df = df[df["Pollutant"].str.contains("CO2")]
            logger.info(f"DF after CO2 filter (shape): {df.shape}")

            df["Emission factor"] = pd.to_numeric(
                df["Emission factor"].astype(str).str.replace(",", ""), errors="coerce"
            )
            logger.info(f"DF after numeric conversion (NaNs): {df['Emission factor'].isnull().sum()}")

            df.dropna(subset=["Emission factor"], inplace=True)
            logger.info(f"DF after dropping NaNs (shape): {df.shape}")

            # --- CRITICAL CHANGE HERE: Clear and update, instead of reassigning ---
            co2_map.clear() # Clear existing content (the initial empty dict)
            co2_map.update(df.groupby("Vehicle category")["Emission factor"].mean().to_dict())
            # ----------------------------------------------------------------------
            logger.info(f"Loaded CO2 data for {len(co2_map)} vehicle types.")
        except FileNotFoundError:
            logger.warning(f"CO2 data CSV not found at {CSV_PATH}. Using default emission factors.")
            co2_map.clear()
            co2_map.update({
                "motorcycle": 150000,
                "pass. car": 180000,
                "LCV": 220000,
                "coach": 800000,
                "HGV": 900000,
                "urban bus": 1200000
            })
        except Exception as e:
            logger.error(f"Error loading CO2 data from CSV: {e}. Using default emission factors.")
            co2_map.clear()
            co2_map.update({
                "motorcycle": 150000,
                "pass. car": 180000,
                "LCV": 250000,
                "coach": 800000,
                "HGV": 900000,
                "urban bus": 1200000
            })
    return co2_map

def vehicle_mass_kg(vehicle: str) -> int:
    """Return vehicle mass in kg"""
    return {
        "motorcycle": 200,
        "pass. car": 1500,
        "LCV": 2500,
        "coach": 11000,
        "HGV": 18000,
        "urban bus": 14000
    }.get(vehicle, 1500) # Default to passenger car mass

def load_elevation_cache():
    """Load elevation cache from file"""
    global elev_cache
    if os.path.exists(ELEVATION_CACHE_PATH):
        try:
            with open(ELEVATION_CACHE_PATH, "r") as f:
                # --- APPLY SAME LOGIC HERE ---
                elev_cache.clear()
                elev_cache.update({int(k): v for k, v in json.load(f).items()})
            logger.info(f"Loaded elevation cache with {len(elev_cache)} entries.")
        except json.JSONDecodeError as e:
            logger.warning(f"Error decoding elevation cache JSON: {e}. Starting with empty cache.")
            elev_cache.clear() # Ensure it's explicitly empty on error
        except Exception as e:
            logger.error(f"Unexpected error loading elevation cache: {e}. Starting with empty cache.")
            elev_cache.clear()
    return elev_cache

def save_elevation_cache():
    """Save elevation cache to file"""
    try:
        with open(ELEVATION_CACHE_PATH, "w") as f:
            json.dump(elev_cache, f)
        logger.debug(f"Elevation cache saved with {len(elev_cache)} entries.")
    except IOError as e:
        logger.error(f"Failed to save elevation cache: {e}")

def load_tomtom_cache():
    """Load TomTom cache from file"""
    global tomtom_cache
    if os.path.exists(TOMTOM_CACHE_PATH):
        try:
            with open(TOMTOM_CACHE_PATH, "r") as f:
                # --- APPLY SAME LOGIC HERE ---
                tomtom_cache.clear()
                tomtom_cache.update(json.load(f))
            logger.info(f"Loaded TomTom cache with {len(tomtom_cache)} entries.")
        except json.JSONDecodeError as e:
            logger.warning(f"Error decoding TomTom cache JSON: {e}. Starting with empty cache.")
            tomtom_cache.clear() # Ensure it's explicitly empty on error
        except Exception as e:
            logger.error(f"Unexpected error loading TomTom cache: {e}. Starting with empty cache.")
            tomtom_cache.clear()
    return tomtom_cache

def save_tomtom_cache():
    """Save TomTom cache to file"""
    try:
        with open(TOMTOM_CACHE_PATH, "w") as f:
            json.dump(tomtom_cache, f)
        logger.debug(f"TomTom cache saved with {len(tomtom_cache)} entries.")
    except IOError as e:
        logger.error(f"Failed to save TomTom cache: {e}")

def fetch_elevation_batch(coords: List[Tuple[float, float]]) -> List[float]:
    """Fetch elevation data for a batch of coordinates from Open Topo Data API."""
    loc_str = "|".join(f"{lat},{lon}" for lat, lon in coords)
    url = f"https://api.opentopodata.org/v1/eudem25m?locations={loc_str}"
    try:
        r = requests.get(url, timeout=10) # Added timeout
        r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        results = r.json().get("results", [])
        return [res.get("elevation", 0.0) for res in results]
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching elevation batch from {url}: {e}")
        return [0.0] * len(coords)
    except Exception as e:
        logger.error(f"Unexpected error in fetch_elevation_batch: {e}")
        return [0.0] * len(coords)

def fetch_speed(item: Tuple[str, Tuple[float, float]], retries: int = 3) -> Tuple[str, Optional[float]]:
    """Fetch speed data from TomTom API for a single point."""
    key, (lat, lon) = item
    api_key_tomtom = os.getenv("API_KEY_TOMTOM")
    if not api_key_tomtom:
        logger.warning("TomTom API key not set. Cannot fetch real-time speeds.")
        return key, None

    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&key={api_key_tomtom}"
    for attempt in range(retries):
        try:
            res = requests.get(url, timeout=5)
            res.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            speed = res.json()["flowSegmentData"]["currentSpeed"]
            return key, float(speed)
        except requests.exceptions.RequestException as e:
            logger.warning(f"TomTom API request failed (attempt {attempt + 1}/{retries}) for {key}: {e}")
            time.sleep(0.5 + random.uniform(0, 0.3)) # Exponential backoff with jitter
        except Exception as e:
            logger.error(f"Unexpected error fetching TomTom speed for {key}: {e}")
            break # No point retrying for unexpected errors
    return key, None

def calculate_bearing(p1: Dict[str, float], p2: Dict[str, float]) -> float:
    """Calculate bearing in degrees between two points (nodes) from OSMnx graph."""
    lat1, lon1 = math.radians(p1["y"]), math.radians(p1["x"])
    lat2, lon2 = math.radians(p2["y"]), math.radians(p2["x"])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def sample_waypoints(coords: List[Tuple[float, float]], max_waypoints: int = 23) -> List[Tuple[float, float]]:
    """
    Sample intermediate waypoints from a list of coordinates for Google Directions API.
    Google Directions API allows up to 23 waypoints in addition to origin and destination.
    """
    if len(coords) <= 2: # Origin and destination only, no intermediate points
        return []
    
    # Exclude origin and destination from sampling
    intermediate_coords = coords[1:-1]
    
    if len(intermediate_coords) <= max_waypoints:
        return intermediate_coords
    
    # Calculate step size to evenly sample
    step = math.ceil(len(intermediate_coords) / max_waypoints)
    sampled = []
    for i in range(0, len(intermediate_coords), step):
        sampled.append(intermediate_coords[i])
        if len(sampled) >= max_waypoints:
            break
            
    return sampled

def safe_get(d: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Safely get a float value from a dictionary, returning a default if not found or not numeric."""
    val = d.get(key)
    if isinstance(val, (int, float)):
        return float(val)
    return default

def check_api_keys():
    """Check if necessary API keys are set."""
    if not os.getenv("API_KEY_GOOGLE"):
        logger.error("API_KEY_GOOGLE environment variable not set.")
        raise ValueError("Google Maps API Key is required but not set.")
    if not os.getenv("API_KEY_TOMTOM"):
        logger.warning("API_KEY_TOMTOM environment variable not set. Real-time traffic data will not be used.")

def interpolate_points_along_route(origin_coords, destination_coords, num_points):
    """
    Create intermediate points along the great circle route between origin and destination.
    
    Args:
        origin_coords: (lat, lon) tuple for origin
        destination_coords: (lat, lon) tuple for destination
        num_points: Number of intermediate points to create
    
    Returns:
        list: List of (lat, lon) tuples including origin and destination
    """
    points = []
    
    # Add origin
    points.append(origin_coords)
    
    # Calculate intermediate points
    for i in range(1, num_points - 1):
        fraction = i / (num_points - 1)
        
        # Linear interpolation of coordinates
        lat = origin_coords[0] + fraction * (destination_coords[0] - origin_coords[0])
        lon = origin_coords[1] + fraction * (destination_coords[1] - origin_coords[1])
        
        points.append((lat, lon))
    
    # Add destination
    points.append(destination_coords)
    
    return points

def get_chunk_cache_key(center_coords, chunk_size_km):
    """
    Generate a unique cache key for a graph chunk.
    
    Args:
        center_coords: (lat, lon) tuple for center
        chunk_size_km: Size of chunk in kilometers
    
    Returns:
        str: Cache key
    """
    # Round coordinates to 4 decimal places for consistent caching
    lat_rounded = round(center_coords[0], 4)
    lon_rounded = round(center_coords[1], 4)
    
    # Create cache key from rounded coordinates and chunk size
    cache_string = f"chunk_{lat_rounded}_{lon_rounded}_{chunk_size_km}"
    
    # Use hash to create shorter filename
    cache_hash = hashlib.md5(cache_string.encode()).hexdigest()[:12]
    
    return f"chunk_{cache_hash}.pkl"

def save_chunk_to_cache(graph_chunk, cache_key):
    """
    Save a graph chunk to cache.
    
    Args:
        graph_chunk: networkx.MultiDiGraph object
        cache_key: Cache key string
    """
    try:
        cache_path = CACHE_DIR / cache_key
        with open(cache_path, 'wb') as f:
            pickle.dump(graph_chunk, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f"Saved chunk to cache: {cache_key}")
    except Exception as e:
        logger.warning(f"Failed to save chunk to cache: {e}")

def load_chunk_from_cache(cache_key):
    """
    Load a graph chunk from cache.
    
    Args:
        cache_key: Cache key string
    
    Returns:
        networkx.MultiDiGraph or None: Cached graph chunk
    """
    try:
        cache_path = CACHE_DIR / cache_key
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                chunk = pickle.load(f)
            logger.debug(f"Loaded chunk from cache: {cache_key}")
            return chunk
    except Exception as e:
        logger.warning(f"Failed to load chunk from cache: {e}")
    
    return None

def clear_old_cache(max_age_days=7):
    """
    Clear cache files older than specified days.
    
    Args:
        max_age_days: Maximum age in days
    """
    try:
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        for cache_file in CACHE_DIR.glob("chunk_*.pkl"):
            if current_time - cache_file.stat().st_mtime > max_age_seconds:
                cache_file.unlink()
                logger.debug(f"Deleted old cache file: {cache_file.name}")
    except Exception as e:
        logger.warning(f"Failed to clear old cache: {e}")

def calculate_chunk_bounds(center_coords, chunk_size_km=25):
    """
    Calculate bounding box for a chunk around center coordinates.
    
    Args:
        center_coords: (lat, lon) tuple for center
        chunk_size_km: Size of chunk in kilometers
    
    Returns:
        tuple: (west, south, east, north) bounds
    """
    center_lat, center_lon = center_coords
    
    # Calculate approximate degree offset for the chunk size
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 km * cos(latitude)
    lat_offset = chunk_size_km / 111.0
    lon_offset = chunk_size_km / (111.0 * math.cos(math.radians(center_lat)))
    
    north = center_lat + lat_offset
    south = center_lat - lat_offset
    east = center_lon + lon_offset
    west = center_lon - lon_offset
    
    return west, south, east, north