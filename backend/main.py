import logging
import time
from typing import Dict, Any, List, Tuple, Optional
import os
import json
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import networkx as nx
import osmnx as ox
import googlemaps
import requests
import polyline

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# Configuration - Use environment variables for API keys
CSV_PATH = "export-hbefa.csv" # This file should be present in the root of your application
API_KEY_GOOGLE = "AIzaSyBn6r80g6qTuIgaKnJpK4-oWAkWls5YtL0"
API_KEY_TOMTOM="WzqAKEeo0JHT9Z3SU05ZkgL8rqFbzHGN"
# Global variables for caching
co2_map = {}
elev_cache = {}
tomtom_cache = {}

# Paths for cache files
ELEVATION_CACHE_PATH = "elevation_cache.json"
TOMTOM_CACHE_PATH = "tomtom_cache.json"

class RouteRequest(BaseModel):
    origin: Tuple[float, float]  # (lat, lon)
    destination: Tuple[float, float]  # (lat, lon)
    vehicle: str  # "motorcycle", "pass. car", "LCV", "coach", "HGV", "urban bus"

class RouteResponse(BaseModel):
    eco_route: List[Tuple[float, float]]  # List of (lat, lon) coordinates
    google_route: List[Tuple[float, float]]  # List of (lat, lon) coordinates
    eco_stats: dict
    google_stats: dict
    comparison: dict

def load_co2_data():
    """Load CO2 emission factors from CSV"""
    global co2_map
    if not co2_map:
        try:
            df = pd.read_csv(CSV_PATH, header=1)
            df = df[df["Pollutant"].str.contains("CO2")]
            # Ensure "Emission factor" is numeric, handling comma and NaN values
            df["Emission factor"] = pd.to_numeric(
                df["Emission factor"].astype(str).str.replace(",", ""), errors="coerce"
            )
            # Drop rows where 'Emission factor' became NaN after coercion
            df.dropna(subset=["Emission factor"], inplace=True)
            co2_map = df.groupby("Vehicle category")["Emission factor"].mean().to_dict()
            logger.info(f"Loaded CO2 data for {len(co2_map)} vehicle types.")
        except FileNotFoundError:
            logger.warning(f"CO2 data CSV not found at {CSV_PATH}. Using default emission factors.")
            # Default CO2 values if CSV not found or parsing fails
            co2_map = {
                "motorcycle": 150000, # g/km
                "pass. car": 180000,
                "LCV": 220000,
                "coach": 800000,
                "HGV": 900000,
                "urban bus": 1200000
            }
        except Exception as e:
            logger.error(f"Error loading CO2 data from CSV: {e}. Using default emission factors.")
            co2_map = {
                "motorcycle": 150000,
                "pass. car": 180000,
                "LCV": 250000,
                "coach": 800000,
                "HGV": 900000,
                "urban bus": 1200000
            }
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
                # Keys are stored as strings in JSON, convert back to int for node IDs
                elev_cache = {int(k): v for k, v in json.load(f).items()}
            logger.info(f"Loaded elevation cache with {len(elev_cache)} entries.")
        except json.JSONDecodeError as e:
            logger.warning(f"Error decoding elevation cache JSON: {e}. Starting with empty cache.")
            elev_cache = {}
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
                tomtom_cache = json.load(f)
            logger.info(f"Loaded TomTom cache with {len(tomtom_cache)} entries.")
        except json.JSONDecodeError as e:
            logger.warning(f"Error decoding TomTom cache JSON: {e}. Starting with empty cache.")
            tomtom_cache = {}
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
    if not API_KEY_TOMTOM:
        logger.warning("TomTom API key not set. Cannot fetch real-time speeds.")
        return key, None

    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&key={API_KEY_TOMTOM}"
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

def base_eco_cost(u: int, v: int, data: Dict[str, Any], vehicle: str, G: nx.MultiDiGraph, co2_map: Dict[str, float]) -> float:
    """Calculate ecological cost for an edge."""
    length = data.get("length", 1.0)  # meters, default to 1m to avoid division by zero
    
    # Elevation gain calculation
    u_elev = G.nodes[u].get("elevation", 0.0)
    v_elev = G.nodes[v].get("elevation", 0.0)
    elev_gain = max(0.0, v_elev - u_elev) # Only positive gain contributes to energy cost
    
    mass = vehicle_mass_kg(vehicle)  # kg
    
    # Base CO2 emission rate (e.g., from HBEFA data), converted from g/km to kg/m
    # co2_map values are typically g/km, so divide by 1,000,000 to get kg/m
    base_rate = co2_map.get(vehicle, 180000) / 1_000_000 # Default to passenger car rate if vehicle not found
    base_emissions = length * base_rate  # kg CO2

    # Approximate slope-induced emissions (energy for climbing)
    # This factor (0.074 / 1e6) seems to be an empirical constant.
    # Mass * gravity * height = potential energy (Joules). Convert Joules to kg CO2.
    slope_emissions = mass * 9.81 * elev_gain * (0.074 / 1e6)  # kg CO2

    # Turn penalty contribution to eco cost
    # The turn_penalty is initially added to travel_time in seconds.
    # Convert it to an equivalent CO2 cost. Assuming 1 second of delay costs X kg CO2.
    # This factor (0.1) is empirical. Adjust as needed.
    turn_penalty_cost = data.get("turn_penalty", 0.0) * 0.1  # kg CO2

    return base_emissions + slope_emissions + turn_penalty_cost

def compute_stats(route: List[int], vehicle: str, G: nx.MultiDiGraph, edge_mapping: Dict[Tuple[int, int], Tuple[int, int, int]], co2_map: Dict[str, float]) -> Tuple[float, float, float]:
    """Compute route statistics (distance, time, CO2) for a given path."""
    total_dist = 0.0 # meters
    total_time = 0.0 # seconds
    total_co2 = 0.0  # kg

    if not route:
        return 0.0, 0.0, 0.0

    for i in range(len(route) - 1):
        u, v = route[i], route[i+1]
        
        # In case of the simplified graph, we need to map back to the original multigraph edge
        # The edge_mapping stores (u, v) -> (original_u, original_v, original_key)
        # If the direct (u,v) is not in mapping, it means it's a simple graph edge already.
        original_u, original_v, original_k = edge_mapping.get((u, v), (u, v, 0)) # Default to key 0 if not mapped

        # Safely get edge data, especially for eco_cost which might be on the original edge
        try:
            # For simplicity, if G_simple uses original u,v, then original_k might be 0, assuming
            # the simplified graph only has one edge between u,v.
            # If the G_simple approach is to pick one k, then G_simple[u][v] will have that chosen k's data directly
            # but for computing stats accurately from original multigraph 'G', we need the right key.
            # However, `G_simple.add_edge(u, v, **data)` copies all attributes, so we can directly use G_simple[u][v].
            # Let's adjust this to use the attributes that were copied to G_simple directly,
            # or if this function is meant for the original graph, pass the original graph's edge data.
            # Given that `compute_stats` takes `G` and `edge_mapping`, it seems designed to work with the *original*
            # graph `G` to look up the correct multigraph edge data.

            # Retrieve the specific edge data from the original graph G
            data = G[original_u][original_v][original_k]
            
            total_dist += data.get("length", 0.0)
            total_time += data.get("travel_time", 0.0)
            total_co2 += data.get("eco_cost", 0.0)
        except KeyError:
            logger.warning(f"Edge ({original_u}, {original_v}, {original_k}) not found in original graph G. Skipping stats for this segment.")
            continue # Skip this segment if edge data is missing

    return total_dist / 1000, total_time / 60, total_co2 / 1000  # km, minutes, kg

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
    if not API_KEY_GOOGLE:
        logger.error("Maps_API_KEY environment variable not set.")
        raise ValueError("Google Maps API Key is required but not set.")
    if not API_KEY_TOMTOM:
        logger.warning("TOMTOM_API_KEY environment variable not set. Real-time traffic data will not be used.")

@app.on_event("startup")
async def startup_event():
    """Initialize data and check API keys on startup."""
    check_api_keys()
    load_co2_data()
    load_elevation_cache()
    load_tomtom_cache()

@app.get("/")
async def root():
    return {"message": "EcoRoute API is running. Visit /docs for API documentation."}

@app.get("/vehicles")
async def get_vehicles():
    """Get available vehicle types for route calculation."""
    # Ensure CO2 data is loaded before returning vehicle types
    load_co2_data() 
    return {
        "vehicles": [
            {"id": vehicle_id, "name": vehicle_id.replace('.', ' ').title()}
            for vehicle_id in co2_map.keys()
        ]
    }

## Route Calculation Endpoint
@app.post("/route", response_model=RouteResponse)
async def calculate_route(request: RouteRequest):
    """
    Calculate eco-friendly and fastest routes between origin and destination for a given vehicle type.
    """
    start_time = time.time()
    logger.info(f"Starting route calculation for vehicle '{request.vehicle}' from {request.origin} to {request.destination}")
    
    try:
        origin_coords = request.origin
        destination_coords = request.destination
        vehicle = request.vehicle

        # Check if the requested vehicle type is in our CO2 map
        if vehicle not in co2_map:
            logger.error(f"Invalid vehicle type: {vehicle}. Available types: {list(co2_map.keys())}")
            raise HTTPException(status_code=400, detail=f"Invalid vehicle type '{vehicle}'. Please choose from: {', '.join(co2_map.keys())}")

        # --- Graph Building ---
        logger.info("Building road network graph...")
        graph_build_start = time.time()
        try:
            # Retrieve graph for a sufficient area around the origin
            G = ox.graph_from_point(origin_coords, dist=25000, network_type="drive", dist_type='bbox')
            # Ensure speeds are added; these are OSMnx default speeds
            G = ox.add_edge_speeds(G)
            # Calculate travel times based on speeds and lengths
            G = ox.add_edge_travel_times(G)
            logger.info(f"Graph built with {len(G.nodes)} nodes and {len(G.edges)} edges in {time.time() - graph_build_start:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to build road network graph: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to build road network graph. Please check coordinates or try a different location.")

        # --- Add Elevation Data ---
        logger.info("Adding elevation data to graph nodes...")
        elev_add_start = time.time()
        try:
            global elev_cache
            uncached_nodes = [node for node in G.nodes if node not in elev_cache]
            logger.debug(f"Found {len(uncached_nodes)} nodes needing elevation data.")
            
            if uncached_nodes:
                # Prepare coordinates for batch fetching
                uncached_node_coords = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in uncached_nodes]
                
                # Fetch in batches to comply with API limits and improve efficiency
                batch_size = 100
                batches = [uncached_node_coords[i:i + batch_size] for i in range(0, len(uncached_node_coords), batch_size)]
                
                # Use ThreadPoolExecutor for concurrent fetching
                fetched_elevations = []
                with ThreadPoolExecutor(max_workers=5) as executor:
                    future_to_batch = {executor.submit(fetch_elevation_batch, batch): batch for batch in batches}
                    for i, future in enumerate(as_completed(future_to_batch)):
                        batch_result = future.result()
                        fetched_elevations.extend(batch_result)
                        logger.debug(f"Processed elevation batch {i+1}/{len(batches)}")
                
                # Update graph nodes and cache
                for node, elev in zip(uncached_nodes, fetched_elevations):
                    G.nodes[node]["elevation"] = elev
                    elev_cache[node] = elev
                
                save_elevation_cache()
                logger.info(f"Fetched and cached elevations for {len(fetched_elevations)} nodes.")
            
            # Ensure all nodes have an elevation attribute (default to 0.0 if not set)
            for node in G.nodes:
                if "elevation" not in G.nodes[node] or G.nodes[node]["elevation"] is None:
                    G.nodes[node]["elevation"] = 0.0
            logger.info(f"Elevation data added in {time.time() - elev_add_start:.2f}s.")
        except Exception as e:
            logger.error(f"Error adding elevation data: {e}", exc_info=True)
            # Decide if this should be a critical failure or allow continuation with default elevations
            logger.warning("Continuing without full elevation data. Eco-route calculation may be less accurate.")

        # --- Add Grades (slope information) ---
        logger.info("Calculating and adding edge grades...")
        grades_start = time.time()
        try:
            G = ox.add_edge_grades(G)
            logger.info(f"Edge grades added in {time.time() - grades_start:.2f}s.")
        except Exception as e:
            logger.error(f"Error adding edge grades: {e}", exc_info=True)
            logger.warning("Continuing without edge grade data. Eco-route calculation may be less accurate.")

        # --- Find Nearest Nodes ---
        logger.info("Finding nearest graph nodes for origin and destination...")
        nearest_nodes_start = time.time()
        try:
            orig_node = ox.distance.nearest_nodes(G, X=origin_coords[1], Y=origin_coords[0])
            dest_node = ox.distance.nearest_nodes(G, X=destination_coords[1], Y=destination_coords[0])
            if orig_node is None or dest_node is None:
                raise ValueError("Could not find nearest nodes for origin or destination.")
            logger.info(f"Nearest nodes found: Origin {orig_node}, Destination {dest_node} in {time.time() - nearest_nodes_start:.2f}s.")
        except Exception as e:
            logger.error(f"Failed to find nearest nodes: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail="Could not find reachable road network nodes for the provided origin or destination. Please ensure they are on or near a road.")

        # --- Calculate Fastest Route (initial path for TomTom data scope) ---
        logger.info("Calculating initial fastest route...")
        fastest_route_calc_start = time.time()
        try:
            # Using Dijkstra's algorithm for shortest path based on travel_time
            fastest_route_nodes = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
            # Collect unique edges from the fastest route to consider for TomTom fetching
            fastest_route_edges_uv = set((u, v) for u, v in zip(fastest_route_nodes[:-1], fastest_route_nodes[1:]))
            logger.info(f"Initial fastest route calculated with {len(fastest_route_nodes)} nodes in {time.time() - fastest_route_calc_start:.2f}s.")
        except nx.NetworkXNoPath:
            logger.error("No path found between origin and destination nodes.")
            raise HTTPException(status_code=404, detail="No route found between the specified origin and destination.")
        except Exception as e:
            logger.error(f"Error calculating initial fastest route: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to calculate an initial route.")

        # --- Fetch and Apply TomTom Traffic Data ---
        logger.info("Fetching and applying TomTom traffic data...")
        tomtom_fetch_start = time.time()
        try:
            global tomtom_cache
            coords_to_fetch = {}
            # Map of (u,v,k) -> key for TomTom cache lookup
            edge_tomtom_keys = [] 
            
            # Identify unique coordinate points for edges on the fastest route that need TomTom data
            for u, v in fastest_route_edges_uv:
                for k in G[u][v]: # Iterate over all parallel edges
                    # Use midpoint of edge for TomTom API query
                    lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
                    lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
                    key = f"{lat:.5f},{lon:.5f}" # Use string key for caching
                    if key not in tomtom_cache:
                        coords_to_fetch[key] = (lat, lon)
                    edge_tomtom_keys.append((u, v, k, key))

            logger.info(f"Identified {len(coords_to_fetch)} unique points for TomTom fetching (from {len(edge_tomtom_keys)} edges).")

            if coords_to_fetch:
                fetch_items = list(coords_to_fetch.items())
                # Use ThreadPoolExecutor for concurrent TomTom API calls
                with ThreadPoolExecutor(max_workers=10) as executor: # Increase workers for I/O bound tasks
                    future_to_key = {executor.submit(fetch_speed, item): item[0] for item in fetch_items}
                    for i, future in enumerate(as_completed(future_to_key)):
                        key, speed = future.result()
                        tomtom_cache[key] = speed # Store even if speed is None (failed fetch)
                        if i % 50 == 0 and i > 0: # Log progress
                            logger.debug(f"Fetched {i}/{len(fetch_items)} TomTom speeds.")
                save_tomtom_cache() # Save cache after fetching all batches
                logger.info(f"Completed TomTom fetching for {len(coords_to_fetch)} points.")

            # Apply TomTom speeds to graph edges
            speeds_applied_count = 0
            for u, v, k, key in edge_tomtom_keys:
                speed_kph = tomtom_cache.get(key)
                if speed_kph is not None and speed_kph > 0:
                    G[u][v][k]["speed_kph"] = speed_kph
                    # Recalculate travel time using real-time speed
                    length_m = safe_get(G[u][v][k], "length", 1.0)
                    G[u][v][k]["travel_time"] = length_m / (speed_kph * 1000 / 3600) # seconds
                    speeds_applied_count += 1
                else:
                    # Fallback to OSMnx calculated speed if TomTom data is unavailable or invalid
                    current_speed = G[u][v][k].get("speed_kph", 30.0) # Default to 30 kph if no speed in OSMnx
                    length_m = safe_get(G[u][v][k], "length", 1.0)
                    G[u][v][k]["travel_time"] = length_m / (current_speed * 1000 / 3600)
            
            logger.info(f"Applied TomTom speeds (or fallbacks) to {speeds_applied_count}/{len(edge_tomtom_keys)} edges in {time.time() - tomtom_fetch_start:.2f}s.")
            
        except Exception as e:
            logger.error(f"Error fetching or applying TomTom data: {e}", exc_info=True)
            logger.warning("Continuing route calculation without real-time TomTom traffic data.")

        # --- Calculate Turn Penalties and Add to Travel Time ---
        logger.info("Calculating turn penalties and applying to travel time...")
        penalty_calc_start = time.time()
        try:
            # Re-calculate fastest route using potentially updated travel times (with TomTom data)
            fastest_route_nodes = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
            
            # Initialize turn_penalty attribute for all edges
            nx.set_edge_attributes(G, 0, "turn_penalty")

            penalties_added_count = 0
            for i in range(1, len(fastest_route_nodes) - 1):
                u, v, w = fastest_route_nodes[i - 1], fastest_route_nodes[i], fastest_route_nodes[i + 1]
                
                # Calculate bearing between segments
                b1 = calculate_bearing(G.nodes[u], G.nodes[v])
                b2 = calculate_bearing(G.nodes[v], G.nodes[w])
                
                delta = abs(b2 - b1)
                if delta > 180:
                    delta = 360 - delta # Normalize angle difference

                penalty_seconds = 0
                if delta > 170: # Near U-turn
                    penalty_seconds = 90
                elif delta > 150: # Sharp turn
                    penalty_seconds = 60
                elif delta > 120:
                    penalty_seconds = 45
                elif delta > 90:
                    penalty_seconds = 30
                elif delta > 45: # Moderate turn
                    penalty_seconds = 15
                else: # Slight turn or straight
                    penalty_seconds = 5 # Minimum penalty for any change in direction

                # Apply penalty to the edge *leading into* the turn (u -> v)
                # This affects travel_time which is used for the fastest route calculation
                # and indirectly affects eco_cost if it relies on updated travel_time.
                # Since eco_cost is based on physical properties, adding penalty to eco_cost directly
                # (as done in `base_eco_cost`) makes more sense for turns than to physical travel time,
                # but adding to travel time here influences *path selection*.
                for k in G[u][v]: # Apply to all parallel edges between u and v
                    G[u][v][k]["turn_penalty"] = safe_get(G[u][v][k], "turn_penalty") + penalty_seconds
                    G[u][v][k]["travel_time"] = safe_get(G[u][v][k], "travel_time") + penalty_seconds
                    penalties_added_count += 1
            
            logger.info(f"Added {penalties_added_count} turn penalties to graph edges in {time.time() - penalty_calc_start:.2f}s.")
        except Exception as e:
            logger.error(f"Error calculating turn penalties: {e}", exc_info=True)
            logger.warning("Continuing without turn penalties. Fastest route may not accurately reflect turning costs.")

        # --- Calculate Eco Costs for all Edges ---
        logger.info("Calculating ecological costs for all graph edges...")
        eco_cost_calc_start = time.time()
        try:
            eco_costs_data = {}
            for u, v, k, d in G.edges(keys=True, data=True):
                # 'd' contains all edge attributes
                cost = base_eco_cost(u, v, d, vehicle, G, co2_map)
                eco_costs_data[(u, v, k)] = cost
            
            nx.set_edge_attributes(G, eco_costs_data, "eco_cost")
            logger.info(f"Calculated eco costs for {len(eco_costs_data)} edges in {time.time() - eco_cost_calc_start:.2f}s.")
        except Exception as e:
            logger.error(f"Error calculating eco costs: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to calculate ecological costs for road network.")

        # --- Create Simplified Graph for Eco Route Calculation ---
        logger.info("Creating a simplified graph for eco-route calculation...")
        simplify_graph_start = time.time()
        try:
            G_simple = nx.DiGraph() # Use a simple directed graph
            edge_mapping = {} # Store mapping from G_simple edge to original G multiedge (u,v,k)

            for u, v in G.edges():
                # Find the parallel edge with the minimum eco_cost
                min_k = None
                min_eco_cost = float('inf')
                for k in G[u][v]:
                    current_eco_cost = G[u][v][k].get("eco_cost", float('inf'))
                    if current_eco_cost < min_eco_cost:
                        min_eco_cost = current_eco_cost
                        min_k = k
                
                if min_k is not None:
                    # Copy all attributes of the chosen multi-edge to the simple graph edge
                    G_simple.add_edge(u, v, **G[u][v][min_k])
                    edge_mapping[(u, v)] = (u, v, min_k) # Map for stats computation later
            
            logger.info(f"Simplified graph created with {len(G_simple.nodes)} nodes and {len(G_simple.edges)} edges in {time.time() - simplify_graph_start:.2f}s.")
        except Exception as e:
            logger.error(f"Error creating simplified graph: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to simplify road network for eco-route calculation.")

        # --- Calculate Eco Route ---
        logger.info("Calculating eco-friendly route...")
        eco_route_calc_start = time.time()
        try:
            # Find the shortest path based on "eco_cost" in the simplified graph
            eco_route_nodes = nx.shortest_path(G_simple, orig_node, dest_node, weight="eco_cost")
            logger.info(f"Eco-friendly route calculated with {len(eco_route_nodes)} nodes in {time.time() - eco_route_calc_start:.2f}s.")
        except nx.NetworkXNoPath:
            logger.error("No eco-friendly path found between origin and destination nodes.")
            raise HTTPException(status_code=404, detail="No eco-friendly route found between the specified origin and destination.")
        except Exception as e:
            logger.error(f"Error calculating eco-friendly route: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to calculate the eco-friendly route.")

        # --- Convert Routes to Geographic Coordinates ---
        logger.info("Converting routes from node IDs to geographic coordinates...")
        convert_coords_start = time.time()
        try:
            eco_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in eco_route_nodes]
            # Use the final fastest route (after all penalties/traffic applied)
            final_fastest_route_nodes = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
            fastest_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in final_fastest_route_nodes]
            logger.info(f"Routes converted to coordinates in {time.time() - convert_coords_start:.2f}s.")
        except Exception as e:
            logger.error(f"Error converting route nodes to coordinates: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to convert route to geographic coordinates.")

        # --- Calculate Statistics for Eco and Fastest (OSMnx) Routes ---
        logger.info("Calculating statistics for eco and fastest (OSMnx) routes...")
        stats_calc_start = time.time()
        try:
            eco_dist, eco_time, eco_co2 = compute_stats(eco_route_nodes, vehicle, G, edge_mapping, co2_map)
            # Recompute fastest route stats in case travel_time changed further (e.g., more precise TomTom)
            fast_dist, fast_time, fast_co2 = compute_stats(final_fastest_route_nodes, vehicle, G, edge_mapping, co2_map)
            logger.info(f"OSMnx Eco Route Stats: Distance={eco_dist:.2f}km, Time={eco_time:.1f}min, CO2={eco_co2:.3f}kg")
            logger.info(f"OSMnx Fastest Route Stats: Distance={fast_dist:.2f}km, Time={fast_time:.1f}min, CO2={fast_co2:.3f}kg")
            logger.info(f"OSMnx stats calculated in {time.time() - stats_calc_start:.2f}s.")
        except Exception as e:
            logger.error(f"Error calculating OSMnx route statistics: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to calculate route statistics.")

        # --- Get Google Route (for comparison) ---
        logger.info("Fetching Google Maps route for comparison...")
        google_route_start = time.time()
        google_client = None
        try:
            google_client = googlemaps.Client(key=API_KEY_GOOGLE)
            google_dir_response = google_client.directions(
                origin_coords,
                destination_coords,
                mode="driving",
                departure_time="now" # Important for real-time traffic
            )
            
            if not google_dir_response:
                raise ValueError("Google Maps API returned no directions.")
                
            google_poly = google_dir_response[0]["overview_polyline"]["points"]
            google_route_coords = polyline.decode(google_poly)
            
            # Extract basic stats from Google response
            google_leg = google_dir_response[0]["legs"][0]
            google_distance = google_leg["distance"]["value"] / 1000 # meters to km
            # Prefer duration_in_traffic if available
            google_duration = google_leg.get("duration_in_traffic", google_leg["duration"])["value"] / 60 # seconds to minutes
            
            # Estimate Google route CO2 using our CO2 map and Google's distance
            google_co2_estimated = google_distance * (co2_map.get(vehicle, 180000) / 1_000_000)
            
            logger.info(f"Google Route Stats: Distance={google_distance:.2f}km, Time={google_duration:.1f}min, CO2={google_co2_estimated:.3f}kg")
            logger.info(f"Google Maps route fetched in {time.time() - google_route_start:.2f}s.")
        except Exception as e:
            logger.error(f"Error fetching Google Maps route: {e}", exc_info=True)
            # If Google API fails, populate with default/empty values
            google_route_coords = []
            google_distance = 0.0
            google_duration = 0.0
            google_co2_estimated = 0.0
            logger.warning("Google Maps route data could not be fetched. Providing empty Google route details.")
            if not API_KEY_GOOGLE:
                 logger.error("Google Maps API Key is missing. Please set Maps_API_KEY environment variable.")

        # --- Get Google ETA for Eco Route (if Google client is available) ---
        logger.info("Fetching Google ETA for the calculated eco-route...")
        google_eco_eta_start = time.time()
        eco_google_duration = eco_time # Default to our calculated time
        if google_client and eco_coords:
            try:
                # Sample waypoints to avoid exceeding Google API limits (max 23 intermediate waypoints)
                waypoints_for_google_eco = sample_waypoints(eco_coords)
                
                # Google Directions API request for the eco-route coordinates
                eco_dir_response = google_client.directions(
                    origin=eco_coords[0],
                    destination=eco_coords[-1],
                    waypoints=waypoints_for_google_eco,
                    mode="driving",
                    departure_time="now"
                )
                
                if eco_dir_response:
                    # Sum up durations from all legs
                    eco_google_duration = sum(
                        leg.get("duration_in_traffic", leg["duration"])["value"]
                        for leg in eco_dir_response[0]["legs"]
                    ) / 60 # seconds to minutes
                    logger.info(f"Google ETA for eco route: {eco_google_duration:.1f}min in {time.time() - google_eco_eta_start:.2f}s.")
                else:
                    logger.warning("Google Maps API returned no directions for eco-route waypoints.")
            except Exception as e:
                logger.error(f"Error getting Google ETA for eco route: {e}", exc_info=True)
                logger.warning("Using internal calculated eco time as Google ETA failed.")
        else:
            logger.info("Skipping Google ETA for eco-route as Google Maps API is not available or eco_coords are empty.")


        # --- Prepare Response ---
        logger.info("Preparing final response data.")
        response = RouteResponse(
            eco_route=eco_coords,
            google_route=google_route_coords,
            eco_stats={
                "distance_km": round(eco_dist, 2),
                "time_minutes": round(eco_time, 1), # Our calculated time
                "time_minutes_google_estimated": round(eco_google_duration, 1), # Google's estimate for our eco-route
                "co2_kg": round(eco_co2, 3)
            },
            google_stats={
                "distance_km": round(google_distance, 2),
                "time_minutes": round(google_duration, 1),
                "co2_kg": round(google_co2_estimated, 3)
            },
            comparison={
                "co2_savings_kg": round(fast_co2 - eco_co2, 3), # Savings comparing our fastest vs our eco
                "co2_savings_percent": round((fast_co2 - eco_co2) / fast_co2 * 100, 1) if fast_co2 > 0 else 0,
                # Time difference between Google's fastest route and Google's estimate for our eco route
                "time_difference_minutes": round(eco_google_duration - google_duration, 1)
            }
        )

        logger.info(f"Route calculation completed successfully in {time.time() - start_time:.2f}s.")
        return response

    except HTTPException:
        # Re-raise HTTP exceptions (e.g., 400, 404, 500 from within the function)
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
    os.environ["Maps_API_KEY"] = "AIzaSyBn6r80g6qTuIgaKnJpK4-oWAkWls5YtL0"
    os.environ["TOMTOM_API_KEY"] = "WzqAKEeo0JHT9Z3SU05ZkgL8rqFbzHGN"
    uvicorn.run(app, host="0.0.0.0", port=8000)