from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional
import osmnx as ox
import networkx as nx
import googlemaps
import requests
import pandas as pd
import polyline
import math
import json
import time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

app = FastAPI(title="EcoRoute API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CSV_PATH = "export-hbefa.csv"
API_KEY_GOOGLE = "AIzaSyBn6r80g6qTuIgaKnJpK4-oWAkWls5YtL0"
API_KEY_TOMTOM = "WzqAKEeo0JHT9Z3SU05ZkgL8rqFbzHGN"

# Global variables for caching
co2_map = {}
elev_cache = {}
tomtom_cache = {}

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
            df["Emission factor"] = pd.to_numeric(df["Emission factor"].str.replace(",", ""), errors="coerce")
            co2_map = df.groupby("Vehicle category")["Emission factor"].mean().to_dict()
        except FileNotFoundError:
            # Default CO2 values if CSV not found
            co2_map = {
                "motorcycle": 150000,
                "pass. car": 180000,
                "LCV": 220000,
                "coach": 800000,
                "HGV": 900000,
                "urban bus": 1200000
            }
    return co2_map

def vehicle_mass_kg(vehicle):
    """Return vehicle mass in kg"""
    return {
        "motorcycle": 200,
        "pass. car": 1500,
        "LCV": 2500,
        "coach": 11000,
        "HGV": 18000,
        "urban bus": 14000
    }.get(vehicle, 1500)

def load_elevation_cache():
    """Load elevation cache from file"""
    global elev_cache
    elev_cache_path = "elevation_cache.json"
    if os.path.exists(elev_cache_path):
        with open(elev_cache_path, "r") as f:
            elev_cache = json.load(f)
        elev_cache = {int(k): v for k, v in elev_cache.items()}
    return elev_cache

def save_elevation_cache():
    """Save elevation cache to file"""
    with open("elevation_cache.json", "w") as f:
        json.dump(elev_cache, f)

def load_tomtom_cache():
    """Load TomTom cache from file"""
    global tomtom_cache
    tomtom_cache_path = "tomtom_cache.json"
    if os.path.exists(tomtom_cache_path):
        with open(tomtom_cache_path, "r") as f:
            tomtom_cache = json.load(f)
    return tomtom_cache

def save_tomtom_cache():
    """Save TomTom cache to file"""
    with open("tomtom_cache.json", "w") as f:
        json.dump(tomtom_cache, f)

def fetch_elevation_batch(coords):
    """Fetch elevation data for a batch of coordinates"""
    loc_str = "|".join(f"{lat},{lon}" for lat, lon in coords)
    try:
        r = requests.get(f"https://api.opentopodata.org/v1/eudem25m?locations={loc_str}")
        results = r.json().get("results", [])
        return [res.get("elevation", 0.0) for res in results]
    except:
        return [0.0] * len(coords)

def fetch_speed(item, retries=3):
    """Fetch speed data from TomTom API"""
    key, (lat, lon) = item
    for _ in range(retries):
        try:
            url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&key={API_KEY_TOMTOM}"
            res = requests.get(url, timeout=5)
            if res.status_code == 200:
                speed = res.json()["flowSegmentData"]["currentSpeed"]
                return key, speed
        except:
            time.sleep(0.5 + random.uniform(0, 0.3))
    return key, None

def calculate_bearing(p1, p2):
    """Calculate bearing between two points"""
    lat1, lon1 = map(math.radians, (p1["y"], p1["x"]))
    lat2, lon2 = map(math.radians, (p2["y"], p2["x"]))
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def base_eco_cost(u, v, data, vehicle, G, co2_map):
    """Calculate ecological cost for an edge"""
    length = data.get("length", 1)  # meters
    elev_gain = max(0, G.nodes[v]["elevation"] - G.nodes[u]["elevation"])
    mass = vehicle_mass_kg(vehicle)  # kg

    base_rate = co2_map[vehicle] / 1000000  # g/km → kg/m
    base_emissions = length * base_rate  # kg

    # Approximate slope-induced emissions
    slope_emissions = mass * 9.81 * elev_gain * (0.074 / 1e6)  # in kg

    turn_penalty = data.get("turn_penalty", 0) * 0.1  # optional penalty in kg
    return base_emissions + slope_emissions + turn_penalty

def compute_stats(route, vehicle, G, edge_mapping, co2_map):
    """Compute route statistics"""
    dist = time_val = co2 = 0
    for u, v in zip(route[:-1], route[1:]):
        u, v, k = edge_mapping.get((u, v), (u, v, 0))
        data = G[u][v][k]
        dist += data.get("length", 0)
        time_val += data.get("travel_time", 0)
        co2 += data.get("eco_cost", 0)
    return dist / 1000, time_val / 60, co2 / 1000  # km, minutes, kg

def sample_waypoints(coords, max_waypoints=23):
    """Sample waypoints from route coordinates"""
    if len(coords) <= 2:
        return []
    step = max(1, len(coords) // max_waypoints)
    return [coords[i] for i in range(1, len(coords) - 1, step)][:max_waypoints]

@app.on_event("startup")
async def startup_event():
    """Initialize data on startup"""
    load_co2_data()
    load_elevation_cache()
    load_tomtom_cache()

@app.get("/")
async def root():
    return {"message": "EcoRoute API is running"}

@app.get("/vehicles")
async def get_vehicles():
    """Get available vehicle types"""
    return {
        "vehicles": [
            {"id": "motorcycle", "name": "Motorcycle"},
            {"id": "pass. car", "name": "Passenger Car"},
            {"id": "LCV", "name": "Light Commercial Vehicle"},
            {"id": "coach", "name": "Coach"},
            {"id": "HGV", "name": "Heavy Goods Vehicle"},
            {"id": "urban bus", "name": "Urban Bus"}
        ]
    }

import logging
import time
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def safe_get(d, key, default=0.0):
    val = d.get(key)
    return val if val is not None else default

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
    """Calculate eco-friendly and fastest routes"""
    start_time = time.time()
    logger.info(f"Starting route calculation: {request.origin} -> {request.destination}, vehicle: {request.vehicle}")
    
    try:
        origin = request.origin
        destination = request.destination
        vehicle = request.vehicle

        # Load data
        logger.debug("Loading CO2 data...")
        load_start = time.time()
        co2_data = load_co2_data()
        log_performance("load_co2_data", load_start, vehicle_types=len(co2_data))
        
        # Build graph
        logger.debug(f"Building graph from origin point: {origin}")
        graph_start = time.time()
        try:
            G = ox.graph_from_point(origin, dist=5000, network_type="drive")
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
            log_performance("build_graph", graph_start)
            log_graph_stats(G, "after_initial_build")
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to build road network: {str(e)}")

        # Add elevation data
        logger.debug("Adding elevation data...")
        elev_start = time.time()
        try:
            global elev_cache
            uncached_nodes = [node for node in G.nodes if node not in elev_cache]
            logger.info(f"Need elevation for {len(uncached_nodes)} nodes (cache has {len(elev_cache)} entries)")
            
            if uncached_nodes:
                uncached_coords = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in uncached_nodes]
                
                batch_size = 100
                batches = [uncached_coords[i:i + batch_size] for i in range(0, len(uncached_coords), batch_size)]
                logger.debug(f"Processing {len(batches)} elevation batches")
                
                elevations = []
                with ThreadPoolExecutor(max_workers=5) as executor:
                    results = executor.map(fetch_elevation_batch, batches)
                    for batch_idx, batch_result in enumerate(results):
                        logger.debug(f"Batch {batch_idx + 1}/{len(batches)} returned {len(batch_result)} elevations")
                        elevations.extend(batch_result)
                
                # Cache elevations
                for node, elev in zip(uncached_nodes, elevations):
                    elev = elev if elev is not None else 0.0
                    G.nodes[node]["elevation"] = elev
                    elev_cache[node] = elev
                
                save_elevation_cache()
                logger.info(f"Cached {len(elevations)} new elevations")
            
            # Ensure all nodes have elevation
            missing_elev = 0
            for node in G.nodes:
                if "elevation" not in G.nodes[node] or G.nodes[node]["elevation"] is None:
                    G.nodes[node]["elevation"] = 0.0
                    missing_elev += 1
            
            if missing_elev > 0:
                logger.warning(f"Set default elevation for {missing_elev} nodes")
                
            log_performance("add_elevation", elev_start, nodes_processed=len(uncached_nodes))
            
        except Exception as e:
            logger.error(f"Error adding elevation data: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to add elevation data: {str(e)}")

        # Add grades
        logger.debug("Adding edge grades...")
        grade_start = time.time()
        try:
            G = ox.add_edge_grades(G)
            log_performance("add_grades", grade_start)
            log_graph_stats(G, "after_grades")
        except Exception as e:
            logger.error(f"Error adding grades: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to add grades: {str(e)}")

        # Find nearest nodes
        logger.debug("Finding nearest nodes...")
        nearest_start = time.time()
        try:
            orig_node = ox.distance.nearest_nodes(G, X=origin[1], Y=origin[0])
            dest_node = ox.distance.nearest_nodes(G, X=destination[1], Y=destination[0])
            logger.info(f"Nearest nodes: origin={orig_node}, destination={dest_node}")
            log_performance("find_nearest", nearest_start)
        except Exception as e:
            logger.error(f"Error finding nearest nodes: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to find nearest nodes: {str(e)}")

        # Calculate fastest route
        logger.debug("Calculating fastest route...")
        fastest_start = time.time()
        try:
            fastest_route = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
            route_edges = set(zip(fastest_route[:-1], fastest_route[1:]))
            logger.info(f"Fastest route: {len(fastest_route)} nodes, {len(route_edges)} edges")
            log_performance("fastest_route", fastest_start)
        except Exception as e:
            logger.error(f"Error calculating fastest route: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to calculate fastest route: {str(e)}")

        # Fetch TomTom data
        logger.debug("Fetching TomTom traffic data...")
        tomtom_start = time.time()
        try:
            global tomtom_cache
            coords_to_fetch = {}
            edge_keys = []
            
            for u, v in route_edges:
                for k in G[u][v]:
                    lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
                    lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
                    key = f"{lat:.5f},{lon:.5f}"
                    if key not in tomtom_cache:
                        coords_to_fetch[key] = (lat, lon)
                    edge_keys.append((u, v, k, key))

            logger.info(f"Need to fetch {len(coords_to_fetch)} TomTom speeds (cache has {len(tomtom_cache)} entries)")

            # Fetch speeds
            if coords_to_fetch:
                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_key = {executor.submit(fetch_speed, item): item[0] for item in coords_to_fetch.items()}
                    completed = 0
                    for future in as_completed(future_to_key):
                        key, speed = future.result()
                        tomtom_cache[key] = speed
                        completed += 1
                        if completed % 50 == 0:
                            logger.debug(f"Fetched {completed}/{len(coords_to_fetch)} speeds")
                save_tomtom_cache()

            # Apply speeds
            speed_applied = 0
            for u, v, k, key in edge_keys:
                speed = tomtom_cache.get(key)
                if speed is not None:
                    G[u][v][k]["speed_kph"] = speed
                    length = safe_get(G[u][v][k], "length", 1.0)
                    G[u][v][k]["travel_time"] = length / (speed * 1000 / 3600)
                    speed_applied += 1
            
            logger.info(f"Applied speeds to {speed_applied}/{len(edge_keys)} edges")
            log_performance("tomtom_data", tomtom_start, coords_fetched=len(coords_to_fetch))
            
        except Exception as e:
            logger.error(f"Error fetching TomTom data: {e}")
            logger.warning("Continuing without TomTom data...")

        # Calculate turn penalties
        logger.debug("Calculating turn penalties...")
        penalty_start = time.time()
        try:
            for u, v, k in G.edges(keys=True):
                G[u][v][k]["turn_penalty"] = 0

            path = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
            penalties_added = 0
            
            for i in range(1, len(path) - 1):
                u, v, w = path[i - 1], path[i], path[i + 1]
                b1 = calculate_bearing(G.nodes[u], G.nodes[v])
                b2 = calculate_bearing(G.nodes[v], G.nodes[w])
                delta = abs(b2 - b1)
                if delta > 180:
                    delta = 360 - delta

                if delta > 170:
                    penalty = 90
                elif delta > 150:
                    penalty = 60
                elif delta > 120:
                    penalty = 45
                elif delta > 90:
                    penalty = 30
                elif delta > 45:
                    penalty = 15
                else:
                    penalty = 5

                for k in G[u][v]:
                    G[u][v][k]["turn_penalty"] = G[u][v][k].get("turn_penalty", 0) + penalty
                    G[u][v][k]["travel_time"] = safe_get(G[u][v][k], "travel_time") + penalty
                    penalties_added += 1

            logger.info(f"Added {penalties_added} turn penalties")
            log_performance("turn_penalties", penalty_start)
            
        except Exception as e:
            logger.error(f"Error calculating turn penalties: {e}")
            logger.warning("Continuing without turn penalties...")

        # Calculate eco costs
        logger.debug("Calculating eco costs...")
        eco_cost_start = time.time()
        try:
            eco_costs = {}
            for u, v, k, d in G.edges(keys=True, data=True):
                try:
                    cost = base_eco_cost(u, v, d, vehicle, G, co2_data)
                    eco_costs[(u, v, k)] = cost
                except Exception as e:
                    logger.warning(f"Error calculating eco cost for edge {u}-{v}-{k}: {e}")
                    eco_costs[(u, v, k)] = 1.0
            
            nx.set_edge_attributes(G, eco_costs, "eco_cost")
            logger.info(f"Calculated eco costs for {len(eco_costs)} edges")
            log_performance("eco_costs", eco_cost_start)
            
        except Exception as e:
            logger.error(f"Error calculating eco costs: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to calculate eco costs: {str(e)}")

        # Create simplified graph for eco route
        logger.debug("Creating simplified graph for eco routing...")
        simplify_start = time.time()
        try:
            G_simple = nx.DiGraph()
            edge_mapping = {}
            for u, v in G.edges():
                min_key = min(G[u][v], key=lambda k: G[u][v][k].get("eco_cost", 1e12))
                data = G[u][v][min_key]
                G_simple.add_edge(u, v, **data)
                edge_mapping[(u, v)] = (u, v, min_key)
            
            logger.info(f"Simplified graph: {len(G_simple.nodes)} nodes, {len(G_simple.edges)} edges")
            log_performance("simplify_graph", simplify_start)
            
        except Exception as e:
            logger.error(f"Error creating simplified graph: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to create simplified graph: {str(e)}")

        # Calculate eco route
        logger.debug("Calculating eco route...")
        eco_route_start = time.time()
        try:
            eco_route = next(nx.shortest_simple_paths(G_simple, orig_node, dest_node, weight="eco_cost"))
            logger.info(f"Eco route: {len(eco_route)} nodes")
            log_performance("eco_route", eco_route_start)
        except Exception as e:
            logger.error(f"Error calculating eco route: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to calculate eco route: {str(e)}")

        # Convert routes to coordinates
        logger.debug("Converting routes to coordinates...")
        coord_start = time.time()
        try:
            eco_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in eco_route]
            fastest_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in fastest_route]
            log_performance("convert_coords", coord_start)
        except Exception as e:
            logger.error(f"Error converting coordinates: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to convert coordinates: {str(e)}")

        # Calculate statistics
        logger.debug("Calculating route statistics...")
        stats_start = time.time()
        try:
            eco_dist, eco_time, eco_co2 = compute_stats(eco_route, vehicle, G, edge_mapping, co2_data)
            fast_dist, fast_time, fast_co2 = compute_stats(fastest_route, vehicle, G, edge_mapping, co2_data)
            logger.info(f"Route stats calculated - Eco: {eco_dist:.1f}km, {eco_time:.1f}min, {eco_co2:.3f}kg CO2")
            logger.info(f"Route stats calculated - Fast: {fast_dist:.1f}km, {fast_time:.1f}min, {fast_co2:.3f}kg CO2")
            log_performance("calculate_stats", stats_start)
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to calculate statistics: {str(e)}")

        # Get Google route
        logger.debug("Fetching Google route...")
        google_start = time.time()
        try:
            gmaps = googlemaps.Client(key=API_KEY_GOOGLE)
            google_dir = gmaps.directions(origin, destination, mode="driving", departure_time="now")
            
            if not google_dir:
                raise ValueError("No Google route found")
                
            google_poly = google_dir[0]["overview_polyline"]["points"]
            google_route = polyline.decode(google_poly)
            google_distance = google_dir[0]["legs"][0]["distance"]["value"] / 1000
            google_duration = google_dir[0]["legs"][0].get("duration_in_traffic", google_dir[0]["legs"][0]["duration"])["value"] / 60
            google_co2 = google_distance * (co2_data[vehicle] / 1000000)
            
            logger.info(f"Google route: {google_distance:.1f}km, {google_duration:.1f}min, {google_co2:.3f}kg CO2")
            log_performance("google_route", google_start)
            
        except Exception as e:
            logger.error(f"Error fetching Google route: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to fetch Google route: {str(e)}")

        # Get Google ETA for eco route
        logger.debug("Getting Google ETA for eco route...")
        google_eco_start = time.time()
        try:
            waypoints = sample_waypoints(eco_coords)
            directions_eco = gmaps.directions(
                origin=eco_coords[0],
                destination=eco_coords[-1],
                waypoints=waypoints,
                mode="driving",
                departure_time="now"
            )
            
            if not directions_eco:
                raise ValueError("No Google directions found for eco route")
                
            eco_google_duration = sum(
                leg.get("duration_in_traffic", leg["duration"])["value"]
                for leg in directions_eco[0]["legs"]
            ) / 60
            
            logger.info(f"Google ETA for eco route: {eco_google_duration:.1f}min")
            log_performance("google_eco_eta", google_eco_start)
            
        except Exception as e:
            logger.error(f"Error getting Google ETA for eco route: {e}")
            logger.warning("Using calculated eco time instead of Google ETA")
            eco_google_duration = eco_time

        # Prepare response
        logger.debug("Preparing response...")
        response = RouteResponse(
            eco_route=eco_coords,
            google_route=google_route,
            eco_stats={
                "distance_km": round(eco_dist, 2),
                "time_minutes": round(eco_time, 1),
                "time_minutes_google": round(eco_google_duration, 1),
                "co2_kg": round(eco_co2, 3)
            },
            google_stats={
                "distance_km": round(google_distance, 2),
                "time_minutes": round(google_duration, 1),
                "co2_kg": round(google_co2, 3)
            },
            comparison={
                "co2_savings_kg": round(fast_co2 - eco_co2, 3),
                "co2_savings_percent": round((fast_co2 - eco_co2) / fast_co2 * 100, 1) if fast_co2 > 0 else 0,
                "time_difference_minutes": round(eco_google_duration - google_duration, 1)
            }
        )

        log_performance("total_route_calculation", start_time)
        logger.info("Route calculation completed successfully")
        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in route calculation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)