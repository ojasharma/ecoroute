import logging
import time
import os
import math
from typing import Dict, Any, List, Tuple, Optional, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

import networkx as nx
import osmnx as ox
import googlemaps
import polyline
from fastapi import HTTPException
from geopy.distance import geodesic

from app.utils import *
from app.models import RouteRequest, RouteResponse

logger = logging.getLogger(__name__)


def base_eco_cost(u: int, v: int, data: Dict[str, Any], vehicle: str, G: nx.MultiDiGraph) -> float:
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

def compute_stats(route: List[int], vehicle: str, G: nx.MultiDiGraph, edge_mapping: Dict[Tuple[int, int], Tuple[int, int, int]]) -> Tuple[float, float, float]:
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

        try:  # Removed incomplete try block to fix syntax error
            data = G[original_u][original_v][original_k]
            
            total_dist += data.get("length", 0.0)
            total_time += data.get("travel_time", 0.0)
            total_co2 += data.get("eco_cost", 0.0)
        except KeyError:
            logger.warning(f"Edge ({original_u}, {original_v}, {original_k}) not found in original graph G. Skipping stats for this segment.")
            continue # Skip this segment if edge data is missing

    return total_dist / 1000, total_time / 60, total_co2 / 1000  # km, minutes, kg

def download_graph_chunk(center_coords, chunk_size_km=25, retry_count=3):
    """
    Download a single graph chunk with retry logic.
    
    Args:
        center_coords: (lat, lon) tuple for center
        chunk_size_km: Size of chunk in kilometers
        retry_count: Number of retry attempts
    
    Returns:
        networkx.MultiDiGraph or None: Road network graph chunk
    """
    west, south, east, north = calculate_chunk_bounds(center_coords, chunk_size_km)
    
    for attempt in range(retry_count):
        try:
            logger.info(f"Downloading chunk {attempt + 1}/{retry_count} around {center_coords}")
            
            bbox = (west, south, east, north)
            G = ox.graph_from_bbox(
                bbox=bbox,
                network_type="drive",
                simplify=True
            )
            
            logger.info(f"Downloaded chunk with {len(G.nodes)} nodes and {len(G.edges)} edges")
            return G
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for chunk {center_coords}: {e}")
            if attempt < retry_count - 1:
                # Wait before retry, with exponential backoff
                wait_time = (2 ** attempt) * 5  # 5, 10, 20 seconds
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download chunk after {retry_count} attempts")
                return None

def merge_graph_chunks(graph_chunks):
    """
    Merge multiple graph chunks into a single graph with memory optimization.
    Uses nx.compose() to handle overlapping nodes properly.
    
    Args:
        graph_chunks: List of networkx.MultiDiGraph objects
    
    Returns:
        networkx.MultiDiGraph: Merged graph
    """
    if not graph_chunks:
        raise ValueError("No graph chunks to merge")
    
    logger.info(f"Merging {len(graph_chunks)} graph chunks...")
    
    # Start with the first chunk
    merged_graph = graph_chunks[0].copy()
    
    # Clear the first chunk from memory
    del graph_chunks[0]
    
    # Merge remaining chunks one by one and clear them immediately
    for i, chunk in enumerate(graph_chunks):
        logger.info(f"Merging chunk {i+2}/{len(graph_chunks)+1}")
        
        # Use compose instead of union to handle overlapping nodes
        merged_graph = nx.compose(merged_graph, chunk)
        
        # Clear the chunk from memory immediately
        del chunk
        
        # Force garbage collection every few chunks
        if i % 3 == 0:
            import gc
            gc.collect()
    
    # Final cleanup
    del graph_chunks
    import gc
    gc.collect()
    
    logger.info(f"Merged graph has {len(merged_graph.nodes)} nodes and {len(merged_graph.edges)} edges")
    return merged_graph

def build_optimized_road_network(origin_coords, destination_coords, chunk_size_km=25, use_cache=True):
    """
    Build road network graph using chunked downloading for long distances with caching.
    
    Args:
        origin_coords: (lat, lon) tuple for origin
        destination_coords: (lat, lon) tuple for destination
        chunk_size_km: Size of each chunk in kilometers
        use_cache: Whether to use caching
    
    Returns:
        networkx.MultiDiGraph: Road network graph
    """
    logger.info("Building road network graph using chunked approach...")
    graph_build_start = time.time()
    
    try:
        
        # Calculate total distance
        total_distance_km = geodesic(origin_coords, destination_coords).kilometers
        logger.info(f"Total route distance: {total_distance_km:.2f} km")
        
        # Determine strategy based on distance
        if total_distance_km <= 20:
            # For short distances, use simple rectangular approach
            logger.info("Using simple rectangular approach for short distance")
            return build_simple_rectangular_network(origin_coords, destination_coords)
        
        # For long distances, use chunked approach
        logger.info(f"Using chunked approach with {chunk_size_km}km chunks")
        
        # Calculate number of chunks needed - more chunks for better coverage
        num_chunks = max(5, int(total_distance_km / (chunk_size_km * 0.7)) + 3)  # More overlap
        
        # Create interpolated points along the route
        route_points = interpolate_points_along_route(origin_coords, destination_coords, num_chunks)
        
        logger.info(f"Created {len(route_points)} points along route for chunked download")
        
        # Download chunks with memory management
        graph_chunks = []
        successful_downloads = 0
        cache_hits = 0
        max_chunks_in_memory = 5  # Limit chunks in memory
        
        for i, point in enumerate(route_points):
            logger.info(f"Processing chunk {i+1}/{len(route_points)}")
            
            # Check if this will be a cache hit
            cache_key = get_chunk_cache_key(point, chunk_size_km)
            cached_chunk = None
            if use_cache:
                cached_chunk = load_chunk_from_cache(cache_key)

            if cached_chunk is not None:
                chunk = cached_chunk
                cache_hits += 1
            else:
                chunk = download_graph_chunk(point, chunk_size_km)
                if use_cache and chunk is not None:
                    save_chunk_to_cache(chunk, cache_key)
                
            if chunk is not None:
                graph_chunks.append(chunk)
                successful_downloads += 1
                
                # If we have too many chunks in memory, start merging
                if len(graph_chunks) >= max_chunks_in_memory:
                    logger.info(f"Merging {len(graph_chunks)} chunks to save memory")
                    merged_partial = merge_graph_chunks(graph_chunks)
                    graph_chunks = [merged_partial]  # Keep only the merged result
            else:
                logger.warning(f"Failed to download chunk {i+1}, continuing with available chunks")
        
        if successful_downloads == 0:
            raise Exception("Failed to download any graph chunks")
        
        logger.info(f"Successfully processed {successful_downloads}/{len(route_points)} chunks "
                   f"({cache_hits} cache hits, {successful_downloads - cache_hits} downloads)")
        
        # Final merge of remaining chunks
        if len(graph_chunks) > 1:
            merged_graph = merge_graph_chunks(graph_chunks)
        else:
            merged_graph = graph_chunks[0]
        
        # Clear chunks from memory
        del graph_chunks
        import gc
        gc.collect()
        
        # Add speeds and travel times with progress logging
        logger.info("Adding edge speeds...")
        merged_graph = ox.add_edge_speeds(merged_graph)
        
        logger.info("Adding travel times...")
        merged_graph = ox.add_edge_travel_times(merged_graph)
        
        # Final memory cleanup
        import gc
        gc.collect()
        
        logger.info(f"Final graph built with {len(merged_graph.nodes)} nodes and {len(merged_graph.edges)} edges "
                   f"in {time.time() - graph_build_start:.2f}s")
        
        return merged_graph
        
    except Exception as e:
        logger.error(f"Failed to build road network graph: {e}", exc_info=True)
        raise Exception(f"Failed to build road network graph: {e}")

def build_simple_rectangular_network(origin_coords, destination_coords, buffer_km=5, width_ratio=0.4):
    """
    Build road network using simple rectangular approach for shorter distances.
    
    Args:
        origin_coords: (lat, lon) tuple for origin
        destination_coords: (lat, lon) tuple for destination
        buffer_km: Buffer distance in kilometers
        width_ratio: Width as ratio of total distance
    
    Returns:
        networkx.MultiDiGraph: Road network graph
    """
    # Calculate distance
    distance_km = geodesic(origin_coords, destination_coords).kilometers
    
    # Calculate center point
    center_lat = (origin_coords[0] + destination_coords[0]) / 2
    center_lon = (origin_coords[1] + destination_coords[1]) / 2
    
    # Calculate bounds with buffer
    total_distance = distance_km + (2 * buffer_km)
    width = max(total_distance * width_ratio, 20)  # Minimum 20km width
    
    # Calculate approximate degree offsets
    lat_offset = total_distance / 111.0 / 2
    lon_offset = width / (111.0 * math.cos(math.radians(center_lat))) / 2
    
    north = center_lat + lat_offset
    south = center_lat - lat_offset
    east = center_lon + lon_offset
    west = center_lon - lon_offset
    
    logger.info(f"Downloading simple rectangular region: "
               f"N={north:.4f}, S={south:.4f}, E={east:.4f}, W={west:.4f}")
    
    bbox = (west, south, east, north)
    G = ox.graph_from_bbox(
        bbox=bbox,
        network_type="drive",
        simplify=True
    )
    
    # Add speeds and travel times
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    
    return G

async def calculate_route(request: RouteRequest) -> RouteResponse:
    """
    Calculates eco-friendly and fastest routes.
    This function contains the main routing logic, extracted from the FastAPI endpoint.
    """
    origin_coords = request.origin
    destination_coords = request.destination
    vehicle = request.vehicle

    if vehicle not in co2_map:
        logger.error(f"Invalid vehicle type: {vehicle}. Available types: {list(co2_map.keys())}")
        raise HTTPException(status_code=400, detail=f"Invalid vehicle type '{vehicle}'. Please choose from: {', '.join(co2_map.keys())}")

    # --- Graph Building ---
    logger.info("Building road network graph...")
    graph_build_start = time.time()
    try:
        G = build_optimized_road_network(origin_coords, destination_coords, chunk_size_km=25)
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
        fastest_route_nodes = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
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
        edge_tomtom_keys = [] 
        
        for u, v in fastest_route_edges_uv:
            for k in G[u][v]:
                lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
                lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
                key = f"{lat:.5f},{lon:.5f}"
                if key not in tomtom_cache:
                    coords_to_fetch[key] = (lat, lon)
                edge_tomtom_keys.append((u, v, k, key))

        logger.info(f"Identified {len(coords_to_fetch)} unique points for TomTom fetching (from {len(edge_tomtom_keys)} edges).")

        if coords_to_fetch:
            fetch_items = list(coords_to_fetch.items())
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_key = {executor.submit(fetch_speed, item): item[0] for item in fetch_items}
                for i, future in enumerate(as_completed(future_to_key)):
                    key, speed = future.result()
                    tomtom_cache[key] = speed
                    if i % 50 == 0 and i > 0:
                        logger.debug(f"Fetched {i}/{len(fetch_items)} TomTom speeds.")
            save_tomtom_cache()
            logger.info(f"Completed TomTom fetching for {len(coords_to_fetch)} points.")

        speeds_applied_count = 0
        for u, v, k, key in edge_tomtom_keys:
            speed_kph = tomtom_cache.get(key)
            if speed_kph is not None and speed_kph > 0:
                G[u][v][k]["speed_kph"] = speed_kph
                length_m = safe_get(G[u][v][k], "length", 1.0)
                G[u][v][k]["travel_time"] = length_m / (speed_kph * 1000 / 3600)
                speeds_applied_count += 1
            else:
                current_speed = G[u][v][k].get("speed_kph", 30.0)
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
        fastest_route_nodes = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
        
        nx.set_edge_attributes(G, 0, "turn_penalty")

        penalties_added_count = 0
        for i in range(1, len(fastest_route_nodes) - 1):
            u, v, w = fastest_route_nodes[i - 1], fastest_route_nodes[i], fastest_route_nodes[i + 1]
            
            b1 = calculate_bearing(G.nodes[u], G.nodes[v])
            b2 = calculate_bearing(G.nodes[v], G.nodes[w])
            
            delta = abs(b2 - b1)
            if delta > 180:
                delta = 360 - delta

            penalty_seconds = 0
            if delta > 170:
                penalty_seconds = 90
            elif delta > 150:
                penalty_seconds = 60
            elif delta > 120:
                penalty_seconds = 45
            elif delta > 90:
                penalty_seconds = 30
            elif delta > 45:
                penalty_seconds = 15
            else:
                penalty_seconds = 5

            for k in G[u][v]:
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
            cost = base_eco_cost(u, v, d, vehicle, G) # Pass G directly for elevation lookup
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
        G_simple = nx.DiGraph()
        edge_mapping = {}

        for u, v in G.edges():
            min_k = None
            min_eco_cost = float('inf')
            for k in G[u][v]:
                current_eco_cost = G[u][v][k].get("eco_cost", float('inf'))
                if current_eco_cost < min_eco_cost:
                    min_eco_cost = current_eco_cost
                    min_k = k
            
            if min_k is not None:
                G_simple.add_edge(u, v, **G[u][v][min_k])
                edge_mapping[(u, v)] = (u, v, min_k)
        
        logger.info(f"Simplified graph created with {len(G_simple.nodes)} nodes and {len(G_simple.edges)} edges in {time.time() - simplify_graph_start:.2f}s.")
    except Exception as e:
        logger.error(f"Error creating simplified graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to simplify road network for eco-route calculation.")

    # --- Calculate Eco Route ---
    logger.info("Calculating eco-friendly route...")
    eco_route_calc_start = time.time()
    try:
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
        eco_dist, eco_time, eco_co2 = compute_stats(eco_route_nodes, vehicle, G, edge_mapping)
        fast_dist, fast_time, fast_co2 = compute_stats(final_fastest_route_nodes, vehicle, G, edge_mapping)
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
    google_route_coords = []
    google_distance = 0.0
    google_duration = 0.0
    google_co2_estimated = 0.0
    
    api_key_google = os.getenv("API_KEY_GOOGLE")
    if not api_key_google:
        logger.error("Google Maps API Key is missing. Skipping Google route fetching.")
    else:
        try:
            google_client = googlemaps.Client(key=api_key_google)
            google_dir_response = google_client.directions(
                origin_coords,
                destination_coords,
                mode="driving",
                departure_time="now"
            )
            
            if not google_dir_response:
                raise ValueError("Google Maps API returned no directions.")
                
            google_poly = google_dir_response[0]["overview_polyline"]["points"]
            google_route_coords = polyline.decode(google_poly)
            
            google_leg = google_dir_response[0]["legs"][0]
            google_distance = google_leg["distance"]["value"] / 1000
            google_duration = google_leg.get("duration_in_traffic", google_leg["duration"])["value"] / 60
            
            google_co2_estimated = google_distance * (co2_map.get(vehicle, 180000) / 1_000_000)
            
            logger.info(f"Google Route Stats: Distance={google_distance:.2f}km, Time={google_duration:.1f}min, CO2={google_co2_estimated:.3f}kg")
            logger.info(f"Google Maps route fetched in {time.time() - google_route_start:.2f}s.")
        except Exception as e:
            logger.error(f"Error fetching Google Maps route: {e}", exc_info=True)
            logger.warning("Google Maps route data could not be fetched. Providing empty Google route details.")

    # --- Get Google ETA for Eco Route (if Google client is available) ---
    logger.info("Fetching Google ETA for the calculated eco-route...")
    google_eco_eta_start = time.time()
    eco_google_duration = eco_time
    if google_client and eco_coords:
        try:
            waypoints_for_google_eco = sample_waypoints(eco_coords)
            
            eco_dir_response = google_client.directions(
                origin=eco_coords[0],
                destination=eco_coords[-1],
                waypoints=waypoints_for_google_eco,
                mode="driving",
                departure_time="now"
            )
            
            if eco_dir_response:
                eco_google_duration = sum(
                    leg.get("duration_in_traffic", leg["duration"])["value"]
                    for leg in eco_dir_response[0]["legs"]
                ) / 60
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
            "time_minutes": round(eco_time, 1),
            "time_minutes_google_estimated": round(eco_google_duration, 1),
            "co2_kg": round(eco_co2, 3)
        },
        google_stats={
            "distance_km": round(google_distance, 2),
            "time_minutes": round(google_duration, 1),
            "co2_kg": round(google_co2_estimated, 3)
        },
        comparison={
            "co2_savings_kg": round(fast_co2 - eco_co2, 3),
            "co2_savings_percent": round((fast_co2 - eco_co2) / fast_co2 * 100, 1) if fast_co2 > 0 else 0,
            "time_difference_minutes": round(eco_google_duration - google_duration, 1)
        }
    )

    return response

async def calculate_route_streamed(request: RouteRequest) -> AsyncGenerator[str, None]:
    """
    Calculates routes and yields logs and the final result for SSE streaming.
    This is an async generator.
    """
    # Helper to format messages for SSE
    def sse_format(data: dict) -> str:
        return f"data: {json.dumps(data)}\n\n"

    try:
        origin_coords = request.origin
        destination_coords = request.destination
        vehicle = request.vehicle

        if vehicle not in co2_map:
            raise ValueError(f"Invalid vehicle type '{vehicle}'.")

        yield sse_format({"type": "log", "message": "Received request. Starting process..."})
        await asyncio.sleep(0.01) # <-- ADDED: Force stream flush

        # --- Graph Building ---
        yield sse_format({"type": "log", "message": "Building road network graph... (this may take a while)"})
        await asyncio.sleep(0.01)
        graph_build_start = time.time()
        G = build_optimized_road_network(origin_coords, destination_coords, chunk_size_km=25)
        yield sse_format({"type": "log", "message": f"Graph built in {time.time() - graph_build_start:.2f}s."})
        await asyncio.sleep(0.01)

        # --- Add Elevation Data (Example with yields inside the logic) ---
        yield sse_format({"type": "log", "message": "Adding elevation data..."})
        await asyncio.sleep(0.01)
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
            logger.warning("Continuing without full elevation data. Eco-route calculation may be less accurate.")        # ... just replace logger.info(...) with yield sse_format(...)
            yield sse_format({"type": "log", "message": "Elevation data added."})
            
            # --- Add Grades (slope information) ---
            yield sse_format({"type": "log", "message": "Calculating and adding edge grades..."})
            G = ox.add_edge_grades(G)
            yield sse_format({"type": "log", "message": "Edge grades added."})

            # --- Find Nearest Nodes ---
            yield sse_format({"type": "log", "message": "Finding nearest graph nodes..."})
            orig_node = ox.distance.nearest_nodes(G, X=origin_coords[1], Y=origin_coords[0])
            dest_node = ox.distance.nearest_nodes(G, X=destination_coords[1], Y=destination_coords[0])
            yield sse_format({"type": "log", "message": f"Nearest nodes found: {orig_node}, {dest_node}"})


        logger.info("Calculating and adding edge grades...")
        grades_start = time.time()
        try:
            G = ox.add_edge_grades(G)
            logger.info(f"Edge grades added in {time.time() - grades_start:.2f}s.")
        except Exception as e:
            logger.error(f"Error adding edge grades: {e}", exc_info=True)
            logger.warning("Continuing without edge grade data. Eco-route calculation may be less accurate.")
        yield sse_format({"type": "log", "message": "Edge grades added."})

        # --- Find Nearest Nodes ---
        logger.info("Finding nearest graph nodes for origin and destination...")
        nearest_nodes_start = time.time()
        try:
            orig_node = ox.distance.nearest_nodes(G, X=origin_coords[1], Y=origin_coords[0])
            dest_node = ox.distance.nearest_nodes(G, X=destination_coords[1], Y=destination_coords[0])
            if orig_node is None or dest_node is None:
                raise ValueError("Could not find nearest nodes for origin or destination.")
            logger.info(f"Nearest nodes found: Origin {orig_node}, Destination {dest_node} in {time.time() - nearest_nodes_start:.2f}s.")
            yield sse_format({"type": "log", "message": f"Nearest nodes found: Origin {orig_node}, Destination {dest_node}."})
        except Exception as e:
            logger.error(f"Failed to find nearest nodes: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail="Could not find reachable road network nodes for the provided origin or destination. Please ensure they are on or near a road.")
        
        yield sse_format({"type": "log", "message": "Nearest nodes found."})
        await asyncio.sleep(0.01)
        
    # --- Calculate Fastest Route (initial path for TomTom data scope) ---
        logger.info("Calculating initial fastest route...")
        fastest_route_calc_start = time.time()
        try:
            fastest_route_nodes = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
            fastest_route_edges_uv = set((u, v) for u, v in zip(fastest_route_nodes[:-1], fastest_route_nodes[1:]))
            logger.info(f"Initial fastest route calculated with {len(fastest_route_nodes)} nodes in {time.time() - fastest_route_calc_start:.2f}s.")
            yield sse_format({"type": "log", "message": f"Initial fastest route calculated with {len(fastest_route_nodes)} nodes."})
        except nx.NetworkXNoPath:
            logger.error("No path found between origin and destination nodes.")
            raise HTTPException(status_code=404, detail="No route found between the specified origin and destination.")
        except Exception as e:
            logger.error(f"Error calculating initial fastest route: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to calculate an initial route.")

        # --- Fetch and Apply TomTom Traffic Data ---
        logger.info("Fetching and applying TomTom traffic data...")
        yield sse_format({"type": "log", "message": "Fetching and applying TomTom traffic data..."})
        tomtom_fetch_start = time.time()
        try:
            global tomtom_cache
            coords_to_fetch = {}
            edge_tomtom_keys = [] 
            
            for u, v in fastest_route_edges_uv:
                for k in G[u][v]:
                    lat = (G.nodes[u]["y"] + G.nodes[v]["y"]) / 2
                    lon = (G.nodes[u]["x"] + G.nodes[v]["x"]) / 2
                    key = f"{lat:.5f},{lon:.5f}"
                    if key not in tomtom_cache:
                        coords_to_fetch[key] = (lat, lon)
                    edge_tomtom_keys.append((u, v, k, key))

            logger.info(f"Identified {len(coords_to_fetch)} unique points for TomTom fetching (from {len(edge_tomtom_keys)} edges).")
            yield sse_format({"type": "log", "message": f"Identified {len(coords_to_fetch)} unique points for TomTom fetching."})

            if coords_to_fetch:
                fetch_items = list(coords_to_fetch.items())
                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_key = {executor.submit(fetch_speed, item): item[0] for item in fetch_items}
                    for i, future in enumerate(as_completed(future_to_key)):
                        key, speed = future.result()
                        tomtom_cache[key] = speed
                        if i % 50 == 0 and i > 0:
                            logger.debug(f"Fetched {i}/{len(fetch_items)} TomTom speeds.")
                save_tomtom_cache()
                logger.info(f"Completed TomTom fetching for {len(coords_to_fetch)} points.")
            yield sse_format({"type": "log", "message": f"Completed TomTom fetching for {len(coords_to_fetch)} points."})

            speeds_applied_count = 0
            for u, v, k, key in edge_tomtom_keys:
                speed_kph = tomtom_cache.get(key)
                if speed_kph is not None and speed_kph > 0:
                    G[u][v][k]["speed_kph"] = speed_kph
                    length_m = safe_get(G[u][v][k], "length", 1.0)
                    G[u][v][k]["travel_time"] = length_m / (speed_kph * 1000 / 3600)
                    speeds_applied_count += 1
                else:
                    current_speed = G[u][v][k].get("speed_kph", 30.0)
                    length_m = safe_get(G[u][v][k], "length", 1.0)
                    G[u][v][k]["travel_time"] = length_m / (current_speed * 1000 / 3600)
            
            logger.info(f"Applied TomTom speeds (or fallbacks) to {speeds_applied_count}/{len(edge_tomtom_keys)} edges in {time.time() - tomtom_fetch_start:.2f}s.")
            yield sse_format({"type": "log", "message": f"Applied TomTom speeds to {speeds_applied_count} edges."})
            
        except Exception as e:
            logger.error(f"Error fetching or applying TomTom data: {e}", exc_info=True)
            logger.warning("Continuing route calculation without real-time TomTom traffic data.")

        # --- Calculate Turn Penalties and Add to Travel Time ---
        logger.info("Calculating turn penalties and applying to travel time...")
        yield sse_format({"type": "log", "message": "Calculating turn penalties and applying to travel time..."})
        penalty_calc_start = time.time()
        try:
            fastest_route_nodes = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
            
            nx.set_edge_attributes(G, 0, "turn_penalty")

            penalties_added_count = 0
            for i in range(1, len(fastest_route_nodes) - 1):
                u, v, w = fastest_route_nodes[i - 1], fastest_route_nodes[i], fastest_route_nodes[i + 1]
                
                b1 = calculate_bearing(G.nodes[u], G.nodes[v])
                b2 = calculate_bearing(G.nodes[v], G.nodes[w])
                
                delta = abs(b2 - b1)
                if delta > 180:
                    delta = 360 - delta

                penalty_seconds = 0
                if delta > 170:
                    penalty_seconds = 90
                elif delta > 150:
                    penalty_seconds = 60
                elif delta > 120:
                    penalty_seconds = 45
                elif delta > 90:
                    penalty_seconds = 30
                elif delta > 45:
                    penalty_seconds = 15
                else:
                    penalty_seconds = 5

                for k in G[u][v]:
                    G[u][v][k]["turn_penalty"] = safe_get(G[u][v][k], "turn_penalty") + penalty_seconds
                    G[u][v][k]["travel_time"] = safe_get(G[u][v][k], "travel_time") + penalty_seconds
                    penalties_added_count += 1
            
            logger.info(f"Added {penalties_added_count} turn penalties to graph edges in {time.time() - penalty_calc_start:.2f}s.")
            yield sse_format({"type": "log", "message": f"Added {penalties_added_count} turn penalties to graph edges."})
        except Exception as e:
            logger.error(f"Error calculating turn penalties: {e}", exc_info=True)
            logger.warning("Continuing without turn penalties. Fastest route may not accurately reflect turning costs.")

        # --- Calculate Eco Costs for all Edges ---
        logger.info("Calculating ecological costs for all graph edges...")
        yield sse_format({"type": "log", "message": "Calculating ecological costs for all graph edges..."})
        eco_cost_calc_start = time.time()
        try:
            eco_costs_data = {}
            for u, v, k, d in G.edges(keys=True, data=True):
                cost = base_eco_cost(u, v, d, vehicle, G) # Pass G directly for elevation lookup
                eco_costs_data[(u, v, k)] = cost
            
            nx.set_edge_attributes(G, eco_costs_data, "eco_cost")
            logger.info(f"Calculated eco costs for {len(eco_costs_data)} edges in {time.time() - eco_cost_calc_start:.2f}s.")
            yield sse_format({"type": "log", "message": f"Calculated eco costs for {len(eco_costs_data)} edges."})
        except Exception as e:
            logger.error(f"Error calculating eco costs: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to calculate ecological costs for road network.")

        # --- Create Simplified Graph for Eco Route Calculation ---
        logger.info("Creating a simplified graph for eco-route calculation...")
        yield sse_format({"type": "log", "message": "Creating a simplified graph for eco-route calculation..."})
        simplify_graph_start = time.time()
        try:
            G_simple = nx.DiGraph()
            edge_mapping = {}

            for u, v in G.edges():
                min_k = None
                min_eco_cost = float('inf')
                for k in G[u][v]:
                    current_eco_cost = G[u][v][k].get("eco_cost", float('inf'))
                    if current_eco_cost < min_eco_cost:
                        min_eco_cost = current_eco_cost
                        min_k = k
                
                if min_k is not None:
                    G_simple.add_edge(u, v, **G[u][v][min_k])
                    edge_mapping[(u, v)] = (u, v, min_k)
            
            logger.info(f"Simplified graph created with {len(G_simple.nodes)} nodes and {len(G_simple.edges)} edges in {time.time() - simplify_graph_start:.2f}s.")
            yield sse_format({"type": "log", "message": f"Simplified graph created with {len(G_simple.nodes)} nodes and {len(G_simple.edges)} edges."})
        except Exception as e:
            logger.error(f"Error creating simplified graph: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to simplify road network for eco-route calculation.")

        yield sse_format({"type": "log", "message": "Data processing complete. Calculating routes..."})
        await asyncio.sleep(0.01)

        # --- ALL ROUTE CALCULATION LOGIC GOES HERE ---
        # (This is a condensed representation of your existing code)
        orig_node = ox.distance.nearest_nodes(G, X=origin_coords[1], Y=origin_coords[0])
        dest_node = ox.distance.nearest_nodes(G, X=destination_coords[1], Y=destination_coords[0])

        # Calculate Eco Costs
        eco_costs_data = {}
        for u, v, k, d in G.edges(keys=True, data=True):
            cost = base_eco_cost(u, v, d, vehicle, G)
            eco_costs_data[(u, v, k)] = cost
        nx.set_edge_attributes(G, eco_costs_data, "eco_cost")
        yield sse_format({"type": "log", "message": f"Calculated eco costs for {len(eco_costs_data)} edges."})
        await asyncio.sleep(0.01)

        # Create Simplified Graph
        G_simple = nx.DiGraph()
        edge_mapping = {}
        for u, v in G.edges():
            min_k = min(G[u][v], key=lambda k: G[u][v][k].get("eco_cost", float('inf')))
            G_simple.add_edge(u, v, **G[u][v][min_k])
            edge_mapping[(u, v)] = (u, v, min_k)
        yield sse_format({"type": "log", "message": "Simplified graph for eco-routing."})
        await asyncio.sleep(0.01)

        # Calculate Eco Route
        eco_route_nodes = nx.shortest_path(G_simple, orig_node, dest_node, weight="eco_cost")
        eco_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in eco_route_nodes]
        yield sse_format({"type": "log", "message": "Eco-friendly route calculated."})
        await asyncio.sleep(0.01)

        # Calculate Fastest Route
        final_fastest_route_nodes = nx.shortest_path(G, orig_node, dest_node, weight="travel_time")
        
        # Calculate Stats
        eco_dist, eco_time, eco_co2 = compute_stats(eco_route_nodes, vehicle, G, edge_mapping)
        fast_dist, fast_time, fast_co2 = compute_stats(final_fastest_route_nodes, vehicle, G, edge_mapping)
        yield sse_format({"type": "log", "message": "Route statistics calculated."})
        await asyncio.sleep(0.01)
        
        # --- Get Google Route (for comparison) ---
        yield sse_format({"type": "log", "message": "Fetching Google Maps route for comparison..."})
        await asyncio.sleep(0.01)
        google_client = None
        google_route_coords = []
        google_distance = 0.0
        google_duration = 0.0
        google_co2_estimated = 0.0
        api_key_google = os.getenv("API_KEY_GOOGLE")

        if api_key_google:
            try:
                google_client = googlemaps.Client(key=api_key_google)
                # ... (rest of google directions logic)
                google_dir_response = google_client.directions(origin_coords, destination_coords, mode="driving", departure_time="now")
                if google_dir_response:
                    google_poly = google_dir_response[0]["overview_polyline"]["points"]
                    google_route_coords = polyline.decode(google_poly)
                    google_leg = google_dir_response[0]["legs"][0]
                    google_distance = google_leg["distance"]["value"] / 1000
                    google_duration = google_leg.get("duration_in_traffic", google_leg["duration"])["value"] / 60
                    google_co2_estimated = google_distance * (co2_map.get(vehicle, 180000) / 1_000_000)
                    yield sse_format({"type": "log", "message": "Google route fetched successfully."})
                    await asyncio.sleep(0.01)
            except Exception as e:
                yield sse_format({"type": "log", "message": f"Could not fetch Google route: {e}"})
                await asyncio.sleep(0.01)
        else:
            yield sse_format({"type": "log", "message": "Google API key not found. Skipping comparison."})
            await asyncio.sleep(0.01)

        # --- Get Google ETA for Eco Route ---
        eco_google_duration = eco_time # Default value
        if google_client and eco_coords:
            yield sse_format({"type": "log", "message": "Fetching Google ETA for eco-route..."})
            await asyncio.sleep(0.01)
            try:
                waypoints_for_google_eco = sample_waypoints(eco_coords)
                eco_dir_response = google_client.directions(
                    origin=eco_coords[0], destination=eco_coords[-1],
                    waypoints=waypoints_for_google_eco, mode="driving", departure_time="now"
                )
                if eco_dir_response:
                    eco_google_duration = sum(leg.get("duration_in_traffic", leg["duration"])["value"] for leg in eco_dir_response[0]["legs"]) / 60
                    yield sse_format({"type": "log", "message": f"Google ETA for eco route: {eco_google_duration:.1f}min"})
                    await asyncio.sleep(0.01)
            except Exception as e:
                yield sse_format({"type": "log", "message": "Could not get Google ETA for eco route."})
                await asyncio.sleep(0.01)

        # ####################################################################
        # ## MOVED THIS BLOCK: This now runs regardless of Google API status ##
        # ####################################################################
        yield sse_format({"type": "log", "message": "Finalizing results..."})
        await asyncio.sleep(0.01)
        
        # Ensure fast_co2 is not zero to avoid division by zero error
        co2_savings_percent = 0
        if fast_co2 and fast_co2 > 0:
            co2_savings_percent = round((fast_co2 - eco_co2) / fast_co2 * 100, 1)

        response_data = RouteResponse(
            eco_route=eco_coords,
            google_route=google_route_coords,
            eco_stats={
                "distance_km": round(eco_dist, 2),
                "time_minutes": round(eco_time, 1),
                "time_minutes_google_estimated": round(eco_google_duration, 1),
                "co2_kg": round(eco_co2, 3)
            },
            google_stats={
                "distance_km": round(google_distance, 2),
                "time_minutes": round(google_duration, 1),
                "co2_kg": round(google_co2_estimated, 3)
            },
            comparison={
                "co2_savings_kg": round(fast_co2 - eco_co2, 3),
                "co2_savings_percent": co2_savings_percent,
                "time_difference_minutes": round(eco_google_duration - google_duration, 1)
            }
        ).model_dump()

        # Yield the final result
        yield sse_format({"type": "result", "data": response_data})
        yield sse_format({"type": "log", "message": "Process complete."})

    except Exception as e:
        logger.error(f"Error during route streaming: {e}", exc_info=True)
        error_message = f"A critical error occurred: {e}"
        yield sse_format({"type": "error", "message": error_message})