from typing import List, Tuple
from pydantic import BaseModel

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