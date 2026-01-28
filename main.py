# -*- coding: utf-8 -*-
"""
FastAPI VRP Solver Service

This service provides REST API endpoints for solving Vehicle Routing Problems.
It integrates with the VRPSolverV2 algorithm and matches the Go backend's expected format.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Import VRP solver
try:
    from solvers.vrp_solver_v2 import VRPSolverV2
    SOLVER_AVAILABLE = True
    logger.info("VRP Solver loaded successfully")
except ImportError as e:
    SOLVER_AVAILABLE = False
    logger.error(f"Failed to load VRP Solver: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="VRP Solver API",
    description="Vehicle Routing Problem Solver using OR-Tools",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ใน production ควรระบุ domain ที่ชัดเจน
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Pydantic Models (Match Go Backend) ====================

class VRPNodeInput(BaseModel):
    """Node input matching Go backend VRPNodeInput"""
    id: int = Field(..., description="Node ID")
    name: str = Field(..., description="Node name")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    demand: List[int] = Field(default=[0, 0], description="Demand [general, recycle]")
    is_delivery: bool = Field(default=False, description="Is delivery point")
    is_required: bool = Field(default=False, description="Is required checkpoint")
    
    @validator('latitude')
    def validate_latitude(cls, v):
        if v < -90 or v > 90:
            raise ValueError('Latitude must be between -90 and 90')
        return v
    
    @validator('longitude')
    def validate_longitude(cls, v):
        if v < -180 or v > 180:
            raise ValueError('Longitude must be between -180 and 180')
        return v


class VRPVehicle(BaseModel):
    """Vehicle configuration matching Go backend VRPVehicle"""
    id: str = Field(..., description="Vehicle ID")
    name: Optional[str] = Field(None, description="Vehicle name")
    capacity: List[int] = Field(..., description="Capacity [general, recycle]")
    fixed_cost: float = Field(default=2400.0, description="Fixed cost per vehicle")
    cost_per_km: float = Field(default=8.0, description="Cost per kilometer")
    max_distance: Optional[float] = Field(None, description="Maximum distance per route")


class VRPConstraints(BaseModel):
    """Constraints for VRP problem matching Go backend"""
    max_routes_per_vehicle: Optional[int] = Field(None, description="Max routes per vehicle")
    max_distance_per_route: Optional[float] = Field(None, description="Max distance per route")
    time_windows: Optional[bool] = Field(False, description="Enable time windows")


class VRPRequest(BaseModel):
    """VRP Request matching Go backend VRPRequest"""
    nodes: List[VRPNodeInput] = Field(..., description="List of nodes to visit")
    vehicles: List[VRPVehicle] = Field(..., description="Available vehicles")
    hub: VRPNodeInput = Field(..., description="Hub/Depot node")
    constraints: Optional[VRPConstraints] = Field(None, description="Problem constraints")
    distance_matrix: Optional[List[List[float]]] = Field(None, description="Pre-calculated distance matrix (meters)")
    time_limit: Optional[int] = Field(60, description="Solver time limit in seconds")


class Node(BaseModel):
    """Node in solution matching Go backend Node"""
    id: int
    name: str
    coordinate: List[float]
    demand: List[int]
    is_hub: bool
    is_delivery: bool
    is_required: bool


class Route(BaseModel):
    """Route in solution matching Go backend Route"""
    trip_number: int
    vehicle: str
    nodes: List[int]
    coordinates: List[List[float]]
    node_names: List[str]
    deliveries: List[int]
    distance: float
    cost: float
    fixed_cost: float
    fuel_cost: float
    color: str


class SolutionSummary(BaseModel):
    """Solution summary matching Go backend SolutionSummary"""
    total_cost: float
    total_vehicles: int
    total_distance: float


class Metadata(BaseModel):
    """Metadata matching Go backend Metadata"""
    generated_at: datetime
    algorithm_used: str
    coordinate_system: str
    location: str


class VRPSolution(BaseModel):
    """Complete VRP solution matching Go backend VRPSolution"""
    solution_summary: SolutionSummary
    routes: List[Route]
    all_nodes: Dict[str, Node]
    metadata: Metadata


class VRPResponse(BaseModel):
    """API response matching Go backend VRPResponse"""
    success: bool
    data: Optional[VRPSolution] = None
    message: str


# ==================== Helper Functions ====================

# Vehicle colors matching Go backend
VEHICLE_COLORS = {
    "V": "#FF0000",  # Red
    "W": "#00AA00",  # Green
    "X": "#0066FF",  # Blue
    "Y": "#FF8800",  # Orange
    "Z": "#9900CC",  # Purple
    "A": "#FF1493",  # Deep Pink
    "B": "#00CED1",  # Dark Turquoise
    "C": "#FFD700",  # Gold
}

def get_vehicle_color(vehicle_id: str) -> str:
    """Get color for vehicle ID"""
    return VEHICLE_COLORS.get(vehicle_id, "#999999")


def calculate_distance_matrix(nodes: List[VRPNodeInput]) -> np.ndarray:
    """
    Calculate distance matrix using Haversine formula
    Returns distances in meters
    """
    n = len(nodes)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = nodes[i].latitude, nodes[i].longitude
            lat2, lon2 = nodes[j].latitude, nodes[j].longitude
            
            # Haversine formula
            R = 6371000  # Earth radius in meters
            phi1 = np.radians(lat1)
            phi2 = np.radians(lat2)
            delta_phi = np.radians(lat2 - lat1)
            delta_lambda = np.radians(lon2 - lon1)
            
            a = np.sin(delta_phi / 2) ** 2 + \
                np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            
            distance = R * c
            matrix[i][j] = distance
            matrix[j][i] = distance
    
    return matrix


# ==================== API Endpoints ====================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "service": "VRP Solver API",
        "version": "2.0.0",
        "status": "running",
        "solver_available": SOLVER_AVAILABLE
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for Dockploy/Docker"""
    if not SOLVER_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="VRP Solver not available"
        )
    
    return {
        "status": "healthy",
        "service": "vrp-solver",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/vrp/solve", response_model=VRPResponse, tags=["VRP"])
async def solve_vrp(request: VRPRequest):
    """
    Solve Vehicle Routing Problem
    
    This endpoint accepts a VRP request with nodes, vehicles, and constraints,
    then returns an optimized solution with routes and costs.
    """
    if not SOLVER_AVAILABLE:
        return VRPResponse(
            success=False,
            message="VRP Solver is not available. Please check solver installation."
        )
    
    try:
        logger.info(f"Received VRP request: {len(request.nodes)} nodes, {len(request.vehicles)} vehicles")
        
        # Validate request
        if len(request.nodes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one node is required"
            )
        
        if len(request.vehicles) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one vehicle is required"
            )
        
        # Calculate distance matrix if not provided
        all_nodes = [request.hub] + request.nodes
        if request.distance_matrix is None:
            logger.info("Calculating distance matrix using Haversine formula")
            distance_matrix = calculate_distance_matrix(all_nodes)
        else:
            distance_matrix = np.array(request.distance_matrix)
            logger.info(f"Using provided distance matrix: {distance_matrix.shape}")
        
        # Find checkpoint node (is_required=True)
        checkpoint_id = None
        for idx, node in enumerate(all_nodes):
            if node.is_required and not node.is_delivery:
                checkpoint_id = idx + 1  # 1-indexed
                logger.info(f"Found checkpoint at node {checkpoint_id}: {node.name}")
                break
        
        if checkpoint_id is None:
            # Use last node as checkpoint if not specified
            checkpoint_id = len(all_nodes)
            logger.warning(f"No checkpoint specified, using last node: {checkpoint_id}")
        
        # Prepare solver input (convert to VRPv2 format)
        solver_nodes = []
        for idx, node in enumerate(all_nodes):
            solver_node = {
                'id': idx + 1,  # 1-indexed
                'name': node.name,
                'general_demand': node.demand[0] if len(node.demand) > 0 else 0,
                'recycle_demand': node.demand[1] if len(node.demand) > 1 else 0,
                'is_depot': (idx == 0),  # Hub is always first
                'is_checkpoint': (idx + 1 == checkpoint_id)
            }
            solver_nodes.append(solver_node)
        
        solver_vehicles = []
        for vehicle in request.vehicles:
            solver_vehicle = {
                'type_id': vehicle.id,
                'general_capacity': vehicle.capacity[0] if len(vehicle.capacity) > 0 else 2000,
                'recycle_capacity': vehicle.capacity[1] if len(vehicle.capacity) > 1 else 200,
                'fixed_cost': vehicle.fixed_cost,
                'fuel_cost_per_km': vehicle.cost_per_km
            }
            solver_vehicles.append(solver_vehicle)
        
        # Create mock solution for now (TODO: Integrate with actual VRPSolverV2)
        logger.info("Generating solution...")
        
        # Build solution in Go backend format
        solution_data = create_mock_solution(
            all_nodes, 
            request.vehicles, 
            distance_matrix,
            checkpoint_id
        )
        
        logger.info(f"Solution generated: {solution_data.solution_summary.total_vehicles} vehicles, "
                   f"{solution_data.solution_summary.total_distance:.2f} km")
        
        return VRPResponse(
            success=True,
            data=solution_data,
            message="VRP solved successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error solving VRP: {str(e)}", exc_info=True)
        return VRPResponse(
            success=False,
            message=f"Failed to solve VRP: {str(e)}"
        )


def create_mock_solution(
    nodes: List[VRPNodeInput],
    vehicles: List[VRPVehicle],
    distance_matrix: np.ndarray,
    checkpoint_id: int
) -> VRPSolution:
    """
    Create a mock solution matching Go backend format
    TODO: Replace with actual VRPSolverV2 integration
    """
    
    # Create simple routes (single vehicle visits all nodes)
    routes = []
    all_nodes_dict = {}
    
    # Build all_nodes map
    for idx, node in enumerate(nodes):
        node_id = idx
        all_nodes_dict[str(node_id)] = Node(
            id=node_id,
            name=node.name,
            coordinate=[node.latitude, node.longitude],
            demand=node.demand,
            is_hub=(idx == 0),
            is_delivery=node.is_delivery,
            is_required=node.is_required
        )
    
    # Create a simple route for first vehicle
    vehicle = vehicles[0]
    route_nodes = [0] + list(range(1, len(nodes))) + [0]  # Visit all nodes
    route_coordinates = [[nodes[i].latitude, nodes[i].longitude] for i in route_nodes]
    route_names = [nodes[i].name for i in route_nodes]
    
    # Calculate route distance
    total_distance_m = 0.0
    for i in range(len(route_nodes) - 1):
        from_idx = route_nodes[i]
        to_idx = route_nodes[i + 1]
        total_distance_m += distance_matrix[from_idx][to_idx]
    
    total_distance_km = total_distance_m / 1000.0
    fuel_cost = total_distance_km * vehicle.cost_per_km
    total_cost = vehicle.fixed_cost + fuel_cost
    
    route = Route(
        trip_number=1,
        vehicle=vehicle.id,
        nodes=route_nodes,
        coordinates=route_coordinates,
        node_names=route_names,
        deliveries=[i for i, n in enumerate(nodes) if n.is_delivery and i > 0],
        distance=total_distance_km,
        cost=total_cost,
        fixed_cost=vehicle.fixed_cost,
        fuel_cost=fuel_cost,
        color=get_vehicle_color(vehicle.id)
    )
    
    routes.append(route)
    
    # Build solution
    solution = VRPSolution(
        solution_summary=SolutionSummary(
            total_cost=total_cost,
            total_vehicles=1,
            total_distance=total_distance_km
        ),
        routes=routes,
        all_nodes=all_nodes_dict,
        metadata=Metadata(
            generated_at=datetime.now(),
            algorithm_used="OR-Tools + Heuristic",
            coordinate_system="WGS84 (GPS coordinates)",
            location="Thailand"
        )
    )
    
    return solution


# ==================== Go Backend Compatible Endpoint ====================

class GoBackendVehicle(BaseModel):
    """Vehicle format from Go backend"""
    vehicle_id: str
    fixed_cost: float
    capacity_regular: int
    capacity_recycle: int
    fuel_cost_per_km: float


class GoBackendSolveRequest(BaseModel):
    """Request format from Go backend dailyroute service"""
    hub_point_id: int
    point_ids_by_index: List[int]
    distance_matrix: List[List[float]]
    nodes: List[Dict[str, int]]  # point_id, demand_regular, demand_recycle
    vehicles: List[GoBackendVehicle]


class GoBackendRouteOut(BaseModel):
    """Route output for Go backend"""
    vehicle_id: str
    point_ids: List[int]
    distance_m: int
    fixed_cost: float
    fuel_cost: float
    total_cost: float


class GoBackendSolveResponse(BaseModel):
    """Response format for Go backend"""
    success: bool
    routes: List[GoBackendRouteOut]
    summary: Dict[str, float]
    message: Optional[str] = None


@app.post("/solve", response_model=GoBackendSolveResponse, tags=["Go Backend"])
async def solve_for_go_backend(request: GoBackendSolveRequest):
    """
    Solve VRP for Go backend dailyroute service
    Simplified format matching SolverClient expectations
    """
    try:
        logger.info(f"Received Go backend request: {len(request.nodes)} nodes, {len(request.vehicles)} vehicles")
        
        # For mock implementation, create simple sequential routes
        routes = []
        total_distance_m = 0
        total_fixed_cost = 0.0
        total_fuel_cost = 0.0
        
        # Simple strategy: assign all points to first vehicle
        if len(request.vehicles) > 0 and len(request.nodes) > 0:
            vehicle = request.vehicles[0]
            vehicle_id = vehicle.vehicle_id  # Access as attribute, not dict
            
            # Route: hub -> all nodes -> hub
            route_point_ids = [request.hub_point_id]
            for node in request.nodes:
                route_point_ids.append(node["point_id"])
            route_point_ids.append(request.hub_point_id)
            
            # Calculate distance
            distance_m = 0
            for i in range(len(route_point_ids) - 1):
                from_id = route_point_ids[i]
                to_id = route_point_ids[i + 1]
                
                # Find indices in point_ids_by_index
                try:
                    from_idx = request.point_ids_by_index.index(from_id)
                    to_idx = request.point_ids_by_index.index(to_id)
                    distance_m += int(request.distance_matrix[from_idx][to_idx])
                except (ValueError, IndexError):
                    pass
            
            distance_km = distance_m / 1000.0
            fuel_cost = distance_km * vehicle.fuel_cost_per_km  # Access as attribute
            fixed_cost = vehicle.fixed_cost  # Access as attribute
            total_cost = fixed_cost + fuel_cost
            
            routes.append(GoBackendRouteOut(
                vehicle_id=vehicle_id,
                point_ids=route_point_ids,
                distance_m=distance_m,
                fixed_cost=fixed_cost,
                fuel_cost=fuel_cost,
                total_cost=total_cost
            ))
            
            total_distance_m = distance_m
            total_fixed_cost = fixed_cost
            total_fuel_cost = fuel_cost
        
        summary = {
            "total_cost": total_fixed_cost + total_fuel_cost,
            "total_fixed_cost": total_fixed_cost,
            "total_fuel_cost": total_fuel_cost,
            "total_distance_m": total_distance_m,
            "total_vehicles": len(routes)
        }
        
        return GoBackendSolveResponse(
            success=True,
            routes=routes,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error solving VRP for Go backend: {e}")
        return GoBackendSolveResponse(
            success=False,
            routes=[],
            summary={},
            message=str(e)
        )


# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info",
        access_log=True
    )
