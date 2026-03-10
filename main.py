# -*- coding: utf-8 -*-
"""
FastAPI VRP Solver Service

This service provides REST API endpoints for solving Vehicle Routing Problems.
It integrates with the VRPSolverV2 algorithm and matches the Go backend's expected format.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Optional, Dict
from datetime import datetime
import numpy as np
import asyncio
import time
import logging
import sys
import threading
from importlib import import_module
from typing import Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Serialize VRP solves to avoid parallel execution and control memory usage.
# This queues overlapping solve requests instead of running them concurrently.
_SOLVE_LOCK = asyncio.Lock()

# -------------------- Lazy solver import (operational) --------------------
#
# We intentionally avoid importing the VRP solver module at process startup.
# The solver module pulls in heavy dependencies (e.g., OR-Tools, pandas, openpyxl)
# which can increase startup CPU/memory and slow container scale-up.
#
# Trade-off: the *first* solve request (or /ready or /warmup) will pay the import
# cost, so it may be slower right after deploy.
_SOLVER_IMPORT_LOCK = threading.Lock()
_SOLVER_SYMBOLS: Tuple[Any, Any, Any, Any] | None = None
_SOLVER_IMPORT_ATTEMPTED = False
_SOLVER_IMPORT_ERROR: str | None = None


def _get_solver_symbols() -> Tuple[Any, Any, Any, Any]:
    """Lazily import and cache VRP solver symbols.

    Returns:
        (VRPSolverV2, Node, Vehicle, Solution)

    Notes:
        - Imports at most once per process.
        - On failure, caches the error message to make readiness checks cheap.
    """
    global _SOLVER_SYMBOLS, _SOLVER_IMPORT_ATTEMPTED, _SOLVER_IMPORT_ERROR

    if _SOLVER_SYMBOLS is not None:
        return _SOLVER_SYMBOLS

    with _SOLVER_IMPORT_LOCK:
        if _SOLVER_SYMBOLS is not None:
            return _SOLVER_SYMBOLS
        if _SOLVER_IMPORT_ATTEMPTED and _SOLVER_IMPORT_ERROR is not None:
            raise ImportError(_SOLVER_IMPORT_ERROR)

        _SOLVER_IMPORT_ATTEMPTED = True
        try:
            mod = import_module("solvers.vrp_solver_v2")
            _SOLVER_SYMBOLS = (mod.VRPSolverV2, mod.Node, mod.Vehicle, mod.Solution)
            _SOLVER_IMPORT_ERROR = None
            logger.info("VRP solver module imported lazily")
            return _SOLVER_SYMBOLS
        except Exception as e:
            _SOLVER_IMPORT_ERROR = str(e)
            logger.error("Failed to lazily import VRP solver: %s", e, exc_info=True)
            raise


def _solver_status_snapshot() -> Dict[str, Any]:
    """Return solver import status without importing it."""
    return {
        "import_attempted": _SOLVER_IMPORT_ATTEMPTED,
        "imported": _SOLVER_SYMBOLS is not None,
        "last_error": _SOLVER_IMPORT_ERROR,
    }

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
    # Do not import the solver here; keep this endpoint cheap.
    solver_status = _solver_status_snapshot()
    return {
        "service": "VRP Solver API",
        "version": "2.0.0",
        "status": "running",
        # With lazy loading, this means "solver already imported in this process".
        # Use /ready to actively verify availability.
        "solver_available": solver_status["imported"],
        "solver_lazy": True,
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Liveness check (cheap).

    Semantics:
        - MUST be fast and always return 200 if the process is alive.
        - MUST NOT import/touch the solver or any external dependencies.
    """
    return {
        "status": "alive",
        "service": "vrp-solver",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check.

    This endpoint *does* perform the lazy solver import to verify the process is
    ready to serve solve requests.
    """
    try:
        _get_solver_symbols()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"VRP Solver not ready: {e}",
        )

    return {
        "status": "ready",
        "service": "vrp-solver",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/warmup", tags=["Health"])
async def warmup_solver():
    """Optional warm-up endpoint.

    Use this to pre-load the solver after deploy so the first solve request
    doesn't pay the import cost.
    """
    try:
        _get_solver_symbols()
        return {"success": True, "message": "Solver warmed up"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Warm-up failed: {e}",
        )


@app.post("/api/vrp/solve", response_model=VRPResponse, tags=["VRP"])
async def solve_vrp(request: VRPRequest):
    """
    Solve Vehicle Routing Problem
    
    This endpoint accepts a VRP request with nodes, vehicles, and constraints,
    then returns an optimized solution with routes and costs.
    """
    # Lazy-load heavy solver module only when we actually handle a solve request.
    try:
        _, SolverNode, SolverVehicle, _ = _get_solver_symbols()
    except Exception:
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
        # CRITICAL: VRP solver expects depot at index 0 (Node 1)
        # Reorder nodes: [hub] + [other nodes]
        all_nodes = [request.hub] + request.nodes
        
        if request.distance_matrix is None:
            logger.info("Calculating distance matrix using Haversine formula")
            distance_matrix = calculate_distance_matrix(all_nodes)
        else:
            distance_matrix = np.array(request.distance_matrix)
            logger.info(f"Using provided distance matrix: {distance_matrix.shape}")
        
        # Find checkpoint node (is_required=True OR is_delivery=True)
        # Checkpoint is typically the dump point where trucks unload before returning
        checkpoint_id = None
        for idx, node in enumerate(all_nodes):
            # Skip depot (first node)
            if idx == 0:
                continue
            # Find delivery/required point (usually the dump point)
            if node.is_delivery or (node.is_required and len(node.demand) == 2 and node.demand[0] == 0 and node.demand[1] == 0):
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
            solver_node = SolverNode(
                id=idx + 1,  # 1-indexed
                name=node.name,
                general_demand=float(node.demand[0] if len(node.demand) > 0 else 0),
                recycle_demand=float(node.demand[1] if len(node.demand) > 1 else 0),
                is_depot=(idx == 0),  # Hub is always first
                is_checkpoint=(idx + 1 == checkpoint_id)
            )
            solver_nodes.append(solver_node)
        
        solver_vehicles = []
        for vehicle in request.vehicles:
            solver_vehicle = SolverVehicle(
                type_id=vehicle.id,
                general_capacity=float(vehicle.capacity[0] if len(vehicle.capacity) > 0 else 2000),
                recycle_capacity=float(vehicle.capacity[1] if len(vehicle.capacity) > 1 else 200),
                fixed_cost=float(vehicle.fixed_cost),
                fuel_cost_per_km=float(vehicle.cost_per_km)
            )
            solver_vehicles.append(solver_vehicle)
        
        # Call actual VRPSolverV2 (single in-flight solve at a time)
        logger.info("Solving VRP using OR-Tools...")
        wait_start = time.monotonic()
        async with _SOLVE_LOCK:
            waited_s = time.monotonic() - wait_start
            if waited_s >= 0.25:
                logger.info(f"Solve request waited {waited_s:.2f}s for the solver lock")

            solver_solution = await run_in_threadpool(
                solve_vrp_internal,
                solver_nodes,
                solver_vehicles,
                distance_matrix,
                checkpoint_id,
                time_limit=request.time_limit or 60,
            )
        
        # Build solution in Go backend format
        solution_data = convert_solver_solution_to_api_format(
            solver_solution,
            all_nodes,
            request.vehicles
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


def solve_vrp_internal(
    nodes: List[Any],
    vehicles: List[Any],
    distance_matrix: np.ndarray,
    checkpoint_id: int,
    time_limit: int = 60
) -> Any:
    """
    Solve VRP using VRPSolverV2 internal algorithm (with checkpoint)
    """
    VRPSolverV2, _, _, _ = _get_solver_symbols()
    solver = VRPSolverV2.__new__(VRPSolverV2)
    
    # Use best vehicle (lowest fuel cost)
    best_vehicle = min(vehicles, key=lambda v: v.fuel_cost_per_km)
    
    # Calculate minimum vehicles needed
    total_general = sum(n.general_demand for n in nodes)
    total_recycle = sum(n.recycle_demand for n in nodes)
    
    import math
    min_veh_gen = math.ceil(total_general / best_vehicle.general_capacity) if best_vehicle.general_capacity > 0 else 1
    min_veh_rec = math.ceil(total_recycle / best_vehicle.recycle_capacity) if best_vehicle.recycle_capacity > 0 else 1
    num_vehicles = max(min_veh_gen, min_veh_rec, 1)
    
    depot_idx = 0  # Node 1 is always depot
    checkpoint_idx = checkpoint_id - 1  # Convert to 0-indexed
    
    logger.info(f"Solving with {num_vehicles} vehicles, checkpoint at node {checkpoint_id}")
    
    # Try OR-Tools solver
    try:
        solution = solver._solve_ortools(
            nodes, best_vehicle, distance_matrix,
            depot_idx, checkpoint_idx, num_vehicles, time_limit
        )
    except Exception as e:
        logger.warning(f"OR-Tools solver failed: {e}, falling back to heuristic")
        solution = solver._solve_heuristic(
            nodes, best_vehicle, distance_matrix,
            depot_idx, checkpoint_idx
        )
    
    # Validate solution
    solution = solver._validate_solution(solution, nodes, checkpoint_idx)
    
    return solution


def solve_vrp_without_checkpoint(
    nodes: List[Any],
    vehicle: Any,
    distance_matrix: np.ndarray,
    depot_idx: int,
    num_vehicles: int,
    time_limit: int = 60
) -> Any:
    """
    Solve VRP without checkpoint requirement (for Go backend)
    Routes: depot → collection nodes → depot
    """
    try:
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    except ImportError:
        raise Exception("OR-Tools not available")
    
    num_nodes = len(nodes)
    
    # OR-Tools setup
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)
    
    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Demand callbacks
    general_demands = [int(n.general_demand) for n in nodes]
    recycle_demands = [int(n.recycle_demand) for n in nodes]
    
    def demand_general_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return general_demands[from_node]
    
    def demand_recycle_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return recycle_demands[from_node]
    
    # Add capacity dimensions
    demand_gen_callback_index = routing.RegisterUnaryTransitCallback(demand_general_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_gen_callback_index,
        0,  # no slack
        [int(vehicle.general_capacity)] * num_vehicles,
        True,  # start cumul to zero
        'GeneralCapacity'
    )
    
    demand_rec_callback_index = routing.RegisterUnaryTransitCallback(demand_recycle_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_rec_callback_index,
        0,
        [int(vehicle.recycle_capacity)] * num_vehicles,
        True,
        'RecycleCapacity'
    )
    
    # Search parameters
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = time_limit
    
    # Solve
    assignment = routing.SolveWithParameters(search_params)
    
    if not assignment:
        raise Exception("OR-Tools could not find a solution")
    
    # Extract solution
    from solvers.vrp_solver_v2 import Route, Solution
    
    routes = []
    total_distance_m = 0
    
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route_nodes = []
        general_load = 0
        recycle_load = 0
        
        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            node = nodes[node_idx]
            
            route_nodes.append(node.id)  # 1-indexed
            general_load += node.general_demand
            recycle_load += node.recycle_demand
            
            index = assignment.Value(routing.NextVar(index))
        
        # Add final depot
        route_nodes.append(nodes[depot_idx].id)
        
        # Only process non-trivial routes
        if len(route_nodes) > 2:  # More than just depot -> depot
            # Calculate distance
            route_distance = 0.0
            for i in range(len(route_nodes) - 1):
                from_idx = route_nodes[i] - 1  # Convert to 0-indexed
                to_idx = route_nodes[i + 1] - 1
                route_distance += distance_matrix[from_idx][to_idx]
            
            distance_km = route_distance / 1000.0
            fuel_cost = distance_km * vehicle.fuel_cost_per_km
            
            route = Route(
                vehicle_id=vehicle_id + 1,
                vehicle_type=vehicle.type_id,
                nodes=route_nodes,  # [depot, ..., depot] with 1-indexed IDs
                distance_meters=route_distance,
                distance_km=distance_km,
                general_load=general_load,
                recycle_load=recycle_load,
                fixed_cost=vehicle.fixed_cost,
                fuel_cost=fuel_cost,
                total_cost=vehicle.fixed_cost + fuel_cost
            )
            routes.append(route)
            total_distance_m += route_distance
    
    total_distance_km = total_distance_m / 1000.0
    total_fixed = sum(r.fixed_cost for r in routes)
    total_fuel = sum(r.fuel_cost for r in routes)
    
    return Solution(
        status='OPTIMAL',
        routes=routes,
        num_vehicles_used=len(routes),
        total_distance_meters=total_distance_m,
        total_distance_km=total_distance_km,
        total_fixed_cost=total_fixed,
        total_fuel_cost=total_fuel,
        total_cost=total_fixed + total_fuel,
        all_nodes_visited=True,
        all_routes_valid=True,
        validation_errors=[]
    )


def solve_vrp_without_checkpoint_multi(
    nodes: List[Any],
    vehicles: List[Any],
    distance_matrix: np.ndarray,
    depot_idx: int,
    time_limit: int = 60,
) -> Any:
    """
    Solve VRP without checkpoint requirement (for Go backend), using distinct vehicles.

    Key property:
    - Each OR-Tools vehicle index maps to exactly one real vehicle (no reuse across routes).
    - Unused vehicles will have a trivial route (depot -> depot) and are filtered out.
    """
    try:
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
    except ImportError:
        raise Exception("OR-Tools not available")

    if not vehicles:
        raise Exception("At least one vehicle is required")

    num_nodes = len(nodes)
    num_vehicles = len(vehicles)

    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback (optimize distance only; keep existing objective behavior)
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    general_demands = [int(n.general_demand) for n in nodes]
    recycle_demands = [int(n.recycle_demand) for n in nodes]

    def demand_general_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return general_demands[from_node]

    def demand_recycle_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return recycle_demands[from_node]

    demand_gen_callback_index = routing.RegisterUnaryTransitCallback(demand_general_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_gen_callback_index,
        0,
        [int(v.general_capacity) for v in vehicles],
        True,
        'GeneralCapacity'
    )

    demand_rec_callback_index = routing.RegisterUnaryTransitCallback(demand_recycle_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_rec_callback_index,
        0,
        [int(v.recycle_capacity) for v in vehicles],
        True,
        'RecycleCapacity'
    )

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = time_limit

    assignment = routing.SolveWithParameters(search_params)
    if not assignment:
        raise Exception("OR-Tools could not find a solution")

    from solvers.vrp_solver_v2 import Route, Solution

    routes = []
    total_distance_m = 0.0

    for vehicle_idx in range(num_vehicles):
        index = routing.Start(vehicle_idx)
        route_nodes = []
        general_load = 0.0
        recycle_load = 0.0

        while not routing.IsEnd(index):
            node_idx = manager.IndexToNode(index)
            node = nodes[node_idx]

            route_nodes.append(node.id)
            general_load += float(node.general_demand)
            recycle_load += float(node.recycle_demand)

            index = assignment.Value(routing.NextVar(index))

        # Add final depot
        route_nodes.append(nodes[depot_idx].id)

        # Only process non-trivial routes
        if len(route_nodes) > 2:
            route_distance = 0.0
            for i in range(len(route_nodes) - 1):
                from_idx = route_nodes[i] - 1
                to_idx = route_nodes[i + 1] - 1
                route_distance += float(distance_matrix[from_idx][to_idx])

            vehicle = vehicles[vehicle_idx]
            distance_km = route_distance / 1000.0
            fuel_cost = distance_km * float(vehicle.fuel_cost_per_km)

            route = Route(
                vehicle_id=vehicle_idx + 1,
                vehicle_type=vehicle.type_id,
                nodes=route_nodes,
                distance_meters=route_distance,
                distance_km=distance_km,
                general_load=general_load,
                recycle_load=recycle_load,
                fixed_cost=float(vehicle.fixed_cost),
                fuel_cost=float(fuel_cost),
                total_cost=float(vehicle.fixed_cost) + float(fuel_cost),
            )
            routes.append(route)
            total_distance_m += route_distance

    total_distance_km = total_distance_m / 1000.0
    total_fixed = sum(float(r.fixed_cost) for r in routes)
    total_fuel = sum(float(r.fuel_cost) for r in routes)

    return Solution(
        status='OPTIMAL',
        routes=routes,
        num_vehicles_used=len(routes),
        total_distance_meters=total_distance_m,
        total_distance_km=total_distance_km,
        total_fixed_cost=total_fixed,
        total_fuel_cost=total_fuel,
        total_cost=total_fixed + total_fuel,
        all_nodes_visited=True,
        all_routes_valid=True,
        validation_errors=[]
    )


def convert_solver_solution_to_api_format(
    solver_solution: Any,
    nodes: List[VRPNodeInput],
    vehicles: List[VRPVehicle]
) -> VRPSolution:
    """
    Convert VRPSolverV2 solution to API response format
    """
    routes = []
    all_nodes_dict = {}
    
    # Build all_nodes map (use 1-indexed node IDs to match solver output)
    for idx, node in enumerate(nodes):
        node_id = idx + 1  # 1-indexed to match solver
        all_nodes_dict[str(node_id)] = Node(
            id=node_id,
            name=node.name,
            coordinate=[node.latitude, node.longitude],
            demand=node.demand,
            is_hub=(idx == 0),  # First node is hub
            is_delivery=node.is_delivery,
            is_required=node.is_required
        )
    
    # Convert each route from solver format
    for solver_route in solver_solution.routes:
        # Solver returns 1-indexed node IDs: [1,2,3,...,20,1]
        # Keep them as-is (do NOT subtract 1)
        route_nodes = solver_route.nodes  # Already 1-indexed
        
        # Build coordinates and names using 0-indexed access to nodes array
        route_coordinates = [[nodes[nid - 1].latitude, nodes[nid - 1].longitude] for nid in route_nodes]
        route_names = [nodes[nid - 1].name for nid in route_nodes]
        
        # Find vehicle
        vehicle = next((v for v in vehicles if v.id == solver_route.vehicle_type), vehicles[0])
        
        route = Route(
            trip_number=solver_route.vehicle_id,
            vehicle=solver_route.vehicle_type,
            nodes=route_nodes,  # 1-indexed: [1, 2, ..., 20, 1]
            coordinates=route_coordinates,
            node_names=route_names,
            deliveries=[nid for nid in route_nodes if nodes[nid - 1].is_delivery and nid > 1],
            distance=solver_route.distance_km,
            cost=solver_route.total_cost,
            fixed_cost=solver_route.fixed_cost,
            fuel_cost=solver_route.fuel_cost,
            color=get_vehicle_color(solver_route.vehicle_type)
        )
        routes.append(route)
    
    # Build solution
    solution = VRPSolution(
        solution_summary=SolutionSummary(
            total_cost=solver_solution.total_cost,
            total_vehicles=solver_solution.num_vehicles_used,
            total_distance=solver_solution.total_distance_km
        ),
        routes=routes,
        all_nodes=all_nodes_dict,
        metadata=Metadata(
            generated_at=datetime.now(),
            algorithm_used="OR-Tools + Guided Local Search" if solver_solution.status == 'OPTIMAL' else "Heuristic",
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
    # Optional alternate name used by some payloads / examples
    fuel_cost_per_distance: Optional[float] = None

    @root_validator(pre=True)
    def _populate_fuel_cost_per_km(cls, values):
        if "fuel_cost_per_km" not in values and "fuel_cost_per_distance" in values:
            values["fuel_cost_per_km"] = values["fuel_cost_per_distance"]
        return values


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
    Uses VRPSolverV2 and maps solver node IDs back to database point IDs
    """
    try:
        logger.info(f"Received Go backend request: {len(request.nodes)} nodes, {len(request.vehicles)} vehicles")
        logger.info(f"Hub point ID: {request.hub_point_id}")
        logger.info(f"Point IDs by index: {request.point_ids_by_index}")
        
        # Lazy-load heavy solver module only when we actually handle a solve request.
        try:
            _, SolverNode, SolverVehicle, _ = _get_solver_symbols()
        except Exception:
            return GoBackendSolveResponse(
                success=False,
                routes=[],
                summary={},
                message="VRP Solver is not available"
            )
        
        # Build solver input
        # point_ids_by_index = [hub_id, collection_point_ids...]
        # Solver expects: Node 1 = hub (index 0), Node 2-N = collection points
        
        num_nodes = len(request.point_ids_by_index)
        distance_matrix = np.array(request.distance_matrix, dtype=float)
        
        # Create SolverNode objects
        solver_nodes = []
        node_demand_map = {n["point_id"]: n for n in request.nodes}
        
        for idx, point_id in enumerate(request.point_ids_by_index):
            node_id = idx + 1  # 1-indexed for solver
            
            # Get demand (hub has 0 demand)
            demand_regular = 0
            demand_recycle = 0
            if point_id != request.hub_point_id and point_id in node_demand_map:
                demand_regular = node_demand_map[point_id]["demand_regular"]
                demand_recycle = node_demand_map[point_id]["demand_recycle"]
            
            solver_node = SolverNode(
                id=node_id,
                name=f"Point {point_id}",
                general_demand=float(demand_regular),
                recycle_demand=float(demand_recycle),
                is_depot=(idx == 0),  # First node is hub/depot
                is_checkpoint=False  # No checkpoint for Go backend
            )
            solver_nodes.append(solver_node)
        
        # Create SolverVehicle objects
        solver_vehicles = []
        for vehicle in request.vehicles:
            solver_vehicle = SolverVehicle(
                type_id=vehicle.vehicle_id,
                general_capacity=float(vehicle.capacity_regular),
                recycle_capacity=float(vehicle.capacity_recycle),
                fixed_cost=float(vehicle.fixed_cost),
                fuel_cost_per_km=float(vehicle.fuel_cost_per_km)
            )
            solver_vehicles.append(solver_vehicle)
        
        # Choose vehicle(s) by fuel cost (keep existing behavior: base calculations on the cheapest one)
        solver_vehicles_sorted = sorted(solver_vehicles, key=lambda v: v.fuel_cost_per_km)
        best_vehicle = solver_vehicles_sorted[0]
        
        # Calculate minimum vehicles needed
        total_general = sum(n.general_demand for n in solver_nodes)
        total_recycle = sum(n.recycle_demand for n in solver_nodes)
        
        import math
        min_veh_gen = math.ceil(total_general / best_vehicle.general_capacity) if best_vehicle.general_capacity > 0 else 1
        min_veh_rec = math.ceil(total_recycle / best_vehicle.recycle_capacity) if best_vehicle.recycle_capacity > 0 else 1
        num_vehicles = max(min_veh_gen, min_veh_rec, 1)

        # Enforce: do not reuse the same real vehicle across multiple routes.
        # We do this by providing N distinct vehicles to OR-Tools (unused vehicles become depot->depot and are ignored).
        num_vehicles = min(num_vehicles, len(solver_vehicles_sorted))
        selected_vehicles = solver_vehicles_sorted[:num_vehicles]
        
        depot_idx = 0
        
        logger.info(f"Solving with {num_vehicles} distinct vehicles using OR-Tools...")
        
        # Solve using OR-Tools directly (without checkpoint requirement)
        # Enforce single in-flight solve at a time to control memory usage.
        wait_start = time.monotonic()
        async with _SOLVE_LOCK:
            waited_s = time.monotonic() - wait_start
            if waited_s >= 0.25:
                logger.info(f"Go-backend solve request waited {waited_s:.2f}s for the solver lock")

            solver_solution = await run_in_threadpool(
                solve_vrp_without_checkpoint_multi,
                solver_nodes,
                selected_vehicles,
                distance_matrix,
                depot_idx,
                60,
            )
        
        # Convert solver solution back to database point IDs
        routes = []
        total_distance_m = 0
        total_fixed_cost = 0.0
        total_fuel_cost = 0.0
        
        for solver_route in solver_solution.routes:
            # Convert solver node IDs (1-indexed) to database point IDs
            point_ids = []
            for node_id in solver_route.nodes:
                idx = node_id - 1  # Convert to 0-indexed
                point_id = request.point_ids_by_index[idx]
                point_ids.append(point_id)
            
            # Find vehicle from original request
            vehicle = next((v for v in request.vehicles if v.vehicle_id == solver_route.vehicle_type), request.vehicles[0])
            
            routes.append(GoBackendRouteOut(
                vehicle_id=solver_route.vehicle_type,
                point_ids=point_ids,
                distance_m=int(solver_route.distance_meters),
                fixed_cost=solver_route.fixed_cost,
                fuel_cost=solver_route.fuel_cost,
                total_cost=solver_route.total_cost
            ))
            
            total_distance_m += solver_route.distance_meters
            total_fixed_cost += solver_route.fixed_cost
            total_fuel_cost += solver_route.fuel_cost
        
        summary = {
            "total_cost": total_fixed_cost + total_fuel_cost,
            "total_fixed_cost": total_fixed_cost,
            "total_fuel_cost": total_fuel_cost,
            "total_distance_m": float(total_distance_m),
            "total_vehicles": len(routes)
        }
        
        logger.info(f"Solution generated: {len(routes)} routes, {total_distance_m/1000:.2f} km, {summary['total_cost']:.2f} THB")
        
        return GoBackendSolveResponse(
            success=True,
            routes=routes,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error solving VRP for Go backend: {e}", exc_info=True)
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
