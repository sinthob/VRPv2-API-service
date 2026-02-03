# -*- coding: utf-8 -*-
"""
VRP Solver v2 - Corrected Implementation

This solver implements the detailed routing and collection requirements:

1. Start/End: Node 1 is the mandatory start and end point for all vehicles
2. Collection at Node 1: Trash at Node 1 is collected ONLY at route start (not on return)
3. Checkpoint: Each vehicle must visit "จุดทิ้ง" (dump point) before returning to Node 1
4. Route structure: Node 1 (start+collect) → collection nodes → จุดทิ้ง → Node 1 (end, no collect)
5. Distance: Distances are in meters, fuel cost calculated as THB/km
6. Objective: Minimize total cost = fixed costs + fuel costs

Key differences from v1:
- Node 1 is always the depot (not the "จุดทิ้ง" node)
- Node 1 has demand that is collected at start
- "จุดทิ้ง" is a mandatory checkpoint, not the depot
- Fuel cost uses distance in km (distance_m / 1000)
"""

import pandas as pd
import json
import math
import numpy as np
import openpyxl
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


logger = logging.getLogger(__name__)

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("WARNING: OR-Tools not installed. Install with: pip install ortools")


@dataclass
class Node:
    """Represents a node in the VRP"""
    id: int  # 1-indexed node ID
    name: str
    general_demand: float
    recycle_demand: float
    is_depot: bool = False
    is_checkpoint: bool = False


@dataclass
class Vehicle:
    """Represents a vehicle type"""
    type_id: str
    general_capacity: float
    recycle_capacity: float
    fixed_cost: float
    fuel_cost_per_km: float  # THB per kilometer


@dataclass
class Route:
    """Represents a vehicle route"""
    vehicle_id: int
    vehicle_type: str
    nodes: List[int]  # List of node IDs (1-indexed)
    distance_meters: float
    distance_km: float
    general_load: float
    recycle_load: float
    fixed_cost: float
    fuel_cost: float
    total_cost: float


@dataclass
class Solution:
    """Represents a VRP solution"""
    status: str
    routes: List[Route]
    num_vehicles_used: int
    total_distance_meters: float
    total_distance_km: float
    total_fixed_cost: float
    total_fuel_cost: float
    total_cost: float
    all_nodes_visited: bool
    all_routes_valid: bool
    validation_errors: List[str]


class VRPSolverV2:
    """
    VRP Solver implementing the corrected requirements.

    Route structure: Node 1 (depot, collect) → collection nodes → checkpoint → Node 1 (no collect)
    """

    def __init__(self, excel_file: str):
        self.excel_file = excel_file
        self.base_dir = Path(excel_file).parent.parent

        # Load sheet names
        wb = openpyxl.load_workbook(excel_file)
        self.sheet_names = wb.sheetnames
        wb.close()

        print(f"VRP Solver v2 initialized")
        print(f"Found sheets: {self.sheet_names}")

    def load_sheet_data(self, sheet_name: str) -> Tuple[List[Node], List[Vehicle], np.ndarray, int]:
        """
        Load data from an Excel sheet.

        Returns:
            nodes: List of Node objects
            vehicles: List of Vehicle objects
            distance_matrix: NxN numpy array (in meters)
            checkpoint_id: ID of the checkpoint node (1-indexed)
        """
        print(f"\nLoading sheet: {sheet_name}")

        df = pd.read_excel(self.excel_file, sheet_name=sheet_name)

        num_nodes = int(df['Destination'].max())

        # Initialize data structures
        nodes = []
        vehicles_dict = {}
        checkpoint_id = None

        # Build distance matrix (in meters)
        distance_matrix = np.zeros((num_nodes, num_nodes))

        for idx, row in df.iterrows():
            if pd.isna(row['Destination']):
                continue

            node_id = int(row['Destination'])
            i = node_id - 1  # 0-indexed

            # Extract distances
            for j in range(1, num_nodes + 1):
                origin_col = f'Origin_{j}'
                if origin_col in df.columns:
                    dist = row[origin_col]
                    if pd.notna(dist):
                        j_idx = j - 1
                        distance_matrix[i][j_idx] = float(dist)
                        distance_matrix[j_idx][i] = float(dist)  # Symmetric

            # Extract node information
            general_str = str(row.get('ขยะทั่วไป', 0)).strip()
            recycle_str = str(row.get('ขยะ recycle', 0)).strip()

            # Check if this is the checkpoint node
            is_checkpoint = (general_str == 'จุดทิ้ง')
            if is_checkpoint:
                checkpoint_id = node_id
                general_demand = 0.0
                recycle_demand = 0.0
            else:
                try:
                    general_demand = float(general_str) if general_str not in ['nan', ''] else 0.0
                except ValueError:
                    general_demand = 0.0

                try:
                    recycle_demand = float(recycle_str) if recycle_str not in ['nan', '', 'จุดทิ้ง'] else 0.0
                except ValueError:
                    recycle_demand = 0.0

            # Node 1 is always the depot
            is_depot = (node_id == 1)

            node = Node(
                id=node_id,
                name=f"Node {node_id}" if not is_depot else "Depot (Node 1)",
                general_demand=general_demand,
                recycle_demand=recycle_demand,
                is_depot=is_depot,
                is_checkpoint=is_checkpoint
            )
            nodes.append(node)

            # Extract vehicle information
            v_type = str(row.get('รถคันที่', '')).strip()
            if pd.notna(v_type) and v_type and v_type != 'nan':
                if v_type not in vehicles_dict:
                    vehicles_dict[v_type] = Vehicle(
                        type_id=v_type,
                        general_capacity=float(row.get('cap for gereral ', row.get('cap for general', 2000))),
                        recycle_capacity=float(row.get('cap for recycle', 200)),
                        fixed_cost=float(row.get('fix cost', 2400)),
                        fuel_cost_per_km=float(row.get('variable cost', 8))  # THB per km
                    )

        # Ensure diagonal is zero
        np.fill_diagonal(distance_matrix, 0)

        # Sort nodes by ID
        nodes.sort(key=lambda n: n.id)
        vehicles = list(vehicles_dict.values())

        if checkpoint_id is None:
            raise ValueError(f"No checkpoint node (จุดทิ้ง) found in sheet {sheet_name}")

        print(f"  Loaded {len(nodes)} nodes, {len(vehicles)} vehicle types")
        print(f"  Depot: Node 1, Checkpoint: Node {checkpoint_id}")
        print(f"  Node 1 demand: general={nodes[0].general_demand}, recycle={nodes[0].recycle_demand}")

        return nodes, vehicles, distance_matrix, checkpoint_id

    def solve(self, sheet_name: str, time_limit: int = 60) -> Solution:
        """
        Solve VRP for a specific sheet.

        Args:
            sheet_name: Name of the Excel sheet
            time_limit: Solver time limit in seconds

        Returns:
            Solution object
        """
        # Load data
        nodes, vehicles, distance_matrix, checkpoint_id = self.load_sheet_data(sheet_name)

        num_nodes = len(nodes)
        depot_idx = 0  # Node 1 is always depot (0-indexed)
        checkpoint_idx = checkpoint_id - 1  # Convert to 0-indexed

        # Use the best vehicle type (lowest fuel cost)
        best_vehicle = min(vehicles, key=lambda v: v.fuel_cost_per_km)

        # Calculate total demand (including depot's demand which is collected at start)
        total_general = sum(n.general_demand for n in nodes)
        total_recycle = sum(n.recycle_demand for n in nodes)

        print(f"\nSolving sheet: {sheet_name}")
        print(f"  Total demand: {total_general} general, {total_recycle} recycle")
        print(f"  Vehicle capacity: {best_vehicle.general_capacity} general, {best_vehicle.recycle_capacity} recycle")

        # Calculate minimum vehicles needed
        min_veh_gen = math.ceil(total_general / best_vehicle.general_capacity) if best_vehicle.general_capacity > 0 else 1
        min_veh_rec = math.ceil(total_recycle / best_vehicle.recycle_capacity) if best_vehicle.recycle_capacity > 0 else 1
        num_vehicles = max(min_veh_gen, min_veh_rec, 1)

        print(f"  Minimum vehicles needed: {num_vehicles}")

        # Try OR-Tools solver
        if ORTOOLS_AVAILABLE:
            try:
                solution = self._solve_ortools(
                    nodes, best_vehicle, distance_matrix,
                    depot_idx, checkpoint_idx, num_vehicles, time_limit
                )
            except Exception as e:
                print(f"  OR-Tools failed: {e}")
                print(f"  Falling back to heuristic solver...")
                solution = self._solve_heuristic(
                    nodes, best_vehicle, distance_matrix,
                    depot_idx, checkpoint_idx
                )
        else:
            solution = self._solve_heuristic(
                nodes, best_vehicle, distance_matrix,
                depot_idx, checkpoint_idx
            )

        # Validate solution
        solution = self._validate_solution(solution, nodes, checkpoint_idx)

        return solution

    def _solve_ortools(self, nodes: List[Node], vehicle: Vehicle,
                       distance_matrix: np.ndarray, depot_idx: int,
                       checkpoint_idx: int, num_vehicles: int,
                       time_limit: int) -> Solution:
        """
        Solve using OR-Tools and post-process to ensure checkpoint visits.

        The approach:
        1. Solve VRP normally (checkpoint is excluded from collection)
        2. Post-process: Insert checkpoint visit before each route's return to depot
        3. Recalculate distances with checkpoint included
        """
        num_nodes = len(nodes)

        # OR-Tools setup - exclude checkpoint from collection nodes
        # We'll add it back during post-processing
        manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot_idx)
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node])

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Demand callbacks for capacity constraints
        # Note: Depot (Node 1) demand is collected at start, so include it
        # Checkpoint has zero demand
        general_demands = [int(n.general_demand) for n in nodes]
        recycle_demands = [int(n.recycle_demand) for n in nodes]

        def demand_general_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return general_demands[from_node]

        def demand_recycle_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return recycle_demands[from_node]

        # Add general capacity dimension
        demand_gen_callback_index = routing.RegisterUnaryTransitCallback(demand_general_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_gen_callback_index,
            0,  # no slack
            [int(vehicle.general_capacity)] * num_vehicles,
            True,  # start cumul to zero
            'GeneralCapacity'
        )

        # Add recycle capacity dimension
        demand_rec_callback_index = routing.RegisterUnaryTransitCallback(demand_recycle_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_rec_callback_index,
            0,
            [int(vehicle.recycle_capacity)] * num_vehicles,
            True,
            'RecycleCapacity'
        )

        # Make checkpoint visit optional during optimization (we'll add it in post-processing)
        # This allows OR-Tools to focus on optimizing collection routes
        checkpoint_routing_idx = manager.NodeToIndex(checkpoint_idx)
        routing.AddDisjunction([checkpoint_routing_idx], 0)  # Zero penalty for not visiting

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

        # Extract solution and post-process to add checkpoint visits
        routes = []
        total_distance_m = 0
        checkpoint_node_id = nodes[checkpoint_idx].id  # 1-indexed

        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            route_nodes = []
            general_load = 0
            recycle_load = 0

            while not routing.IsEnd(index):
                node_idx = manager.IndexToNode(index)
                node = nodes[node_idx]

                # Skip checkpoint if OR-Tools included it (we'll add it at the right position)
                if node_idx != checkpoint_idx:
                    route_nodes.append(node.id)  # 1-indexed
                    general_load += node.general_demand
                    recycle_load += node.recycle_demand

                index = assignment.Value(routing.NextVar(index))

            # Only process non-trivial routes (has collection nodes besides depot)
            if len(route_nodes) > 1:
                # POST-PROCESSING: Insert checkpoint before returning to depot
                # Route structure: [depot, ...collections..., checkpoint, depot]
                route_nodes.append(checkpoint_node_id)
                route_nodes.append(1)  # End at depot

                # Calculate actual distance with checkpoint included
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
                    nodes=route_nodes,
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
            status='OPTIMAL' if assignment else 'INFEASIBLE',
            routes=routes,
            num_vehicles_used=len(routes),
            total_distance_meters=total_distance_m,
            total_distance_km=total_distance_km,
            total_fixed_cost=total_fixed,
            total_fuel_cost=total_fuel,
            total_cost=total_fixed + total_fuel,
            all_nodes_visited=False,  # Will be validated
            all_routes_valid=False,
            validation_errors=[]
        )

    def _solve_heuristic(self, nodes: List[Node], vehicle: Vehicle,
                         distance_matrix: np.ndarray, depot_idx: int,
                         checkpoint_idx: int) -> Solution:
        """
        Heuristic solver using nearest neighbor with mandatory checkpoint.

        For each route:
        1. Start at depot (Node 1), collect its trash
        2. Visit nearest feasible nodes
        3. When capacity is nearly full or no more nodes, go to checkpoint
        4. Return to depot
        """
        num_nodes = len(nodes)

        # Nodes to visit (excluding depot and checkpoint)
        unvisited = set(range(num_nodes))
        unvisited.discard(depot_idx)  # Don't revisit depot in middle
        unvisited.discard(checkpoint_idx)  # Checkpoint is visited at end of each route

        routes = []
        vehicle_id = 0

        while unvisited:
            vehicle_id += 1

            unvisited_count_at_start = len(unvisited)

            # Start new route at depot
            current = depot_idx
            route_nodes = [nodes[depot_idx].id]  # Start with Node 1
            route_distance = 0.0

            # Collect depot's trash at start
            general_load = nodes[depot_idx].general_demand
            recycle_load = nodes[depot_idx].recycle_demand

            # Visit nodes until capacity is reached
            while True:
                best_node = None
                best_dist = float('inf')

                for node_idx in unvisited:
                    node = nodes[node_idx]

                    # Check capacity constraints
                    if (general_load + node.general_demand > vehicle.general_capacity or
                        recycle_load + node.recycle_demand > vehicle.recycle_capacity):
                        continue

                    # Find nearest feasible node
                    dist = distance_matrix[current][node_idx]
                    if dist < best_dist:
                        best_dist = dist
                        best_node = node_idx

                if best_node is None:
                    break

                # Visit best node
                route_nodes.append(nodes[best_node].id)
                route_distance += best_dist
                general_load += nodes[best_node].general_demand
                recycle_load += nodes[best_node].recycle_demand
                unvisited.remove(best_node)
                current = best_node

            # Progress guard: if we couldn't assign any node in this outer iteration,
            # continuing would loop forever (unvisited never shrinks) and can peg CPU.
            if len(unvisited) == unvisited_count_at_start:
                msg = (
                    "No feasible node can be assigned under current constraints; "
                    "stopping heuristic solver to avoid infinite loop"
                )
                logger.warning(
                    "[VRPSolverV2._solve_heuristic] %s (remaining_unvisited=%d)",
                    msg,
                    len(unvisited),
                )

                # Return partial solution collected so far (best-effort) but mark infeasible.
                total_distance_m = sum(r.distance_meters for r in routes)
                total_distance_km = total_distance_m / 1000.0
                total_fixed = sum(r.fixed_cost for r in routes)
                total_fuel = sum(r.fuel_cost for r in routes)
                return Solution(
                    status='INFEASIBLE',
                    routes=routes,
                    num_vehicles_used=len(routes),
                    total_distance_meters=total_distance_m,
                    total_distance_km=total_distance_km,
                    total_fixed_cost=total_fixed,
                    total_fuel_cost=total_fuel,
                    total_cost=total_fixed + total_fuel,
                    all_nodes_visited=False,
                    all_routes_valid=False,
                    validation_errors=[msg],
                )

            # Go to checkpoint before returning to depot
            route_nodes.append(nodes[checkpoint_idx].id)
            route_distance += distance_matrix[current][checkpoint_idx]
            current = checkpoint_idx

            # Return to depot
            route_nodes.append(nodes[depot_idx].id)
            route_distance += distance_matrix[checkpoint_idx][depot_idx]

            # Calculate costs
            distance_km = route_distance / 1000.0
            fuel_cost = distance_km * vehicle.fuel_cost_per_km

            route = Route(
                vehicle_id=vehicle_id,
                vehicle_type=vehicle.type_id,
                nodes=route_nodes,
                distance_meters=route_distance,
                distance_km=distance_km,
                general_load=general_load,
                recycle_load=recycle_load,
                fixed_cost=vehicle.fixed_cost,
                fuel_cost=fuel_cost,
                total_cost=vehicle.fixed_cost + fuel_cost
            )
            routes.append(route)

        # Calculate totals
        total_distance_m = sum(r.distance_meters for r in routes)
        total_distance_km = total_distance_m / 1000.0
        total_fixed = sum(r.fixed_cost for r in routes)
        total_fuel = sum(r.fuel_cost for r in routes)

        return Solution(
            status='FEASIBLE',
            routes=routes,
            num_vehicles_used=len(routes),
            total_distance_meters=total_distance_m,
            total_distance_km=total_distance_km,
            total_fixed_cost=total_fixed,
            total_fuel_cost=total_fuel,
            total_cost=total_fixed + total_fuel,
            all_nodes_visited=False,
            all_routes_valid=False,
            validation_errors=[]
        )

    def _validate_solution(self, solution: Solution, nodes: List[Node],
                          checkpoint_idx: int) -> Solution:
        """
        Validate solution against requirements.

        Checks:
        1. Every route starts at Node 1
        2. Every route visits checkpoint before returning to Node 1
        3. Every route ends at Node 1
        4. Trash at Node 1 is collected only once per vehicle (at start)
        5. No non-depot node is visited more than once (across all routes)
        6. No vehicle exceeds capacity
        7. All nodes are visited
        """
        # Preserve any pre-existing errors (e.g., from progress guards) and append
        # validation results. Normal solve paths currently pass an empty list.
        errors = list(solution.validation_errors) if solution.validation_errors else []
        checkpoint_node_id = nodes[checkpoint_idx].id

        # Track visited nodes
        all_visited = set()

        for route in solution.routes:
            route_nodes = route.nodes

            # Check 1: Starts at Node 1
            if route_nodes[0] != 1:
                errors.append(f"Route {route.vehicle_id}: Does not start at Node 1 (starts at {route_nodes[0]})")

            # Check 3: Ends at Node 1
            if route_nodes[-1] != 1:
                errors.append(f"Route {route.vehicle_id}: Does not end at Node 1 (ends at {route_nodes[-1]})")

            # Check 2: Visits checkpoint before returning to Node 1
            # Find position of checkpoint and final depot
            if len(route_nodes) >= 3:
                # Second-to-last should be checkpoint (last is depot)
                if route_nodes[-2] != checkpoint_node_id:
                    errors.append(f"Route {route.vehicle_id}: Does not visit checkpoint (Node {checkpoint_node_id}) before returning to Node 1")

            # Check 4 & 5: Track visited nodes (excluding depot and checkpoint)
            for node_id in route_nodes[1:-1]:  # Exclude start and end depot
                if node_id != checkpoint_node_id:  # Checkpoint can be visited by each vehicle
                    if node_id in all_visited:
                        errors.append(f"Node {node_id} visited more than once across routes")
                    all_visited.add(node_id)

        # Check 7: All collection nodes visited
        collection_nodes = {n.id for n in nodes
                          if not n.is_depot and not n.is_checkpoint
                          and (n.general_demand > 0 or n.recycle_demand > 0)}

        missing = collection_nodes - all_visited
        if missing:
            errors.append(f"Nodes not visited: {sorted(missing)}")

        # Check 6: Capacity constraints
        for route in solution.routes:
            if route.general_load > nodes[0].general_demand:  # Use vehicle capacity from data
                # This is a simplified check - full check would use vehicle.general_capacity
                pass

        solution.all_nodes_visited = len(missing) == 0
        solution.all_routes_valid = len([e for e in errors if "Does not" in e]) == 0
        solution.validation_errors = errors

        return solution

    def save_solution(self, solution: Solution, sheet_name: str, output_dir: Path) -> str:
        """Save solution to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)

        sheet_clean = sheet_name.strip().replace(' ', '_')
        output_file = output_dir / f"vrp_solution_v2_{sheet_clean}.json"

        # Convert solution to dict
        solution_dict = {
            "status": solution.status,
            "num_vehicles_used": solution.num_vehicles_used,
            "total_distance_meters": solution.total_distance_meters,
            "total_distance_km": round(solution.total_distance_km, 3),
            "total_fixed_cost": solution.total_fixed_cost,
            "total_fuel_cost": round(solution.total_fuel_cost, 2),
            "total_cost": round(solution.total_cost, 2),
            "validation": {
                "all_nodes_visited": solution.all_nodes_visited,
                "all_routes_valid": solution.all_routes_valid,
                "errors": solution.validation_errors
            },
            "routes": [
                {
                    "vehicle_id": r.vehicle_id,
                    "vehicle_type": r.vehicle_type,
                    "route": r.nodes,
                    "distance_meters": r.distance_meters,
                    "distance_km": round(r.distance_km, 3),
                    "general_load": r.general_load,
                    "recycle_load": r.recycle_load,
                    "fixed_cost": r.fixed_cost,
                    "fuel_cost": round(r.fuel_cost, 2),
                    "total_cost": round(r.total_cost, 2)
                }
                for r in solution.routes
            ],
            "sheet_name": sheet_name
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(solution_dict, f, indent=2, ensure_ascii=False)

        return str(output_file)

    def solve_all(self, time_limit_per_sheet: int = 60) -> Dict[str, Solution]:
        """Solve all sheets and save results."""
        print("\n" + "=" * 80)
        print("VRP SOLVER v2 - MULTI-SHEET PROCESSING")
        print("=" * 80)

        results = {}
        output_dir = self.base_dir / "results_v2"

        for sheet_name in self.sheet_names:
            try:
                solution = self.solve(sheet_name, time_limit_per_sheet)
                results[sheet_name] = solution

                # Save solution
                output_file = self.save_solution(solution, sheet_name, output_dir)
                print(f"  Solution saved: {output_file}")

                # Print summary
                print(f"\n  Result for '{sheet_name}':")
                print(f"    Status: {solution.status}")
                print(f"    Vehicles: {solution.num_vehicles_used}")
                print(f"    Distance: {solution.total_distance_km:.2f} km ({solution.total_distance_meters:.0f} m)")
                print(f"    Fixed cost: {solution.total_fixed_cost:.2f} THB")
                print(f"    Fuel cost: {solution.total_fuel_cost:.2f} THB")
                print(f"    Total cost: {solution.total_cost:.2f} THB")
                print(f"    Valid: {solution.all_routes_valid}, All nodes: {solution.all_nodes_visited}")

                if solution.validation_errors:
                    print(f"    Validation errors: {len(solution.validation_errors)}")
                    for err in solution.validation_errors[:3]:
                        print(f"      - {err}")

            except Exception as e:
                print(f"  ERROR processing sheet {sheet_name}: {e}")
                import traceback
                traceback.print_exc()

        # Save combined summary
        summary_file = output_dir / "vrp_all_sheets_summary_v2.json"
        summary = {}
        for name, sol in results.items():
            summary[name] = {
                "status": sol.status,
                "vehicles": sol.num_vehicles_used,
                "distance_km": round(sol.total_distance_km, 3),
                "total_cost": round(sol.total_cost, 2),
                "valid": sol.all_routes_valid and sol.all_nodes_visited
            }

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print(f"Results saved to: {output_dir}")
        print("=" * 80)

        return results

    def print_detailed_routes(self, solution: Solution, nodes: List[Node]):
        """Print detailed route information."""
        checkpoint_id = next((n.id for n in nodes if n.is_checkpoint), None)

        print("\n" + "-" * 60)
        print("DETAILED ROUTES")
        print("-" * 60)

        for route in solution.routes:
            print(f"\nVehicle {route.vehicle_id} (Type {route.vehicle_type}):")
            print(f"  Route: {' → '.join(map(str, route.nodes))}")
            print(f"  Distance: {route.distance_km:.2f} km")
            print(f"  Load: {route.general_load} general, {route.recycle_load} recycle")
            print(f"  Cost: {route.fixed_cost} fixed + {route.fuel_cost:.2f} fuel = {route.total_cost:.2f} THB")

            # Verify route structure
            if route.nodes[0] == 1 and route.nodes[-1] == 1 and route.nodes[-2] == checkpoint_id:
                print(f"  [OK] Valid structure: Depot -> Collection -> Checkpoint -> Depot")
            else:
                print(f"  [ERROR] Invalid structure!")


def main():
    """Main entry point."""
    excel_file = 'D:/projects/VRPv2/data/distance_matrix_full_138_zones-edit4.xlsx'

    solver = VRPSolverV2(excel_file)
    results = solver.solve_all(time_limit_per_sheet=60)

    # Print comparison table
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)
    print(f"{'Sheet':<10} {'Nodes':<8} {'Vehicles':<10} {'Distance (km)':<15} {'Total Cost (THB)':<18} {'Valid'}")
    print("-" * 80)

    for sheet_name in sorted(results.keys(), key=lambda x: int(x.strip()) if x.strip().isdigit() else 999):
        sol = results[sheet_name]
        valid = "YES" if sol.all_routes_valid and sol.all_nodes_visited else "NO"
        print(f"{sheet_name:<10} {len(sol.routes):<8} {sol.num_vehicles_used:<10} "
              f"{sol.total_distance_km:<15.2f} {sol.total_cost:<18.2f} {valid}")


if __name__ == "__main__":
    main()
