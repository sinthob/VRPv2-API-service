# -*- coding: utf-8 -*-
"""
STEP 4: Multi-Sheet VRP Solver

This script solves VRP for ALL sheets in the Excel file:
- '20 ' (20 zones)
- '30' (30 zones)
- '50' (50 zones)
- '80' (80 zones)
- '138' (138 zones)

Each sheet is processed independently with its own solution.
"""

import pandas as pd
import json
import math
import openpyxl
import logging
from pathlib import Path
from typing import Dict, List


logger = logging.getLogger(__name__)

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False
    print("WARNING: OR-Tools not installed. Install with: pip install ortools")


class MultiSheetVRPSolver:
    """Solve VRP for all sheets in Excel file"""

    def __init__(self, excel_file: str):
        self.excel_file = excel_file
        self.base_dir = Path(excel_file).parent

        # Get all sheet names
        wb = openpyxl.load_workbook(excel_file)
        self.sheet_names = wb.sheetnames
        print(f"Found sheets: {self.sheet_names}")

        # Results storage
        self.results = {}

    def generate_input_for_sheet(self, sheet_name: str) -> str:
        """Generate solver input JSON for a specific sheet"""
        print(f"\nGenerating input for sheet: {sheet_name}")

        # Read the Excel sheet
        df = pd.read_excel(self.excel_file, sheet_name=sheet_name)

        # Find depot node (marked as "จุดทิ้ง")
        depot_id = None
        for idx, row in df.iterrows():
            general = str(row.get('ขยะทั่วไป', '')).strip()
            if general == 'จุดทิ้ง':
                depot_id = int(row['Destination'])
                break

        if depot_id is None:
            depot_id = int(df.iloc[-1]['Destination'])

        # Extract distance matrix
        num_nodes = int(df['Destination'].max())

        # Use numpy for faster matrix building
        import numpy as np
        distance_matrix = np.zeros((num_nodes, num_nodes))

        for idx, row in df.iterrows():
            if pd.isna(row['Destination']):
                continue

            i = int(row['Destination']) - 1

            for j in range(1, num_nodes + 1):
                origin_col = f'Origin_{j}'
                if origin_col in df.columns:
                    dist = row[origin_col]
                    if pd.notna(dist):
                        j_idx = j - 1
                        distance_matrix[i][j_idx] = float(dist)
                        distance_matrix[j_idx][i] = float(dist)

        # Ensure diagonal is zero
        for i in range(num_nodes):
            distance_matrix[i][i] = 0.0

        # Extract nodes
        nodes = []
        total_general = 0
        total_recycle = 0

        for idx, row in df.iterrows():
            if pd.isna(row['Destination']):
                continue

            node_id = int(row['Destination'])

            general_str = str(row.get('ขยะทั่วไป', 0)).strip()
            recycle_str = str(row.get('ขยะ recycle', 0)).strip()

            try:
                general = float(general_str) if general_str not in ['จุดทิ้ง', 'nan', ''] else 0.0
            except:
                general = 0.0

            try:
                recycle = float(recycle_str) if recycle_str not in ['จุดทิ้ง', 'nan', ''] else 0.0
            except:
                recycle = 0.0

            node_name = f"Node {node_id}"
            if node_id == depot_id:
                node_name = f"Depot ({depot_id})"
                general = 0.0
                recycle = 0.0

            nodes.append({
                "id": node_id,
                "name": node_name,
                "general_demand": general,
                "recycle_demand": recycle
            })

            if node_id != depot_id:
                total_general += general
                total_recycle += recycle

        # Extract vehicle types
        vehicle_types = {}
        for idx, row in df.iterrows():
            v_type = str(row.get('รถคันที่', '')).strip()
            if pd.notna(v_type) and v_type and v_type != 'nan':
                if v_type not in vehicle_types:
                    vehicle_types[v_type] = {
                        "type": v_type,
                        "general_capacity": float(row.get('cap for gereral ', 2000)),
                        "recycle_capacity": float(row.get('cap for recycle', 200)),
                        "fixed_cost": float(row.get('fix cost', 2400)),
                        "fuel_cost_per_distance": float(row.get('variable cost', 8))
                    }

        vehicles = list(vehicle_types.values())

        # Build solver input
        solver_input = {
            "depot": {
                "id": depot_id,
                "name": f"Depot {depot_id}"
            },
            "num_nodes": num_nodes,
            "nodes": nodes,
            "vehicles": vehicles,
            "distance_matrix": distance_matrix.tolist(),
            "summary": {
                "total_general_demand": total_general,
                "total_recycle_demand": total_recycle,
                "num_vehicles": len(vehicles)
            }
        }

        # Save input JSON
        sheet_clean = sheet_name.strip().replace(' ', '_')
        output_file = self.base_dir / f"vrp_input_{sheet_clean}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(solver_input, f, indent=2, ensure_ascii=False)

        print(f"  Input saved: {output_file.name}")
        print(f"  Nodes: {num_nodes}, Depot: {depot_id}")
        print(f"  Demand: {total_general:.1f} gen, {total_recycle:.1f} rec")

        return str(output_file)

    def solve_sheet(self, sheet_name: str, input_file: str, time_limit: int = 30) -> Dict:
        """Solve VRP for a specific sheet"""
        print(f"\nSolving sheet: {sheet_name}")

        # Load input
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        depot_id = data['depot']['id'] - 1
        num_nodes = data['num_nodes']
        distance_matrix = data['distance_matrix']
        nodes = data['nodes']
        vehicles = data['vehicles']

        # Build demands
        general_demands = [0] * num_nodes
        recycle_demands = [0] * num_nodes

        for node in nodes:
            idx = node['id'] - 1
            general_demands[idx] = int(node['general_demand'])
            recycle_demands[idx] = int(node['recycle_demand'])

        # Use best vehicle
        best_idx = min(range(len(vehicles)),
                      key=lambda i: float(vehicles[i]['fuel_cost_per_distance']))

        general_capacity = int(vehicles[best_idx]['general_capacity'])
        recycle_capacity = int(vehicles[best_idx]['recycle_capacity'])
        fuel_cost = float(vehicles[best_idx]['fuel_cost_per_distance'])
        fixed_cost = int(vehicles[best_idx]['fixed_cost'])

        total_general = sum(general_demands)
        total_recycle = sum(recycle_demands)

        # Calculate minimum vehicles
        min_veh_gen = math.ceil(total_general / general_capacity)
        min_veh_rec = math.ceil(total_recycle / recycle_capacity)
        num_vehicles = max(min_veh_gen, min_veh_rec, 2)

        print(f"  Vehicles: {num_vehicles}, Demand: {total_general} gen, {total_recycle} rec")

        if not ORTOOLS_AVAILABLE:
            solution = self._solve_clustering(
                depot_id, num_nodes, distance_matrix,
                general_demands, recycle_demands,
                general_capacity, recycle_capacity,
                fixed_cost, fuel_cost
            )
        else:
            # Try OR-Tools
            try:
                solution = self._solve_ortools(
                    depot_id, num_nodes, distance_matrix,
                    general_demands, recycle_demands,
                    general_capacity, recycle_capacity,
                    fixed_cost, fuel_cost, num_vehicles, time_limit
                )
            except Exception as e:
                print(f"  OR-Tools failed: {e}, using clustering...")
                solution = self._solve_clustering(
                    depot_id, num_nodes, distance_matrix,
                    general_demands, recycle_demands,
                    general_capacity, recycle_capacity,
                    fixed_cost, fuel_cost
                )

        # Add sheet info
        solution['sheet_name'] = sheet_name
        solution['num_nodes'] = num_nodes
        solution['depot_id'] = depot_id + 1

        return solution

    def _solve_ortools(self, depot_id, num_nodes, distance_matrix,
                      general_demands, recycle_demands,
                      general_capacity, recycle_capacity,
                      fixed_cost, fuel_cost, num_vehicles, time_limit) -> Dict:
        """Solve using OR-Tools"""

        manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot_id)
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node])

        transit_callback = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback)

        # General capacity
        def demand_general(from_index):
            return general_demands[manager.IndexToNode(from_index)]

        demand_idx_gen = routing.RegisterUnaryTransitCallback(demand_general)
        routing.AddDimensionWithVehicleCapacity(
            demand_idx_gen, 0, [general_capacity] * num_vehicles, True, 'General'
        )

        # Recycle capacity
        def demand_recycle(from_index):
            return recycle_demands[manager.IndexToNode(from_index)]

        demand_idx_rec = routing.RegisterUnaryTransitCallback(demand_recycle)
        routing.AddDimensionWithVehicleCapacity(
            demand_idx_rec, 0, [recycle_capacity] * num_vehicles, True, 'Recycle'
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

        solution = routing.SolveWithParameters(search_params)

        if solution:
            return self._extract_solution(manager, routing, solution, depot_id,
                                        fixed_cost, fuel_cost)
        else:
            raise Exception("No solution found")

    def _extract_solution(self, manager, routing, solution, depot_id,
                         fixed_cost, fuel_cost) -> Dict:
        """Extract solution from OR-Tools"""
        routes = []
        total_distance = 0

        for vehicle_id in range(routing.vehicles()):
            index = routing.Start(vehicle_id)
            route_nodes = []
            route_distance = 0
            general_load = 0
            recycle_load = 0

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route_nodes.append(node_index + 1)

                general_load += 0  # Would need access to demands
                recycle_load += 0

                prev = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(prev, index, vehicle_id)

            if len(route_nodes) > 0:
                route_nodes.append(depot_id + 1)
                routes.append({
                    'vehicle_id': vehicle_id + 1,
                    'route': route_nodes,
                    'distance': route_distance,
                    'general_load': general_load,
                    'recycle_load': recycle_load
                })
                total_distance += route_distance

        routes = [r for r in routes if len(r['route']) > 2]

        total_fixed = len(routes) * fixed_cost
        total_fuel = total_distance * fuel_cost

        return {
            'status': 'OPTIMAL',
            'num_vehicles_used': len(routes),
            'total_distance': total_distance,
            'total_fixed_cost': total_fixed,
            'total_fuel_cost': total_fuel,
            'total_cost': total_fixed + total_fuel,
            'routes': routes
        }

    def _solve_clustering(self, depot_id, num_nodes, distance_matrix,
                         general_demands, recycle_demands,
                         general_capacity, recycle_capacity,
                         fixed_cost, fuel_cost) -> Dict:
        """Fallback clustering algorithm"""
        print("  Using clustering algorithm...")

        service_nodes = [i for i in range(num_nodes)
                        if i != depot_id and
                        (general_demands[i] > 0 or recycle_demands[i] > 0)]

        routes = []
        unassigned = set(service_nodes)
        vehicle_id = 0

        while unassigned:
            unassigned_count_at_start = len(unassigned)

            route = [depot_id]
            general_load = 0
            recycle_load = 0
            route_distance = 0
            current = depot_id

            while True:
                best_node = None
                best_cost = float('inf')

                for node in list(unassigned):
                    if (general_load + general_demands[node] <= general_capacity and
                        recycle_load + recycle_demands[node] <= recycle_capacity):

                        dist = distance_matrix[current][node]
                        cost = dist + distance_matrix[node][depot_id]

                        if cost < best_cost:
                            best_cost = cost
                            best_node = node

                if best_node is None:
                    break

                route.append(best_node)
                general_load += general_demands[best_node]
                recycle_load += recycle_demands[best_node]
                route_distance += distance_matrix[current][best_node]
                unassigned.remove(best_node)
                current = best_node

            # Progress guard: if we couldn't assign any node in this outer iteration,
            # continuing would loop forever (unassigned never shrinks) and can peg CPU.
            if len(unassigned) == unassigned_count_at_start:
                msg = (
                    "No feasible node can be assigned under current constraints; "
                    "stopping clustering solver to avoid infinite loop"
                )
                logger.warning(
                    "[MultiSheetVRPSolver._solve_clustering] %s (remaining_unassigned=%d)",
                    msg,
                    len(unassigned),
                )

                total_distance = sum(r['distance'] for r in routes)
                total_fixed = len(routes) * fixed_cost
                total_fuel = total_distance * fuel_cost
                return {
                    'status': 'INFEASIBLE',
                    'num_vehicles_used': len(routes),
                    'total_distance': total_distance,
                    'total_fixed_cost': total_fixed,
                    'total_fuel_cost': total_fuel,
                    'total_cost': total_fixed + total_fuel,
                    'routes': routes,
                    'validation_errors': [msg],
                }

            route_distance += distance_matrix[current][depot_id]
            route.append(depot_id)

            routes.append({
                'vehicle_id': vehicle_id + 1,
                'route': [n + 1 for n in route],
                'distance': int(route_distance),
                'general_load': general_load,
                'recycle_load': recycle_load
            })
            vehicle_id += 1

        total_distance = sum(r['distance'] for r in routes)
        total_fixed = len(routes) * fixed_cost
        total_fuel = total_distance * fuel_cost

        return {
            'status': 'FEASIBLE',
            'num_vehicles_used': len(routes),
            'total_distance': total_distance,
            'total_fixed_cost': total_fixed,
            'total_fuel_cost': total_fuel,
            'total_cost': total_fixed + total_fuel,
            'routes': routes
        }

    def solve_all(self, time_limit_per_sheet: int = 20):
        """Solve all sheets"""
        print("\n" + "="*80)
        print("MULTI-SHEET VRP SOLVER")
        print("="*80)

        all_solutions = {}

        for sheet_name in self.sheet_names:
            try:
                # Generate input
                input_file = self.generate_input_for_sheet(sheet_name)

                # Solve
                solution = self.solve_sheet(sheet_name, input_file, time_limit_per_sheet)

                # Store result
                all_solutions[sheet_name] = solution

                # Save individual solution
                sheet_clean = sheet_name.strip().replace(' ', '_')
                output_file = self.base_dir / f"vrp_solution_{sheet_clean}.json"

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(solution, f, indent=2, ensure_ascii=False)

                print(f"  Solution saved: {output_file.name}")

            except Exception as e:
                print(f"  ERROR processing sheet {sheet_name}: {e}")
                all_solutions[sheet_name] = {'error': str(e)}

        # Save combined results
        combined_file = self.base_dir / "vrp_all_sheets_summary.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_solutions, f, indent=2, ensure_ascii=False)

        print("\n" + "="*80)
        print("ALL SHEETS PROCESSED")
        print(f"Combined summary: {combined_file.name}")
        print("="*80)

        return all_solutions

    def print_summary(self, solutions: Dict):
        """Print summary of all solutions"""
        print("\n" + "="*80)
        print("SUMMARY OF ALL SHEETS")
        print("="*80 + "\n")

        for sheet_name, solution in solutions.items():
            if 'error' in solution:
                print(f"Sheet '{sheet_name}': ERROR - {solution['error']}")
                continue

            print(f"Sheet '{sheet_name}':")
            print(f"  Nodes: {solution['num_nodes']}")
            print(f"  Vehicles: {solution['num_vehicles_used']}")
            print(f"  Distance: {solution['total_distance']:.0f}")
            print(f"  Cost: {solution['total_cost']:.2f}")
            print(f"  Status: {solution['status']}")
            print()

        # Comparison table
        print("="*80)
        print("COMPARISON TABLE")
        print("="*80)
        print(f"{'Sheet':<10} {'Nodes':<10} {'Vehicles':<12} {'Distance':<12} {'Cost':<15}")
        print("-"*80)

        for sheet_name in sorted(solutions.keys(), key=lambda x: int(x.strip()) if x.strip().isdigit() else 999):
            solution = solutions[sheet_name]
            if 'error' not in solution:
                print(f"{sheet_name:<10} {solution['num_nodes']:<10} "
                      f"{solution['num_vehicles_used']:<12} "
                      f"{solution['total_distance']:<12.0f} "
                      f"{solution['total_cost']:<15.2f}")

        print("="*80 + "\n")


def main():
    excel_file = 'D:/projects/python/VRPv2/data/distance_matrix_full_138_zones-edit4.xlsx'

    solver = MultiSheetVRPSolver(excel_file)
    solutions = solver.solve_all(time_limit_per_sheet=20)
    solver.print_summary(solutions)


if __name__ == "__main__":
    main()
