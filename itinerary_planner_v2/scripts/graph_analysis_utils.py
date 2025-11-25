"""
Graph analysis utilities for validating TSP algorithm requirements.

This module provides graph theory algorithms to validate that the knowledge graph
meets the connectivity and reachability requirements for the TSP-based itinerary planner.
"""

import csv
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional


@dataclass
class GraphEdge:
    """Represents a directed edge in the graph."""
    from_id: str
    to_id: str
    is_step_free: bool
    stroller_friendly: bool
    base_walk_min: float
    family_walk_min: float
    elder_walk_min: float


@dataclass
class POI:
    """Represents a point of interest."""
    poi_id: str
    name: str
    zone: str
    category: str
    accessibility_level: str


class DirectedGraph:
    """Directed graph structure with accessibility-aware edge filtering."""

    def __init__(self):
        self.adjacency: Dict[str, List[GraphEdge]] = defaultdict(list)
        self.nodes: Set[str] = set()
        self.reverse_adjacency: Dict[str, List[GraphEdge]] = defaultdict(list)

    def add_edge(self, edge: GraphEdge):
        """Add a directed edge to the graph."""
        self.adjacency[edge.from_id].append(edge)
        self.reverse_adjacency[edge.to_id].append(edge)
        self.nodes.add(edge.from_id)
        self.nodes.add(edge.to_id)

    def get_neighbors(self, node_id: str, accessibility: str = "any") -> List[str]:
        """Get reachable neighbors from a node with accessibility filter."""
        edges = self.adjacency.get(node_id, [])
        neighbors = []

        for edge in edges:
            if accessibility == "step_free" and not edge.is_step_free:
                continue
            if accessibility == "stroller" and not (edge.is_step_free and edge.stroller_friendly):
                continue
            neighbors.append(edge.to_id)

        return neighbors

    def get_in_degree(self, node_id: str, accessibility: str = "any") -> int:
        """Get in-degree of a node (number of incoming edges)."""
        edges = self.reverse_adjacency.get(node_id, [])

        if accessibility == "any":
            return len(edges)

        count = 0
        for edge in edges:
            if accessibility == "step_free" and not edge.is_step_free:
                continue
            if accessibility == "stroller" and not (edge.is_step_free and edge.stroller_friendly):
                continue
            count += 1

        return count

    def get_out_degree(self, node_id: str, accessibility: str = "any") -> int:
        """Get out-degree of a node (number of outgoing edges)."""
        return len(self.get_neighbors(node_id, accessibility))

    def has_edge(self, from_id: str, to_id: str, accessibility: str = "any") -> bool:
        """Check if an edge exists between two nodes."""
        neighbors = self.get_neighbors(from_id, accessibility)
        return to_id in neighbors


def load_graph_from_csv(
    connections_path: Path,
    accessibility: str = "any"
) -> DirectedGraph:
    """
    Load graph from connections.csv file.

    Args:
        connections_path: Path to connections.csv
        accessibility: Filter level ("any", "step_free", "stroller")

    Returns:
        DirectedGraph instance
    """
    graph = DirectedGraph()

    with open(connections_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            edge = GraphEdge(
                from_id=row['from_id'],  # Column name is 'from_id', not 'from_poi_id'
                to_id=row['to_id'],  # Column name is 'to_id', not 'to_poi_id'
                is_step_free=row['is_step_free'].lower() == 'true',
                stroller_friendly=row['stroller_friendly'].lower() == 'true',
                base_walk_min=float(row['base_walk_min']),
                family_walk_min=float(row['family_walk_min']),
                elder_walk_min=float(row['elder_walk_min'])
            )
            graph.add_edge(edge)

    return graph


def load_pois_from_csv(pois_path: Path) -> Dict[str, POI]:
    """
    Load POIs from pois.csv file.

    Returns:
        Dictionary mapping POI ID to POI object
    """
    pois = {}

    with open(pois_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            poi = POI(
                poi_id=row['id'],  # Column name is 'id', not 'poi_id'
                name=row['name'],
                zone=row['zone'],
                category=row['category'],
                accessibility_level=row.get('accessibility_level', 'full')
            )
            pois[poi.poi_id] = poi

    return pois


def bfs_reachable(graph: DirectedGraph, start_id: str, accessibility: str = "any") -> Set[str]:
    """
    Find all nodes reachable from start node using BFS.

    Args:
        graph: DirectedGraph instance
        start_id: Starting node ID
        accessibility: Accessibility filter

    Returns:
        Set of reachable node IDs (including start)
    """
    if start_id not in graph.nodes:
        return set()

    visited = {start_id}
    queue = deque([start_id])

    while queue:
        current = queue.popleft()
        neighbors = graph.get_neighbors(current, accessibility)

        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited


def find_unreachable_nodes(
    graph: DirectedGraph,
    start_id: str,
    target_nodes: Set[str],
    accessibility: str = "any"
) -> Set[str]:
    """
    Find nodes in target_nodes that are unreachable from start.

    Args:
        graph: DirectedGraph instance
        start_id: Starting node ID
        target_nodes: Set of node IDs that should be reachable
        accessibility: Accessibility filter

    Returns:
        Set of unreachable node IDs
    """
    reachable = bfs_reachable(graph, start_id, accessibility)
    return target_nodes - reachable


def find_shortest_path_length(
    graph: DirectedGraph,
    start_id: str,
    end_id: str,
    accessibility: str = "any",
    max_length: int = 20
) -> Optional[int]:
    """
    Find shortest path length between two nodes using BFS.

    Args:
        graph: DirectedGraph instance
        start_id: Starting node ID
        end_id: Ending node ID
        accessibility: Accessibility filter
        max_length: Maximum path length to search

    Returns:
        Path length if found, None if no path or exceeds max_length
    """
    if start_id not in graph.nodes or end_id not in graph.nodes:
        return None

    if start_id == end_id:
        return 0

    visited = {start_id}
    queue = deque([(start_id, 0)])

    while queue:
        current, distance = queue.popleft()

        if distance >= max_length:
            continue

        neighbors = graph.get_neighbors(current, accessibility)

        for neighbor in neighbors:
            if neighbor == end_id:
                return distance + 1

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

    return None


def find_dead_ends(
    graph: DirectedGraph,
    exclude_nodes: Set[str],
    accessibility: str = "any"
) -> Dict[str, str]:
    """
    Find nodes with out-degree = 0 (dead ends).

    Args:
        graph: DirectedGraph instance
        exclude_nodes: Nodes to exclude (e.g., exit nodes)
        accessibility: Accessibility filter

    Returns:
        Dict mapping dead-end node IDs to reason
    """
    dead_ends = {}

    for node_id in graph.nodes:
        if node_id in exclude_nodes:
            continue

        out_degree = graph.get_out_degree(node_id, accessibility)
        if out_degree == 0:
            dead_ends[node_id] = "no_outgoing_edges"

    return dead_ends


def find_isolated_starts(
    graph: DirectedGraph,
    exclude_nodes: Set[str],
    accessibility: str = "any"
) -> Dict[str, str]:
    """
    Find nodes with in-degree = 0 (isolated starts).

    Args:
        graph: DirectedGraph instance
        exclude_nodes: Nodes to exclude (e.g., entrance nodes)
        accessibility: Accessibility filter

    Returns:
        Dict mapping isolated node IDs to reason
    """
    isolated = {}

    for node_id in graph.nodes:
        if node_id in exclude_nodes:
            continue

        in_degree = graph.get_in_degree(node_id, accessibility)
        if in_degree == 0:
            isolated[node_id] = "no_incoming_edges"

    return isolated


def check_bidirectionality(
    graph: DirectedGraph,
    node_filter: Optional[Set[str]] = None,
    accessibility: str = "any"
) -> List[Tuple[str, str]]:
    """
    Find edges that don't have a reverse edge (one-way connections).

    Args:
        graph: DirectedGraph instance
        node_filter: Only check edges between nodes in this set (if provided)
        accessibility: Accessibility filter

    Returns:
        List of (from_id, to_id) tuples representing missing reverse edges
    """
    missing_reverse = []

    for from_id in graph.nodes:
        if node_filter and from_id not in node_filter:
            continue

        neighbors = graph.get_neighbors(from_id, accessibility)

        for to_id in neighbors:
            if node_filter and to_id not in node_filter:
                continue

            # Check if reverse edge exists
            if not graph.has_edge(to_id, from_id, accessibility):
                missing_reverse.append((from_id, to_id))

    return missing_reverse


def find_strongly_connected_components(graph: DirectedGraph, accessibility: str = "any") -> List[Set[str]]:
    """
    Find strongly connected components using Tarjan's algorithm.

    A strongly connected component is a maximal set of nodes where every node
    is reachable from every other node.

    Args:
        graph: DirectedGraph instance
        accessibility: Accessibility filter

    Returns:
        List of strongly connected component sets
    """
    index_counter = [0]
    stack = []
    lowlinks = {}
    index = {}
    on_stack = set()
    components = []

    def strongconnect(node: str):
        index[node] = index_counter[0]
        lowlinks[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
        on_stack.add(node)

        neighbors = graph.get_neighbors(node, accessibility)
        for neighbor in neighbors:
            if neighbor not in index:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], index[neighbor])

        if lowlinks[node] == index[node]:
            component = set()
            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.add(w)
                if w == node:
                    break
            components.append(component)

    for node in graph.nodes:
        if node not in index:
            strongconnect(node)

    return components


def get_zone_from_poi_id(poi_id: str) -> str:
    """
    Extract zone from POI ID.

    Args:
        poi_id: POI identifier (e.g., "versailles:Room:grande-chambre-du-roi")

    Returns:
        Zone string ("Room", "Garden", "Trianon", "Park", "Facility")
    """
    parts = poi_id.split(':')
    if len(parts) >= 2:
        return parts[1]
    return "Unknown"


def is_castle_zone(poi_id: str) -> bool:
    """Check if POI is in the castle (one-way tour) zone."""
    zone = get_zone_from_poi_id(poi_id)
    return zone == "Room"


def is_garden_zone(poi_id: str) -> bool:
    """Check if POI is in gardens/park/trianon (bidirectional) zones."""
    zone = get_zone_from_poi_id(poi_id)
    return zone in ("Garden", "Park", "Trianon", "Facility")


def verify_hamiltonian_path(
    graph: DirectedGraph,
    nodes: Set[str],
    accessibility: str = "any"
) -> Tuple[bool, Optional[List[str]]]:
    """
    Verify that a Hamiltonian path exists through all nodes.

    A Hamiltonian path visits each node exactly once.
    Uses backtracking for small node sets (<= 15 nodes).

    Args:
        graph: DirectedGraph instance
        nodes: Set of nodes to visit
        accessibility: Accessibility filter

    Returns:
        (exists, path) where exists is True if path found, path is the node sequence
    """
    if len(nodes) > 15:
        # Too expensive to compute, return unknown
        return (False, None)

    nodes_list = list(nodes)

    def backtrack(path: List[str], remaining: Set[str]) -> Optional[List[str]]:
        if not remaining:
            return path

        current = path[-1] if path else None

        for next_node in remaining:
            # If path is empty, try each node as start
            if current is None:
                result = backtrack([next_node], remaining - {next_node})
                if result:
                    return result
            # Otherwise check if edge exists
            elif graph.has_edge(current, next_node, accessibility):
                result = backtrack(path + [next_node], remaining - {next_node})
                if result:
                    return result

        return None

    path = backtrack([], nodes)
    return (path is not None, path)
