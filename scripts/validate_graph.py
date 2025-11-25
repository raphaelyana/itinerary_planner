#!/usr/bin/env python3
"""
Graph validation script for Versailles knowledge graph.

Validates that the graph meets all requirements for the TSP-based itinerary planner:
- Zone-specific connectivity (directed castle tour, bidirectional gardens)
- Entrance reachability
- Accessibility subgraph connectivity
- Path length limits
- Dead-end detection

Usage:
    python scripts/validate_graph.py
    python scripts/validate_graph.py --output-json report.json
    python scripts/validate_graph.py --accessibility step_free
"""

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Set, Optional

from graph_analysis_utils import (
    load_graph_from_csv,
    load_pois_from_csv,
    bfs_reachable,
    find_unreachable_nodes,
    find_shortest_path_length,
    find_dead_ends,
    find_isolated_starts,
    check_bidirectionality,
    find_strongly_connected_components,
    get_zone_from_poi_id,
    is_castle_zone,
    is_garden_zone,
    verify_hamiltonian_path,
)


# Constants
DEFAULT_ENTRANCE = "versailles:Castle:cour-dhonneur"
TRIANON_ENTRANCE = "versailles:Trianon:entree-grand-trianon"
EXIT_NODE = "versailles:Castle:cour-dhonneur-sortie"
MAX_PATH_LENGTH = 20  # APOC maxLevel limit in planner_utils.py


@dataclass
class ValidationIssue:
    """Represents a validation issue found in the graph."""
    severity: str  # "error", "warning", "info"
    rule: str
    message: str
    details: Optional[Dict] = None


@dataclass
class ValidationReport:
    """Complete validation report."""
    total_pois: int
    total_connections: int
    total_issues: int
    errors: int
    warnings: int
    issues: List[ValidationIssue]
    zone_stats: Dict[str, int]
    accessibility_level: str


class GraphValidator:
    """Validates graph structure for TSP algorithm compatibility."""

    def __init__(
        self,
        pois_path: Path,
        connections_path: Path,
        accessibility: str = "any",
        bidirectional_zones: Optional[Set[str]] = None,
        allow_long_paths: bool = False,
    ):
        self.pois_path = pois_path
        self.connections_path = connections_path
        self.accessibility = accessibility
        self.bidirectional_zones = bidirectional_zones or {"Garden", "Park", "Trianon"}
        self.allow_long_paths = allow_long_paths
        self.issues: List[ValidationIssue] = []

        # Load data
        self.pois = load_pois_from_csv(pois_path)
        self.graph = load_graph_from_csv(connections_path, accessibility)

        def normalize_zone(raw: Optional[str]) -> Optional[str]:
            if not raw:
                return None
            raw_norm = raw.strip()
            if raw_norm.lower() == "gardens":
                return "Garden"
            return raw_norm

        # Categorize POIs by declared zone (not by ID prefix) to match CSV
        self.castle_pois = {
            pid for pid, poi in self.pois.items()
            if normalize_zone(getattr(poi, "zone", None)) == "Castle"
        }
        self.bidirectional_pois = {
            pid for pid, poi in self.pois.items()
            if normalize_zone(getattr(poi, "zone", None)) in self.bidirectional_zones
        }

    def add_issue(self, severity: str, rule: str, message: str, details: Optional[Dict] = None):
        """Add a validation issue."""
        self.issues.append(ValidationIssue(
            severity=severity,
            rule=rule,
            message=message,
            details=details or {}
        ))

    def validate_all(self) -> ValidationReport:
        """Run all validation checks."""
        print(f"Validating graph with accessibility level: {self.accessibility}")
        print(f"Total POIs: {len(self.pois)}")
        print(f"Total graph nodes: {len(self.graph.nodes)}")
        print(f"Castle POIs (directed): {len(self.castle_pois)}")
        print(f"Garden POIs (bidirectional): {len(self.bidirectional_pois)} (zones: {', '.join(sorted(self.bidirectional_zones))})")
        print()

        # Run all validation rules
        self.validate_orphaned_pois()
        self.validate_entrance_reachability()
        self.validate_zone_bidirectionality()
        self.validate_dead_ends()
        self.validate_path_lengths()
        self.validate_accessibility_connectivity()
        self.validate_zone_bridges()
        self.validate_castle_tour()

        # Generate report
        zone_stats = {}
        for poi in self.pois.values():
            zone_stats[poi.zone] = zone_stats.get(poi.zone, 0) + 1

        errors = sum(1 for issue in self.issues if issue.severity == "error")
        warnings = sum(1 for issue in self.issues if issue.severity == "warning")

        return ValidationReport(
            total_pois=len(self.pois),
            total_connections=len(self.graph.adjacency),
            total_issues=len(self.issues),
            errors=errors,
            warnings=warnings,
            issues=self.issues,
            zone_stats=zone_stats,
            accessibility_level=self.accessibility
        )

    def validate_orphaned_pois(self):
        """Check that all POIs appear in the connection graph."""
        print("[1/8] Checking for orphaned POIs...")

        poi_ids = set(self.pois.keys())
        graph_nodes = self.graph.nodes
        orphaned = poi_ids - graph_nodes

        if orphaned:
            self.add_issue(
                severity="error",
                rule="orphaned_pois",
                message=f"Found {len(orphaned)} POI(s) with no connections",
                details={"orphaned_pois": sorted(list(orphaned))}
            )
            print(f"  ❌ FAILED: {len(orphaned)} orphaned POIs")
        else:
            print(f"  ✅ PASSED: All POIs have connections")

    def validate_entrance_reachability(self):
        """Check that all POIs are reachable from entrance nodes."""
        print("[2/8] Checking entrance reachability...")

        # Check main entrance
        reachable_from_main = bfs_reachable(self.graph, DEFAULT_ENTRANCE, self.accessibility)
        unreachable_from_main = set(self.pois.keys()) - reachable_from_main

        if unreachable_from_main:
            # Check if they're in Trianon zone (may be reachable from Trianon entrance)
            trianon_unreachable = {pid for pid in unreachable_from_main if "Trianon" not in pid}

            if trianon_unreachable:
                self.add_issue(
                    severity="error",
                    rule="entrance_reachability",
                    message=f"Found {len(trianon_unreachable)} POI(s) unreachable from main entrance",
                    details={
                        "entrance": DEFAULT_ENTRANCE,
                        "unreachable_pois": sorted(list(trianon_unreachable))
                    }
                )
                print(f"  ❌ FAILED: {len(trianon_unreachable)} POIs unreachable from main entrance")

        # Check Trianon entrance
        if TRIANON_ENTRANCE in self.graph.nodes:
            reachable_from_trianon = bfs_reachable(self.graph, TRIANON_ENTRANCE, self.accessibility)
            trianon_pois = {pid for pid in self.pois.keys() if "Trianon" in pid}
            unreachable_trianon = trianon_pois - reachable_from_trianon

            if unreachable_trianon:
                self.add_issue(
                    severity="error",
                    rule="entrance_reachability",
                    message=f"Found {len(unreachable_trianon)} Trianon POI(s) unreachable from Trianon entrance",
                    details={
                        "entrance": TRIANON_ENTRANCE,
                        "unreachable_pois": sorted(list(unreachable_trianon))
                    }
                )
                print(f"  ❌ FAILED: {len(unreachable_trianon)} Trianon POIs unreachable from Trianon entrance")

        if not unreachable_from_main and not (TRIANON_ENTRANCE in self.graph.nodes and unreachable_trianon):
            print(f"  ✅ PASSED: All POIs reachable from appropriate entrances")

    def validate_zone_bidirectionality(self):
        """Check bidirectionality for garden zones (castle is allowed to be directed)."""
        print("[3/8] Checking zone-specific bidirectionality...")

        if not self.bidirectional_zones:
            print("  ⏭️  SKIPPED: Bidirectionality check disabled (no zones configured)")
            return

        # Only check configured zones for bidirectionality
        missing_reverse = check_bidirectionality(self.graph, self.bidirectional_pois, self.accessibility)

        if missing_reverse:
            self.add_issue(
                severity="error",
                rule="bidirectionality",
                message=f"Found {len(missing_reverse)} one-way edge(s) in bidirectional zones ({', '.join(sorted(self.bidirectional_zones))})",
                details={
                    "missing_reverse_edges": [
                        {"from": from_id, "to": to_id}
                        for from_id, to_id in missing_reverse
                    ]
                }
            )
            print(f"  ❌ FAILED: {len(missing_reverse)} missing reverse edges in garden zones")
        else:
            print(f"  ✅ PASSED: Garden zones have bidirectional connections")

        # Info: Report on castle directionality
        castle_edges = [(f, t) for f, t in check_bidirectionality(self.graph, self.castle_pois, self.accessibility)]
        if castle_edges:
            self.add_issue(
                severity="info",
                rule="castle_directionality",
                message=f"Castle has {len(castle_edges)} one-way edges (expected for museum tour)",
                details={"one_way_count": len(castle_edges)}
            )
            print(f"  ℹ️  INFO: Castle has {len(castle_edges)} directed edges (expected)")

    def validate_dead_ends(self):
        """Check for dead-end nodes (except exit nodes)."""
        print("[4/8] Checking for dead ends...")

        exclude = {EXIT_NODE}
        dead_ends = find_dead_ends(self.graph, exclude, self.accessibility)

        # Separate castle dead ends (expected for tour end) from garden dead ends (error)
        castle_dead_ends = {nid: reason for nid, reason in dead_ends.items() if is_castle_zone(nid)}
        garden_dead_ends = {nid: reason for nid, reason in dead_ends.items() if is_garden_zone(nid)}

        if garden_dead_ends:
            self.add_issue(
                severity="error",
                rule="dead_ends",
                message=f"Found {len(garden_dead_ends)} dead-end node(s) in garden zones",
                details={"dead_end_nodes": sorted(list(garden_dead_ends.keys()))}
            )
            print(f"  ❌ FAILED: {len(garden_dead_ends)} dead ends in garden zones")

        if castle_dead_ends:
            self.add_issue(
                severity="info",
                rule="castle_tour_end",
                message=f"Castle has {len(castle_dead_ends)} dead-end node(s) (expected for tour exit)",
                details={"tour_end_nodes": sorted(list(castle_dead_ends.keys()))}
            )
            print(f"  ℹ️  INFO: Castle has {len(castle_dead_ends)} tour end point(s)")

        if not dead_ends:
            print(f"  ✅ PASSED: No unexpected dead ends found")

        # Check for isolated starts (no incoming edges) except entrances
        exclude_starts = {DEFAULT_ENTRANCE, TRIANON_ENTRANCE}
        isolated = find_isolated_starts(self.graph, exclude_starts, self.accessibility)

        if isolated:
            self.add_issue(
                severity="warning",
                rule="isolated_starts",
                message=f"Found {len(isolated)} node(s) with no incoming edges",
                details={"isolated_nodes": sorted(list(isolated.keys()))}
            )
            print(f"  ⚠️  WARNING: {len(isolated)} isolated start nodes")

    def validate_path_lengths(self):
        """Check that all paths are within APOC maxLevel limit."""
        print("[5/8] Checking maximum path lengths...")

        if self.allow_long_paths:
            print("  ⏭️  SKIPPED: Path length check disabled (--allow-long-paths)")
            return

        long_paths = []
        poi_ids = list(self.pois.keys())
        max_length_found = 0

        # Sample check: test paths from entrances to all POIs
        for entrance in [DEFAULT_ENTRANCE, TRIANON_ENTRANCE]:
            if entrance not in self.graph.nodes:
                continue

            for poi_id in poi_ids:
                if poi_id == entrance:
                    continue

                length = find_shortest_path_length(
                    self.graph, entrance, poi_id, self.accessibility, MAX_PATH_LENGTH + 1
                )

                if length is not None:
                    max_length_found = max(max_length_found, length)

                    if length > MAX_PATH_LENGTH:
                        long_paths.append({
                            "from": entrance,
                            "to": poi_id,
                            "length": length
                        })

        if long_paths:
            self.add_issue(
                severity="error",
                rule="path_length",
                message=f"Found {len(long_paths)} path(s) exceeding APOC limit of {MAX_PATH_LENGTH} hops",
                details={"long_paths": long_paths}
            )
            print(f"  ❌ FAILED: {len(long_paths)} paths exceed {MAX_PATH_LENGTH} hops")
        else:
            print(f"  ✅ PASSED: All paths within {MAX_PATH_LENGTH} hop limit (max found: {max_length_found})")

    def validate_accessibility_connectivity(self):
        """Check that accessibility-filtered graphs remain connected."""
        print("[6/8] Checking accessibility-filtered connectivity...")

        if self.accessibility == "any":
            print(f"  ⏭️  SKIPPED: No filtering applied for 'any' accessibility")
            return

        # Check which POIs become unreachable with accessibility filter
        all_pois = set(self.pois.keys())

        # Load unfiltered graph
        unfiltered_graph = load_graph_from_csv(self.connections_path, "any")

        # Find POIs that were reachable but now aren't
        reachable_unfiltered = bfs_reachable(unfiltered_graph, DEFAULT_ENTRANCE, "any")
        reachable_filtered = bfs_reachable(self.graph, DEFAULT_ENTRANCE, self.accessibility)

        lost_connectivity = reachable_unfiltered - reachable_filtered

        if lost_connectivity:
            # Check if these POIs have accessibility_level="none"
            missing_metadata = []
            for poi_id in lost_connectivity:
                poi = self.pois.get(poi_id)
                if poi and poi.accessibility_level != "none":
                    missing_metadata.append(poi_id)

            if missing_metadata:
                self.add_issue(
                    severity="warning",
                    rule="accessibility_connectivity",
                    message=f"Found {len(missing_metadata)} POI(s) unreachable with {self.accessibility} filter but not marked accessibility_level='none'",
                    details={
                        "unreachable_pois": sorted(missing_metadata),
                        "recommendation": "Update POI metadata or add step-free connections"
                    }
                )
                print(f"  ⚠️  WARNING: {len(missing_metadata)} POIs lost connectivity with {self.accessibility} filter")

            self.add_issue(
                severity="info",
                rule="accessibility_impact",
                message=f"Accessibility filter '{self.accessibility}' reduces reachable POIs by {len(lost_connectivity)}",
                details={"lost_count": len(lost_connectivity)}
            )
            print(f"  ℹ️  INFO: {len(lost_connectivity)} POIs require non-{self.accessibility} paths")

        else:
            print(f"  ✅ PASSED: No connectivity lost with {self.accessibility} filter")

    def validate_zone_bridges(self):
        """Check that connections exist between zones."""
        print("[7/8] Checking zone bridge connections...")

        # Key bridge connections to verify
        bridges_to_check = [
            ("Castle", "Gardens", "versailles:Room:", "versailles:Garden:"),
            ("Gardens", "Trianon", "versailles:Garden:", "versailles:Trianon:"),
            ("Gardens", "Park", "versailles:Garden:", "versailles:Park:"),
        ]

        missing_bridges = []

        for zone1, zone2, prefix1, prefix2 in bridges_to_check:
            # Find POIs in each zone
            zone1_pois = {pid for pid in self.pois if pid.startswith(prefix1)}
            zone2_pois = {pid for pid in self.pois if pid.startswith(prefix2)}

            if not zone1_pois or not zone2_pois:
                continue

            # Check if any POI in zone1 can reach any POI in zone2
            bridge_exists = False
            for poi1 in zone1_pois:
                reachable = bfs_reachable(self.graph, poi1, self.accessibility)
                if reachable & zone2_pois:
                    bridge_exists = True
                    break

            if not bridge_exists:
                missing_bridges.append(f"{zone1} → {zone2}")

        if missing_bridges:
            self.add_issue(
                severity="error",
                rule="zone_bridges",
                message=f"Missing zone bridge connections: {', '.join(missing_bridges)}",
                details={"missing_bridges": missing_bridges}
            )
            print(f"  ❌ FAILED: Missing bridges: {', '.join(missing_bridges)}")
        else:
            print(f"  ✅ PASSED: All zone bridges exist")

    def validate_castle_tour(self):
        """Validate that castle tour has valid structure (Hamiltonian path)."""
        print("[8/8] Checking castle tour structure...")

        if not self.castle_pois:
            print(f"  ⏭️  SKIPPED: No castle POIs found")
            return

        # For large castle POI sets, just check basic connectivity
        if len(self.castle_pois) > 15:
            castle_nodes = self.castle_pois

            def castle_neighbors(node_id: str) -> List[str]:
                neighbors = []
                for edge in self.graph.adjacency.get(node_id, []):
                    if edge.to_id not in castle_nodes:
                        continue
                    if self.accessibility == "step_free" and not edge.is_step_free:
                        continue
                    if self.accessibility == "stroller" and not (edge.is_step_free and edge.stroller_friendly):
                        continue
                    neighbors.append(edge.to_id)
                return neighbors

            def castle_in_degree(node_id: str) -> int:
                count = 0
                for edge in self.graph.reverse_adjacency.get(node_id, []):
                    if edge.from_id not in castle_nodes:
                        continue
                    if self.accessibility == "step_free" and not edge.is_step_free:
                        continue
                    if self.accessibility == "stroller" and not (edge.is_step_free and edge.stroller_friendly):
                        continue
                    count += 1
                return count

            def castle_out_degree(node_id: str) -> int:
                return len(castle_neighbors(node_id))

            candidate_starts = [node for node in castle_nodes if castle_in_degree(node) == 0]
            if not candidate_starts:
                candidate_starts = list(castle_nodes)

            def reachable_from(start: str) -> Set[str]:
                visited = {start}
                queue = [start]
                while queue:
                    current = queue.pop(0)
                    for neighbor in castle_neighbors(current):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                return visited

            covering_start = None
            coverage_map: Dict[str, Set[str]] = {}

            for start in candidate_starts:
                reach = reachable_from(start)
                coverage_map[start] = reach
                if castle_nodes.issubset(reach):
                    covering_start = start
                    break

            if covering_start:
                end_candidates = [node for node in castle_nodes if castle_out_degree(node) == 0]
                print(f"  ✅ PASSED: Castle tour has a covering start node ({covering_start})")
                if end_candidates:
                    print(f"         Possible tour ends (out-degree 0): {', '.join(sorted(end_candidates)[:5])}")
            else:
                best_start, best_reach = None, set()
                if coverage_map:
                    best_start, best_reach = max(coverage_map.items(), key=lambda kv: len(kv[1]))
                missing = sorted(list(castle_nodes - best_reach))
                self.add_issue(
                    severity="error",
                    rule="castle_tour",
                    message="Castle tour does not have a single-direction covering path from any start",
                    details={
                        "best_start": best_start,
                        "reachable_from_best_start": len(best_reach),
                        "missing_from_best_start": missing,
                    }
                )
                print(f"  ❌ FAILED: No start node reaches all castle rooms (best start '{best_start}' reaches {len(best_reach)}/{len(castle_nodes)})")
        else:
            # Small enough to verify Hamiltonian path
            has_path, path = verify_hamiltonian_path(self.graph, self.castle_pois, self.accessibility)

            if not has_path:
                self.add_issue(
                    severity="error",
                    rule="castle_tour",
                    message="No Hamiltonian path through castle rooms",
                    details={"castle_pois_count": len(self.castle_pois)}
                )
                print(f"  ❌ FAILED: No valid tour path through {len(self.castle_pois)} castle rooms")
            else:
                print(f"  ✅ PASSED: Valid Hamiltonian path exists through {len(self.castle_pois)} castle rooms")


def print_report(report: ValidationReport):
    """Print validation report to console."""
    print("\n" + "="*70)
    print("VALIDATION REPORT")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Accessibility Level: {report.accessibility_level}")
    print(f"\nGraph Statistics:")
    print(f"  Total POIs: {report.total_pois}")
    print(f"  Zone Distribution:")
    for zone, count in sorted(report.zone_stats.items()):
        print(f"    - {zone}: {count}")
    print(f"\nValidation Results:")
    print(f"  Total Issues: {report.total_issues}")
    print(f"  Errors: {report.errors}")
    print(f"  Warnings: {report.warnings}")
    print(f"  Info: {report.total_issues - report.errors - report.warnings}")

    if report.issues:
        print(f"\nIssues Found:")
        print("-" * 70)

        # Group by severity
        errors = [i for i in report.issues if i.severity == "error"]
        warnings = [i for i in report.issues if i.severity == "warning"]
        infos = [i for i in report.issues if i.severity == "info"]

        for severity_label, issues in [("ERRORS", errors), ("WARNINGS", warnings), ("INFO", infos)]:
            if not issues:
                continue

            print(f"\n{severity_label}:")
            for issue in issues:
                icon = "❌" if issue.severity == "error" else "⚠️" if issue.severity == "warning" else "ℹ️"
                print(f"\n  {icon} [{issue.rule}]")
                print(f"     {issue.message}")

                if issue.details and len(str(issue.details)) < 200:
                    print(f"     Details: {issue.details}")
                elif issue.details:
                    print(f"     Details: (see JSON output for full details)")

    print("\n" + "="*70)

    if report.errors == 0:
        print("✅ VALIDATION PASSED - Graph is ready for TSP algorithms")
    else:
        print(f"❌ VALIDATION FAILED - {report.errors} error(s) must be fixed")

    print("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate Versailles knowledge graph for TSP algorithms"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "main_data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--accessibility",
        choices=["any", "step_free", "stroller"],
        default="any",
        help="Accessibility filter to validate"
    )
    parser.add_argument(
        "--bidirectional-zones",
        type=str,
        default="Garden,Park,Trianon",
        help="Comma-separated list of zones to enforce bidirectionality on (default: Garden,Park,Trianon)"
    )
    parser.add_argument(
        "--allow-long-paths",
        action="store_true",
        help="Skip failing the check for paths longer than the APOC maxLevel (20 hops)."
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output validation report as JSON to this file"
    )

    args = parser.parse_args()

    pois_path = args.data_dir / "pois.csv"
    connections_path = args.data_dir / "connections.csv"

    # Check files exist
    if not pois_path.exists():
        print(f"Error: POIs file not found: {pois_path}")
        sys.exit(1)

    if not connections_path.exists():
        print(f"Error: Connections file not found: {connections_path}")
        sys.exit(1)

    # Run validation
    bidir_zones = set()
    for z in args.bidirectional_zones.split(","):
        z = z.strip()
        if not z:
            continue
        if z.lower() == "gardens":
            z = "Garden"
        bidir_zones.add(z)

    validator = GraphValidator(
        pois_path,
        connections_path,
        args.accessibility,
        bidirectional_zones=bidir_zones,
        allow_long_paths=args.allow_long_paths,
    )
    report = validator.validate_all()

    # Print report
    print_report(report)

    # Save JSON if requested
    if args.output_json:
        output_data = {
            "total_pois": report.total_pois,
            "total_connections": report.total_connections,
            "total_issues": report.total_issues,
            "errors": report.errors,
            "warnings": report.warnings,
            "accessibility_level": report.accessibility_level,
            "zone_stats": report.zone_stats,
            "issues": [asdict(issue) for issue in report.issues]
        }

        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"JSON report saved to: {args.output_json}")

    # Exit with error code if validation failed
    sys.exit(0 if report.errors == 0 else 1)


if __name__ == "__main__":
    main()
