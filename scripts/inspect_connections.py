#!/usr/bin/env python3
"""
Connection inspector for Versailles knowledge graph.

Quickly inspect connections to/from any POI to verify against castle plans.

Usage:
    python scripts/inspect_connections.py <poi_id>
    python scripts/inspect_connections.py galerie-des-glaces
    python scripts/inspect_connections.py --search "galerie"
    python scripts/inspect_connections.py --accessibility step_free <poi_id>
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict


def load_pois(pois_path: Path) -> Dict[str, Dict]:
    """Load POI data with full details."""
    pois = {}
    with open(pois_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pois[row['id']] = row
    return pois


def load_connections(connections_path: Path) -> List[Dict]:
    """Load all connections."""
    connections = []
    with open(connections_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            connections.append(row)
    return connections


def find_poi_by_partial_id(pois: Dict[str, Dict], search_term: str) -> List[str]:
    """Find POI IDs that match a partial search term."""
    matches = []
    search_lower = search_term.lower()

    for poi_id, poi_data in pois.items():
        # Check ID
        if search_lower in poi_id.lower():
            matches.append(poi_id)
        # Check name
        elif search_lower in poi_data.get('name', '').lower():
            matches.append(poi_id)

    return sorted(matches)


def get_connections_for_poi(
    poi_id: str,
    connections: List[Dict],
    accessibility: str = "any"
) -> Tuple[List[Dict], List[Dict]]:
    """
    Get outgoing and incoming connections for a POI.

    Returns:
        (outgoing_connections, incoming_connections)
    """
    outgoing = []
    incoming = []

    for conn in connections:
        # Check accessibility filter
        if accessibility == "step_free":
            if conn.get('is_step_free', '').lower() != 'true':
                continue
        elif accessibility == "stroller":
            if conn.get('is_step_free', '').lower() != 'true':
                continue
            if conn.get('stroller_friendly', '').lower() != 'true':
                continue

        # Outgoing: from this POI to others
        if conn['from_id'] == poi_id:
            outgoing.append(conn)

        # Incoming: from others to this POI
        if conn['to_id'] == poi_id:
            incoming.append(conn)

    return outgoing, incoming


def print_poi_details(poi_id: str, pois: Dict[str, Dict]):
    """Print details about a POI."""
    if poi_id not in pois:
        print(f"❌ POI not found: {poi_id}")
        print()
        return False

    poi = pois[poi_id]

    print("=" * 80)
    print(f"POI: {poi_id}")
    print("=" * 80)
    print(f"Name:         {poi.get('name', 'N/A')}")
    print(f"Zone:         {poi.get('zone', 'N/A')}")
    print(f"Category:     {poi.get('category', 'N/A')}")
    print(f"Floor:        {poi.get('floor', 'N/A')}")
    print(f"Wing:         {poi.get('wing', 'N/A')}")
    print(f"Accessibility: {poi.get('accessibility_level', 'N/A')}")
    print(f"Priority:     {poi.get('priority_score', 'N/A')}")
    print(f"Visit Time:   {poi.get('estimated_visit_minutes', 'N/A')} minutes")

    interest_tags = poi.get('interest_tags', '')
    if interest_tags:
        print(f"Tags:         {interest_tags}")

    print()
    return True


def print_connections(
    poi_id: str,
    pois: Dict[str, Dict],
    connections: List[Dict],
    accessibility: str = "any"
):
    """Print all connections for a POI."""
    outgoing, incoming = get_connections_for_poi(poi_id, connections, accessibility)

    # Print summary
    print(f"Connection Summary (accessibility: {accessibility}):")
    print(f"  Outgoing: {len(outgoing)} connections")
    print(f"  Incoming: {len(incoming)} connections")
    print()

    # Print outgoing connections
    if outgoing:
        print("─" * 80)
        print(f"OUTGOING CONNECTIONS ({len(outgoing)}):")
        print("─" * 80)

        for conn in sorted(outgoing, key=lambda c: c['to_id']):
            to_id = conn['to_id']
            to_name = pois.get(to_id, {}).get('name', 'Unknown')

            walk_time = conn.get('base_walk_min', 'N/A')
            is_step_free = conn.get('is_step_free', 'false').lower() == 'true'
            stroller = conn.get('stroller_friendly', 'false').lower() == 'true'
            path_type = conn.get('path_type', 'N/A')

            # Icon based on accessibility
            icon = "♿" if is_step_free else "🚶"
            if stroller:
                icon = "👶"

            print(f"\n  → {to_id}")
            print(f"     Name: {to_name}")
            print(f"     Walk Time: {walk_time} min (base)")
            print(f"     {icon} Step-free: {is_step_free} | Stroller: {stroller}")
            print(f"     Path Type: {path_type}")

            if conn.get('notes'):
                print(f"     Notes: {conn['notes']}")
    else:
        print("─" * 80)
        print("⚠️  NO OUTGOING CONNECTIONS (Dead end!)")
        print("─" * 80)

    print()

    # Print incoming connections
    if incoming:
        print("─" * 80)
        print(f"INCOMING CONNECTIONS ({len(incoming)}):")
        print("─" * 80)

        for conn in sorted(incoming, key=lambda c: c['from_id']):
            from_id = conn['from_id']
            from_name = pois.get(from_id, {}).get('name', 'Unknown')

            walk_time = conn.get('base_walk_min', 'N/A')
            is_step_free = conn.get('is_step_free', 'false').lower() == 'true'
            stroller = conn.get('stroller_friendly', 'false').lower() == 'true'
            path_type = conn.get('path_type', 'N/A')

            icon = "♿" if is_step_free else "🚶"
            if stroller:
                icon = "👶"

            print(f"\n  ← {from_id}")
            print(f"     Name: {from_name}")
            print(f"     Walk Time: {walk_time} min (base)")
            print(f"     {icon} Step-free: {is_step_free} | Stroller: {stroller}")
            print(f"     Path Type: {path_type}")

            if conn.get('notes'):
                print(f"     Notes: {conn['notes']}")
    else:
        print("─" * 80)
        print("⚠️  NO INCOMING CONNECTIONS (Isolated start!)")
        print("─" * 80)

    print()


def check_bidirectionality(
    poi_id: str,
    connections: List[Dict],
    accessibility: str = "any"
):
    """Check if outgoing connections have matching reverse connections."""
    outgoing, _ = get_connections_for_poi(poi_id, connections, accessibility)

    if not outgoing:
        return

    print("─" * 80)
    print("BIDIRECTIONALITY CHECK:")
    print("─" * 80)

    missing_reverse = []

    for conn in outgoing:
        to_id = conn['to_id']

        # Check if reverse connection exists
        reverse_exists = False
        for other_conn in connections:
            if other_conn['from_id'] == to_id and other_conn['to_id'] == poi_id:
                # Check accessibility match
                if accessibility == "step_free":
                    if other_conn.get('is_step_free', '').lower() == 'true':
                        reverse_exists = True
                        break
                elif accessibility == "stroller":
                    if (other_conn.get('is_step_free', '').lower() == 'true' and
                        other_conn.get('stroller_friendly', '').lower() == 'true'):
                        reverse_exists = True
                        break
                else:
                    reverse_exists = True
                    break

        if not reverse_exists:
            missing_reverse.append(to_id)

    if missing_reverse:
        print(f"\n⚠️  Missing {len(missing_reverse)} reverse connection(s):")
        for to_id in missing_reverse:
            print(f"    ❌ No reverse path from {to_id} back to {poi_id}")
    else:
        print("\n✅ All outgoing connections have matching reverse paths")

    print()


def list_all_pois(pois: Dict[str, Dict], zone_filter: str = None):
    """List all POIs, optionally filtered by zone."""
    by_zone = defaultdict(list)

    for poi_id, poi_data in pois.items():
        zone = poi_data.get('zone', 'Unknown')
        by_zone[zone].append((poi_id, poi_data.get('name', 'N/A')))

    print("=" * 80)
    print("ALL POIs BY ZONE")
    print("=" * 80)

    for zone in sorted(by_zone.keys()):
        if zone_filter and zone.lower() != zone_filter.lower():
            continue

        pois_in_zone = sorted(by_zone[zone])
        print(f"\n{zone} ({len(pois_in_zone)} POIs):")
        print("─" * 80)

        for poi_id, name in pois_in_zone:
            # Extract short ID (last part after colon)
            short_id = poi_id.split(':')[-1]
            print(f"  {short_id:40s} - {name}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Inspect POI connections in Versailles knowledge graph"
    )
    parser.add_argument(
        "poi_id",
        nargs="?",
        help="POI ID to inspect (can be partial, e.g., 'galerie-des-glaces' instead of full ID)"
    )
    parser.add_argument(
        "--search", "-s",
        help="Search for POIs by ID or name"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all POIs grouped by zone"
    )
    parser.add_argument(
        "--zone", "-z",
        help="Filter by zone (use with --list)"
    )
    parser.add_argument(
        "--accessibility", "-a",
        choices=["any", "step_free", "stroller"],
        default="any",
        help="Filter connections by accessibility level"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "main_data",
        help="Path to data directory"
    )

    args = parser.parse_args()

    pois_path = args.data_dir / "pois.csv"
    connections_path = args.data_dir / "connections.csv"

    if not pois_path.exists():
        print(f"❌ Error: POIs file not found: {pois_path}")
        sys.exit(1)

    if not connections_path.exists():
        print(f"❌ Error: Connections file not found: {connections_path}")
        sys.exit(1)

    # Load data
    pois = load_pois(pois_path)
    connections = load_connections(connections_path)

    # List mode
    if args.list:
        list_all_pois(pois, args.zone)
        return

    # Search mode
    if args.search:
        matches = find_poi_by_partial_id(pois, args.search)

        if not matches:
            print(f"❌ No POIs found matching '{args.search}'")
            sys.exit(1)

        print(f"Found {len(matches)} POI(s) matching '{args.search}':")
        print()

        for poi_id in matches:
            name = pois[poi_id].get('name', 'N/A')
            print(f"  {poi_id}")
            print(f"    → {name}")

        print()

        if len(matches) == 1:
            print("Showing details for the only match:")
            print()
            args.poi_id = matches[0]
        else:
            print("Use one of the IDs above to inspect connections.")
            return

    # Inspect mode
    if not args.poi_id:
        print("❌ Error: Please provide a POI ID to inspect")
        print()
        print("Examples:")
        print("  python scripts/inspect_connections.py galerie-des-glaces")
        print("  python scripts/inspect_connections.py --search galerie")
        print("  python scripts/inspect_connections.py --list")
        sys.exit(1)

    # Try to find POI (handle partial IDs)
    poi_id = args.poi_id

    if poi_id not in pois:
        # Try to find by partial match
        matches = find_poi_by_partial_id(pois, poi_id)

        if not matches:
            print(f"❌ POI not found: {poi_id}")
            print()
            print("Try searching with: --search <term>")
            sys.exit(1)

        if len(matches) > 1:
            print(f"❌ Ambiguous POI ID '{poi_id}'. Multiple matches:")
            print()
            for match in matches:
                print(f"  {match}")
            print()
            print("Please use the full ID or be more specific.")
            sys.exit(1)

        poi_id = matches[0]

    # Print details and connections
    if print_poi_details(poi_id, pois):
        print_connections(poi_id, pois, connections, args.accessibility)
        check_bidirectionality(poi_id, connections, args.accessibility)


if __name__ == "__main__":
    main()
