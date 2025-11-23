#!/usr/bin/env python3
"""
Detect duplicate connections between the same two POIs.

By default, duplicates are considered for the directed pair (from_id, to_id).
Use --undirected to treat edges as undirected and flag repeated pairs regardless
of direction.

Usage:
    python scripts/find_duplicate_connections.py
    python scripts/find_duplicate_connections.py --connections data/main_data/connections.csv
    python scripts/find_duplicate_connections.py --undirected
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find duplicate POI connections.")
    parser.add_argument(
        "--connections",
        type=Path,
        default=Path("data/main_data/connections.csv"),
        help="Path to connections.csv",
    )
    parser.add_argument(
        "--undirected",
        action="store_true",
        help="Treat connections as undirected when checking duplicates.",
    )
    return parser.parse_args()


def load_connections(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_key(
    row: Dict[str, str],
    undirected: bool,
) -> Tuple[str, str]:
    """Return a tuple key for the connection row."""
    from_id = row["from_id"].strip()
    to_id = row["to_id"].strip()

    if undirected:
        return tuple(sorted((from_id, to_id)))
    return (from_id, to_id)


def find_duplicates(
    connections: List[Dict[str, str]],
    undirected: bool,
) -> Dict[Tuple[str, str], List[Tuple[int, Dict[str, str]]]]:
    """Map duplicate keys to their occurrences with line numbers."""
    seen: Dict[Tuple[str, str], List[Tuple[int, Dict[str, str]]]] = defaultdict(list)

    for idx, row in enumerate(connections, start=2):  # start=2 to account for header line
        key = make_key(row, undirected)
        seen[key].append((idx, row))

    return {k: v for k, v in seen.items() if len(v) > 1}


def print_duplicates(dupes: Dict[Tuple[str, str], List[Tuple[int, Dict[str, str]]]]):
    if not dupes:
        print("✅ No duplicate connections found.")
        return

    print(f"❌ Found {len(dupes)} duplicated connection pair(s):\n")
    for key, rows in sorted(dupes.items()):
        from_id, to_id = key
        print(f"- {from_id} -> {to_id} (occurrences: {len(rows)})")
        for line_no, row in rows:
            base = row.get("base_walk_min", "N/A")
            path_type = row.get("path_type", "N/A")
            print(f"    • line {line_no}: base_walk_min={base}, path_type={path_type}")
        print()


def main():
    args = parse_args()

    if not args.connections.exists():
        print(f"❌ Connections file not found: {args.connections}")
        sys.exit(1)

    connections = load_connections(args.connections)
    duplicates = find_duplicates(connections, undirected=args.undirected)
    print_duplicates(duplicates)

    if duplicates:
        sys.exit(1)


if __name__ == "__main__":
    main()
