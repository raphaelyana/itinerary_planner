from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Literal, Optional

# Add parent directory to path for imports when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase, Session
from scripts.planner_utils import normalize_neo4j_uri

# Override a few overlong visit durations to keep the “main castle track” in the
# 2h–2h30 range while retaining generous dwell times for highlights.
VISIT_TIME_OVERRIDES: Dict[str, int] = {
    "versailles:Room:galeries-de-lhistoire": 45,
    "versailles:Room:salles-louis-xiv": 35,
    "versailles:Room:galerie-des-batailles": 15,
    "versailles:Room:salle-1792": 15,
}

# Import validation utilities
try:
    from validate_graph import GraphValidator
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False
    print("Warning: Graph validation not available (validate_graph.py not found)")
    print("Continuing without pre-ingestion validation...")
    print()


IngestMode = Literal["all", "pois_only", "connections_only"]


def read_csv(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"} if value else False


def parse_float(value: str) -> float:
    return float(value) if value else 0.0


def parse_int(value: str) -> int:
    return int(float(value)) if value else 0


def split_tags(raw: str) -> List[str]:
    if not raw:
        return []
    return [tag.strip() for tag in raw.split(";") if tag.strip()]


def prepare_poi(row: Dict[str, str]) -> Tuple[str, str, Dict[str, object]]:
    poi_id = row["id"]
    category = row["category"]
    est_minutes = parse_int(row.get("estimated_visit_minutes", ""))
    if poi_id in VISIT_TIME_OVERRIDES:
        est_minutes = VISIT_TIME_OVERRIDES[poi_id]

    props: Dict[str, object] = {
        "id": poi_id,
        "id_num": row.get("id_num"),
        "name": row.get("name"),
        "zone": row.get("zone"),
        "category": category,
        "interest_tags": split_tags(row.get("interest_tags", "")),
        "estimated_visit_minutes": est_minutes,
        "accessibility_level": row.get("accessibility_level"),
        "priority_score": float(row["priority_score"]) if row.get("priority_score") else 0.0,
    }

    optional_fields = {
        "accessibility_notes": row.get("accessibility_notes"),
        "opening_ruleset_id": row.get("opening_ruleset_id"),
        "exception_calendar_id": row.get("exception_calendar_id"),
        "floor": row.get("floor"),
        "wing": row.get("wing"),
        "garden_type": row.get("garden_type"),
        "facility_type": row.get("facility_type"),
        "cost_category": row.get("cost_category"),
        "entrance_type": row.get("entrance_type"),
        # Date-specific opening hours (needed for planner filters)
        "opening_time": row.get("opening_time"),
        "closing_time": row.get("closing_time"),
    }

    for key, value in optional_fields.items():
        if value:
            props[key] = value

    if row.get("is_fountain_area"):
        props["is_fountain_area"] = parse_bool(row["is_fountain_area"])

    return poi_id, category, props


def prepare_connection(row: Dict[str, str]) -> Tuple[str, str, Dict[str, object]]:
    props: Dict[str, object] = {
        "base_walk_min": parse_float(row.get("base_walk_min", "")),
        "family_walk_min": parse_float(row.get("family_walk_min", "")),
        "elder_walk_min": parse_float(row.get("elder_walk_min", "")),
        "is_step_free": parse_bool(row.get("is_step_free", "")),
        "stroller_friendly": parse_bool(row.get("stroller_friendly", "")),
        "path_type": row.get("path_type"),
    }

    if row.get("notes"):
        props["notes"] = row["notes"]

    return row["from_id"], row["to_id"], props


def ingest_pois(session: Session, rows: Iterable[Dict[str, str]]) -> int:
    count = 0
    for row in rows:
        poi_id, category, props = prepare_poi(row)
        labels_clause = ""
        if category:
            labels_clause = f"SET poi:`{category}`"
        query = f"""
        MERGE (poi:POI {{id: $id}})
        SET poi += $props
        {labels_clause}
        """
        params = {"id": poi_id, "props": props}
        session.run(query, params)
        count += 1
    return count


def ingest_connections(session: Session, rows: Iterable[Dict[str, str]]) -> int:
    query = """
    MATCH (from:POI {id: $from_id}), (to:POI {id: $to_id})
    MERGE (from)-[rel:CONNECTS_TO]->(to)
    SET rel += $props
    """
    count = 0
    for row in rows:
        from_id, to_id, props = prepare_connection(row)
        params = {"from_id": from_id, "to_id": to_id, "props": props}
        session.run(query, params)
        count += 1
    return count


def run_ingestion(
    mode: IngestMode = "all",
    *,
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
    pois_path: Optional[Path] = None,
    connections_path: Optional[Path] = None,
    skip_validation: bool = True,
) -> Tuple[int, int]:
    """
    Run the ingestion pipeline without relying on CLI flags.

    Parameters
    ----------
    mode:
        Choose whether to ingest everything ("all"), only POIs ("pois_only"),
        or only connections ("connections_only").
    uri, user, password:
        Neo4j connection credentials. Defaults fall back to environment variables:
        NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD. If unspecified, uses the
        Neo4j defaults (neo4j://localhost:7687, neo4j/neo4j).
    pois_path, connections_path:
        Override CSV paths if desired. Defaults to the repository main-data files.
    skip_validation:
        If True, skip pre-ingestion graph validation. Default is True (skipped by default).
        Set to False to enable validation locally.

    Returns
    -------
    tuple[int, int]
        Number of POIs and connections processed respectively.
    """

    uri = normalize_neo4j_uri(uri or os.getenv("NEO4J_URI"))
    user = user or os.getenv("NEO4J_USERNAME", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "neo4j")

    database = database or os.getenv("NEO4J_DATABASE", "neo4j")
    pois_path = Path(pois_path) if pois_path else Path("data/main_data/pois.csv")
    connections_path = (
        Path(connections_path) if connections_path else Path("data/main_data/connections.csv")
    )

    # Pre-ingestion validation
    if not skip_validation and VALIDATION_AVAILABLE and mode in ("all", "connections_only"):
        print("="*70)
        print("PRE-INGESTION VALIDATION")
        print("="*70)
        print()

        validator = GraphValidator(pois_path, connections_path, accessibility="any")
        report = validator.validate_all()

        if report.errors > 0:
            print()
            print("="*70)
            print("❌ VALIDATION FAILED")
            print("="*70)
            print(f"Found {report.errors} critical error(s) in graph data.")
            print("Please fix these issues before ingestion.")
            print()
            print("To see detailed validation report, run:")
            print("  python scripts/validate_graph.py")
            print()
            print("To skip validation and ingest anyway, set skip_validation=True")
            print("="*70)
            sys.exit(1)
        else:
            print()
            print("="*70)
            print("✅ VALIDATION PASSED")
            print("="*70)
            if report.warnings > 0:
                print(f"Note: {report.warnings} warning(s) found (see above)")
            print("Proceeding with ingestion...")
            print()

    poi_count = 0
    rel_count = 0

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            if mode in ("all", "pois_only"):
                poi_count = ingest_pois(session, read_csv(pois_path))
            if mode in ("all", "connections_only"):
                rel_count = ingest_connections(session, read_csv(connections_path))
    finally:
        driver.close()

    print(f"Ingested {poi_count} POIs and {rel_count} connections.")
    return poi_count, rel_count


if __name__ == "__main__":
    run_ingestion()
