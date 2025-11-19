# Versailles Knowledge Graph & Itinerary Planner

This project models Versailles Palace points of interest in Neo4j and exposes a planning toolkit that generates accessibility-aware itineraries.  
The repository is structured around three pillars:

- **Data** – curated CSVs for POIs, opening schedules, and walking connections.  
- **Graph Tooling** – ingestion + updater scripts to populate and refresh Neo4j.  
- **Planner** – routing helpers, itinerary orchestration, FastAPI service, and tests.

---

## 1. Project Layout

| Path | Purpose |
| --- | --- |
| `data/main_data/pois.csv` | POI definitions aligned with `configs/required.yaml`. |
| `data/main_data/opening_hours.csv` | Ruleset-based high/low season schedules. |
| `data/main_data/connections.csv` | Directed walks with multi-profile travel times. |
| `scripts/ingest.py` | Loads POIs + edges into Neo4j via MERGE. |
| `scripts/updater.py` | Applies daily opening/closing windows by ruleset. |
| `scripts/planner_utils.py` | APOC-backed shortest-path helper. |
| `scripts/planner.py` | Visit selection, routing, and itinerary scheduling. |
| `scripts/api.py` | FastAPI wrapper exposing `/itinerary`. |
| `tests/` | Pytest suites for planner core and API contract. |
| `scripts/benchmarks/` | Helper scripts for route/solver benchmarking. |

---

## 2. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Required services:

1. **Neo4j 5.x** with the APOC plugin enabled. Default URI `neo4j://localhost:7687`, credentials configurable via `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`.
2. **Python 3.10+** (for `dataclasses`, `typing.Literal`, etc.).

---

## 3. Validate Graph Data

**Before loading data**, validate that your graph meets TSP algorithm requirements:

```bash
# Validate with default accessibility (any)
python scripts/validate_graph.py

# Validate step-free accessibility
python scripts/validate_graph.py --accessibility step_free

# Generate JSON report
python scripts/validate_graph.py --output-json validation_report.json
```

The validator checks **8 critical rules**:

1. **Orphaned POIs** – All POIs in `pois.csv` must have connections
2. **Entrance Reachability** – Every POI reachable from main/Trianon entrances
3. **Zone Bidirectionality** – Gardens/Trianon/Park have two-way paths (Castle tour is one-way)
4. **Dead Ends** – No garden nodes with out-degree=0 (Castle tour end is allowed)
5. **Path Length** – All paths ≤20 hops (APOC `maxLevel` limit)
6. **Accessibility Connectivity** – Step-free/stroller graphs remain connected
7. **Zone Bridges** – Connections exist: Castle→Gardens, Gardens↔Trianon, Gardens↔Park
8. **Castle Tour Structure** – Directed path exists through all castle rooms

### Understanding Validation Results

**Example Output:**
```
✅ PASSED: All POIs have connections
❌ FAILED: Found 3 POI(s) unreachable from main entrance
  - versailles:Room:escalier-de-la-reine
  - versailles:Room:escalier-des-princes
  - versailles:Room:salles-louis-xiv
⚠️  WARNING: 2 POIs lost connectivity with step_free filter
```

**Critical Issues** (errors) prevent TSP algorithms from working:
- **Orphaned POIs**: Add connections to/from these nodes
- **Unreachable nodes**: Add paths from entrance or mark as non-bookable
- **Missing zone bridges**: Add connections between zones
- **Dead ends in gardens**: Add return paths (Castle tour ends are OK)

**Warnings** indicate data quality issues:
- POIs unreachable with accessibility filter should have `accessibility_level="none"` metadata
- Isolated starts (no incoming edges) may trap greedy solver

### Data Requirements for TSP Algorithms

The planner uses **three TSP solvers** with different requirements:

| Solver | POI Count | Graph Requirement |
|--------|-----------|-------------------|
| **Permutation** | ≤7 | Complete graph (all N² paths exist) |
| **Greedy** | 8-10 | Connected graph (no dead ends) |
| **Held-Karp** | >10 | All POIs reachable from entrance |

**Zone-Specific Rules:**
- **Castle (Rooms)**: One-way tour path (directed graph) – visitors follow museum flow
- **Gardens/Trianon/Park**: Free movement (undirected graph) – all paths should be bidirectional

---

## 4. Load & Refresh the Graph

1. **Initial ingest** (includes automatic validation)

```bash
python scripts/ingest.py
# or programmatically
from scripts.ingest import run_ingestion
run_ingestion()  # Validates before ingesting

# Skip validation if needed
run_ingestion(skip_validation=True)
```

2. **Daily opening-time refresh**

```bash
python scripts/updater.py
# programmatic usage
from scripts.updater import run_daily_update
run_daily_update()
```

The updater currently relies on a static fallback schedule that matches `data/main_data/opening_hours.csv`. Swap in a live scraper by implementing `OpeningHoursSource`.

---

## 5. Planner Usage

```python
from datetime import datetime
from scripts.planner import PlannerConstraints, plan_versailles_itinerary

constraints = PlannerConstraints(
    interests=["history", "must_see"],
    user_profile="family",
    accessibility="step_free",
    must_include=["versailles:Room:galerie-des-glaces"]
)

itinerary = plan_versailles_itinerary(
    start_time=datetime(2024, 6, 1, 9, 0),
    total_duration_minutes=240,
    constraints=constraints,
)
```

Returned objects include visit steps, total travel/visit/idle minutes, and the path metadata used during routing.

### REST API

Run the HTTP façade with:

```bash
uvicorn scripts.api:app --reload
```

For deployment platforms (e.g., Render), use:

```bash
uvicorn scripts.api:app --host 0.0.0.0 --port $PORT
```

Example request:

```bash
curl -X POST http://localhost:8000/itinerary \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-06-01T09:00:00",
    "total_duration_minutes": 240,
    "constraints": {
      "interests": ["history"],
      "user_profile": "standard",
      "accessibility": "any",
      "must_include": [],
      "exclude_ids": []
    }
  }'
```

---

## 6. Testing

Pytest suites cover itinerary selection logic and FastAPI response contracts.

```bash
pytest
```

> **Note:** Some sandboxed environments may terminate long-running test processes. If that happens, execute the command locally after activating your virtualenv.

---

## 7. Next Steps

- Swap the static `StaticScheduleSource` with a BeautifulSoup scraper of the official Versailles site.
- Extend `data/main_data/pois.csv` with Facility/Entrance entries (restrooms, ticketing) to enrich accessibility routing.
- Integrate a heuristic solver (e.g., OR-Tools) for itineraries spanning more than seven POIs.
- Run the benchmark harness in `scripts/benchmarks/performance.py` to profile solver quality and speed across scenarios.
  ```bash
  python scripts/benchmarks/performance.py --max-pois 9 --trials 5 --output benchmarks.json
  python scripts/benchmarks/performance.py --offline --solver greedy --output benchmarks_offline.json
  ```
