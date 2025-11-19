# Graph Validation Summary

This document summarizes the validation results for the Versailles knowledge graph and provides actionable recommendations for fixing the identified issues.

## Quick Start

```bash
# Run validation
python scripts/validate_graph.py

# Check step-free accessibility
python scripts/validate_graph.py --accessibility step_free

# Generate JSON report
python scripts/validate_graph.py --output-json report.json
```

---

## Current Data Issues (Accessibility: "any")

### ❌ Critical Errors (Must Fix)

#### 1. Orphaned POIs (3 POIs with no connections)

These POIs exist in `pois.csv` but have NO connections in `connections.csv`:

- `versailles:Room:escalier-de-la-reine`
- `versailles:Room:escalier-des-princes`
- `versailles:Room:salles-louis-xiv`

**Impact**: TSP algorithms cannot route to these POIs. They will never appear in itineraries.

**Fix**: Add connections in `connections.csv`:
- Connect to adjacent rooms in the castle tour sequence
- Or remove from `pois.csv` if not meant to be visitor destinations

#### 2. Unreachable POIs from Main Entrance (Same 3 POIs)

The same 3 orphaned POIs cannot be reached from the main entrance (`versailles:Garden:cour-dhonneur`).

**Fix**: Same as above - add connections to integrate them into the castle tour path.

---

## Step-Free Accessibility Issues (Accessibility: "step_free")

When validating with `--accessibility step_free`, additional issues appear:

### ❌ Critical Errors

#### 1. Still 3 Orphaned POIs (same as above)

#### 2. 6 POIs Unreachable from Main Entrance

- `versailles:Room:escalier-de-la-reine` (orphaned)
- `versailles:Room:escalier-des-princes` (orphaned)
- `versailles:Room:salles-louis-xiv` (orphaned)
- `versailles:Room:opera-royal` ⚠️ **NEW**
- `versailles:Room:petits-appartements-du-roi` ⚠️ **NEW**
- `versailles:Room:salles-des-croisades` ⚠️ **NEW**

**Reason**: All paths to Opéra Royal, Petits Appartements, and Salles des Croisades require stairs (`is_step_free=False` edges).

**Fix Options**:
1. Add step-free alternative paths (elevators, ramps) if they exist
2. Mark these POIs with `accessibility_level="none"` in `pois.csv` to indicate they require stairs

#### 3. 3 Trianon POIs Unreachable from Trianon Entrance

- `versailles:Trianon:belvedere`
- `versailles:Trianon:grotte`
- `versailles:Trianon:hameau-buildings`

**Reason**: All paths require non-step-free edges (terrain, stairs).

**Fix**: Same as above - add step-free paths or mark as `accessibility_level="none"`.

#### 4. Dead Ends in Garden Zones (6 Trianon nodes)

These Trianon POIs have outgoing edges but become dead ends with step-free filter:

- `versailles:Trianon:belvedere`
- `versailles:Trianon:grotte`
- `versailles:Trianon:hameau-buildings`
- `versailles:Trianon:maison-reine`
- `versailles:Trianon:theatre-reine`
- `versailles:Trianon:tour-marlborough`

**Impact**: Greedy solver gets trapped at these nodes and fails to complete itinerary.

**Fix**: Add step-free return paths from these nodes.

### ⚠️ Warnings

#### 1. 9 Isolated Start Nodes

These nodes have no incoming edges with step-free filter:

- Castle tour ends (expected): `opera-royal`, `petits-appartements-du-roi`, `salles-des-croisades`
- Trianon nodes: `belvedere`, `grotte`, `hameau-buildings`, `maison-reine`, `theatre-reine`, `tour-marlborough`

**Impact**: Can be visited but cannot be reached via step-free path.

#### 2. 2 POIs Missing Accessibility Metadata

- `versailles:Room:salles-des-croisades`
- `versailles:Trianon:belvedere`

These POIs are unreachable with step-free filter but don't have `accessibility_level="none"` in `pois.csv`.

**Fix**: Update `pois.csv`:
```csv
versailles:Room:salles-des-croisades,...,none,...
versailles:Trianon:belvedere,...,none,...
```

---

## Understanding Zone-Specific Rules

### Castle (Rooms) - One-Way Tour
- **Expected behavior**: Directed graph (visitors follow museum flow)
- **No bidirectional requirement**: One-way paths are correct
- **38 one-way edges** found (this is normal)
- **3 tour end points**: `opera-royal`, `petits-appartements-du-roi`, `salles-des-croisades`

### Gardens/Trianon/Park - Free Movement
- **Required**: Bidirectional edges (visitors can walk in any direction)
- **Current status**: ✅ All garden zones have proper two-way paths
- **Dead ends not allowed**: Every garden node must have return paths

---

## Validation Integration

### Automatic Pre-Ingestion Validation

The ingestion script (`scripts/ingest.py`) now automatically validates data before loading:

```python
from scripts.ingest import run_ingestion

# Validates automatically (fails if errors found)
run_ingestion()

# Skip validation if needed
run_ingestion(skip_validation=True)
```

### Manual Validation

Run standalone validation anytime:

```bash
# Quick check
python scripts/validate_graph.py

# All accessibility levels
python scripts/validate_graph.py --accessibility any
python scripts/validate_graph.py --accessibility step_free
python scripts/validate_graph.py --accessibility stroller
```

---

## Recommended Action Plan

### Phase 1: Fix Critical Errors (Priority: HIGH)

1. **Add connections for 3 orphaned POIs**
   - `escalier-de-la-reine`
   - `escalier-des-princes`
   - `salles-louis-xiv`

   Action: Integrate into castle tour path in `connections.csv`

2. **Mark stairs-only POIs with accessibility metadata**
   - Update `pois.csv` to set `accessibility_level="none"` for:
     - `opera-royal`
     - `petits-appartements-du-roi`
     - `salles-des-croisades`
     - `belvedere`
     - `grotte` (if no step-free access exists)

### Phase 2: Improve Step-Free Routing (Priority: MEDIUM)

3. **Add step-free return paths for Trianon garden nodes**
   - Review terrain and add bidirectional step-free connections for:
     - `maison-reine`
     - `theatre-reine`
     - `tour-marlborough`
     - `hameau-buildings` (if accessible)

4. **Verify step-free alternatives**
   - Check if elevators/ramps exist to:
     - Opéra Royal
     - Petits Appartements du Roi
   - Add connections if yes, mark as `none` if no

### Phase 3: Validate Thoroughly (Priority: MEDIUM)

5. **Run validation for all accessibility levels**
   ```bash
   python scripts/validate_graph.py --accessibility any
   python scripts/validate_graph.py --accessibility step_free
   python scripts/validate_graph.py --accessibility stroller
   ```

6. **Test with real itinerary requests**
   ```python
   from scripts.planner import plan_versailles_itinerary
   from datetime import datetime

   # Test step-free routing
   itinerary = plan_versailles_itinerary(
       start_time=datetime(2024, 6, 1, 9, 0),
       total_duration_minutes=180,
       constraints={"accessibility": "step_free"}
   )
   ```

---

## Why Some Trajectories Fail

Based on the validation results and code analysis, here are the root causes:

### Failure Type 1: Orphaned POIs
- **Symptom**: "Unable to pre-compute pairwise paths for permutation solver"
- **Cause**: 3 POIs have no connections at all
- **Solution**: Add connections in `connections.csv`

### Failure Type 2: Greedy Solver Traps
- **Symptom**: "Unable to build a step-free route for all POIs using greedy heuristic"
- **Cause**: Greedy solver reaches a dead-end Trianon node with no step-free return path
- **Solution**: Add bidirectional step-free paths in Trianon zone

### Failure Type 3: Accessibility Filter Too Restrictive
- **Symptom**: "No path found between X and Y" with `accessibility="step_free"`
- **Cause**: Some POI pairs only connected via stairs
- **Solution**: Add step-free alternatives OR mark POIs as `accessibility_level="none"`

### Failure Type 4: Unreachable from Entrance
- **Symptom**: "Unable to assemble a route that connects all POIs" (Held-Karp)
- **Cause**: POI cannot be reached from entrance (even with Trianon entrance fallback)
- **Solution**: Add connection path from at least one entrance

---

## Data Quality Checklist

Before deploying or running tests, ensure:

- [ ] No orphaned POIs (all POIs in `pois.csv` appear in `connections.csv`)
- [ ] All POIs reachable from main entrance (or Trianon entrance if in Trianon zone)
- [ ] Garden/Trianon/Park zones have bidirectional connections
- [ ] No unexpected dead ends in garden zones
- [ ] All paths ≤20 hops
- [ ] POIs requiring stairs marked with `accessibility_level="none"`
- [ ] Step-free subgraph remains connected (if offering step-free routing)
- [ ] Zone bridges exist (Castle→Gardens, Gardens↔Trianon, Gardens↔Park)

Run this command to verify:
```bash
python scripts/validate_graph.py && echo "✅ All checks passed!"
```

---

## Additional Resources

- **Main Documentation**: [README.md](README.md)
- **Validation Script**: [scripts/validate_graph.py](scripts/validate_graph.py)
- **Graph Utilities**: [scripts/graph_analysis_utils.py](scripts/graph_analysis_utils.py)
- **TSP Algorithms**: [scripts/planner.py](scripts/planner.py) (lines 410-703)
- **Path Finding**: [scripts/planner_utils.py](scripts/planner_utils.py) (lines 84-218)
