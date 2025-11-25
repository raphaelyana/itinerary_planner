#!/usr/bin/env bash
# Master benchmark script - runs all 3 classical scenarios
set -euo pipefail

echo "======================================================================"
echo "VERSAILLES KNOWLEDGE GRAPH - BENCHMARK SUITE"
echo "======================================================================"
echo ""

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="benchmarks/results"
mkdir -p "${OUTPUT_DIR}"

# Export Neo4j credentials (no defaults for secrets)
export NEO4J_URI="${NEO4J_URI:-neo4j+s://41e180bf.databases.neo4j.io}"
export NEO4J_USERNAME="${NEO4J_USERNAME:-neo4j}"
export NEO4J_PASSWORD="${NEO4J_PASSWORD:-}"
export NEO4J_DATABASE="${NEO4J_DATABASE:-neo4j}"

# Mode selection
MODE="${1:-offline}"
if [[ "${MODE}" == "online" ]]; then
    echo "Running in ONLINE mode (requires Neo4j AuraDB connection)"
    if [[ -z "${NEO4J_PASSWORD}" ]]; then
        echo "Error: set NEO4J_PASSWORD before running in online mode." >&2
        exit 1
    fi
    OFFLINE_FLAG=""
else
    echo "Running in OFFLINE mode (using mock distance solver)"
    OFFLINE_FLAG="--offline"
fi
echo ""

# Scenario 1: 7 POIs (Palace Enfilade)
echo "----------------------------------------------------------------------"
echo "Scenario 1: Palace Enfilade (7 POIs, Permutation Solver)"
echo "----------------------------------------------------------------------"
python3.11 scripts/benchmarks/performance.py \
    --max-pois 7 \
    --trials 3 \
    --solver default \
    ${OFFLINE_FLAG} \
    --output "${OUTPUT_DIR}/scenario1_7pois_${TIMESTAMP}.json"
echo "✅ Scenario 1 complete"
echo ""

# Scenario 2: 8 POIs (Palace + Courtyards)
echo "----------------------------------------------------------------------"
echo "Scenario 2: Palace + Courtyards (8 POIs, Greedy Solver)"
echo "----------------------------------------------------------------------"
python3.11 scripts/benchmarks/performance.py \
    --max-pois 8 \
    --trials 3 \
    --solver greedy \
    ${OFFLINE_FLAG} \
    --output "${OUTPUT_DIR}/scenario2_8pois_${TIMESTAMP}.json"
echo "✅ Scenario 2 complete"
echo ""

# Scenario 3: 9 POIs (Extended Mixed)
echo "----------------------------------------------------------------------"
echo "Scenario 3: Extended Mixed (9 POIs, Greedy Solver)"
echo "----------------------------------------------------------------------"
python3.11 scripts/benchmarks/performance.py \
    --max-pois 9 \
    --trials 3 \
    --solver greedy \
    ${OFFLINE_FLAG} \
    --output "${OUTPUT_DIR}/scenario3_9pois_${TIMESTAMP}.json"
echo "✅ Scenario 3 complete"
echo ""

# Summary
echo "======================================================================"
echo "BENCHMARK SUITE COMPLETE"
echo "======================================================================"
echo ""
echo "Results saved to: ${OUTPUT_DIR}/"
echo ""
echo "View results:"
echo "  ls -lh ${OUTPUT_DIR}/*${TIMESTAMP}.json"
echo ""
echo "Read summary:"
echo "  cat benchmarks/README.md"
echo ""
