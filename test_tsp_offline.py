#!/usr/bin/env python3.11
"""
Offline TSP Testing Suite
Tests TSP algorithm using offline mode (no Neo4j required).
"""
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.benchmarks.performance import (
    generate_mock_candidates,
    make_mock_shortest_path,
    DEFAULT_SCENARIOS
)
from scripts.planner import (
    CandidatePOI,
    PlannerConstraints,
    determine_route,
    determine_route_greedy,
    select_pois
)
import scripts.planner_utils as planner_utils

print("="*70)
print("OFFLINE TSP TESTING SUITE")
print("="*70)
print("Testing TSP algorithm logic without Neo4j")
print()

# Mock the shortest_path function
original_get_shortest_path = planner_utils.get_shortest_path
planner_utils.get_shortest_path = make_mock_shortest_path()

test_results = []

# Test 1: POI Selection
print("Test 1: POI Selection Logic")
print("-" * 70)
candidates = generate_mock_candidates(
    ["poi1", "poi2", "poi3", "poi4", "poi5"],
    visit_minutes=15
)
selected = select_pois(
    candidates,
    total_duration_minutes=120,
    must_include=["poi1"],
    max_pois=4
)
print(f"✅ Selected {len(selected)} POIs from 5 candidates")
print(f"   Must-include POI present: {'poi1' in [p.poi_id for p in selected]}")
test_results.append(("POI Selection", len(selected) > 0 and "poi1" in [p.poi_id for p in selected]))
print()

# Test 2: Permutation Solver (≤7 POIs)
print("Test 2: Permutation Solver (7 POIs)")
print("-" * 70)
poi_ids = [f"poi{i}" for i in range(1, 8)]
constraints = PlannerConstraints(
    interests=[],
    user_profile="standard",
    accessibility="any"
)
try:
    order, pair_paths, merged = determine_route(poi_ids, constraints=constraints)
    print(f"✅ Route computed: {len(order)} POIs")
    print(f"   Total travel time: {merged.total_minutes:.1f} minutes")
    print(f"   Route: {' → '.join(order[:3])} ... {order[-1]}")
    test_results.append(("Permutation Solver", len(order) == 7))
except Exception as e:
    print(f"❌ Failed: {e}")
    test_results.append(("Permutation Solver", False))
print()

# Test 3: Greedy Solver (8 POIs)
print("Test 3: Greedy Solver (8 POIs)")
print("-" * 70)
poi_ids = [f"poi{i}" for i in range(1, 9)]
try:
    order, pair_paths, merged = determine_route_greedy(poi_ids, constraints=constraints)
    print(f"✅ Route computed: {len(order)} POIs")
    print(f"   Total travel time: {merged.total_minutes:.1f} minutes")
    print(f"   Route: {' → '.join(order[:3])} ... {order[-1]}")
    test_results.append(("Greedy Solver 8 POIs", len(order) == 8))
except Exception as e:
    print(f"❌ Failed: {e}")
    test_results.append(("Greedy Solver 8 POIs", False))
print()

# Test 4: Greedy Solver (10 POIs)
print("Test 4: Greedy Solver (10 POIs)")
print("-" * 70)
poi_ids = [f"poi{i}" for i in range(1, 11)]
try:
    order, pair_paths, merged = determine_route_greedy(poi_ids, constraints=constraints)
    print(f"✅ Route computed: {len(order)} POIs")
    print(f"   Total travel time: {merged.total_minutes:.1f} minutes")
    print(f"   Route: {' → '.join(order[:4])} ... {order[-1]}")
    test_results.append(("Greedy Solver 10 POIs", len(order) == 10))
except Exception as e:
    print(f"❌ Failed: {e}")
    test_results.append(("Greedy Solver 10 POIs", False))
print()

# Test 5: Different User Profiles
print("Test 5: User Profile Variations")
print("-" * 70)
poi_ids = [f"poi{i}" for i in range(1, 6)]
profiles_tested = {}
for profile in ["standard", "family", "elder"]:
    constraints = PlannerConstraints(
        interests=[],
        user_profile=profile,
        accessibility="any"
    )
    try:
        order, pair_paths, merged = determine_route(poi_ids, constraints=constraints)
        profiles_tested[profile] = merged.total_minutes
        print(f"   {profile}: {merged.total_minutes:.1f} min")
    except Exception as e:
        print(f"   {profile}: Failed - {e}")
        profiles_tested[profile] = None

all_profiles_work = all(v is not None for v in profiles_tested.values())
print(f"{'✅' if all_profiles_work else '❌'} All profiles tested")
test_results.append(("User Profiles", all_profiles_work))
print()

# Test 6: Accessibility Filters
print("Test 6: Accessibility Variations")
print("-" * 70)
poi_ids = [f"poi{i}" for i in range(1, 6)]
accessibility_tested = {}
for access in ["any", "step_free", "stroller"]:
    constraints = PlannerConstraints(
        interests=[],
        user_profile="standard",
        accessibility=access
    )
    try:
        order, pair_paths, merged = determine_route(poi_ids, constraints=constraints)
        accessibility_tested[access] = merged.total_minutes
        print(f"   {access}: {merged.total_minutes:.1f} min")
    except Exception as e:
        print(f"   {access}: Failed - {e}")
        accessibility_tested[access] = None

all_accessibility_work = all(v is not None for v in accessibility_tested.values())
print(f"{'✅' if all_accessibility_work else '❌'} All accessibility modes tested")
test_results.append(("Accessibility Modes", all_accessibility_work))
print()

# Test 7: Edge Cases
print("Test 7: Edge Cases")
print("-" * 70)

# Single POI
try:
    order, pair_paths, merged = determine_route(["poi1"], constraints=constraints)
    single_poi_works = len(order) == 1 and merged.total_minutes == 0
    print(f"   Single POI: {'✅' if single_poi_works else '❌'}")
    test_results.append(("Single POI", single_poi_works))
except Exception as e:
    print(f"   Single POI: ❌ {e}")
    test_results.append(("Single POI", False))

# Two POIs
try:
    order, pair_paths, merged = determine_route(["poi1", "poi2"], constraints=constraints)
    two_poi_works = len(order) == 2
    print(f"   Two POIs: {'✅' if two_poi_works else '❌'}")
    test_results.append(("Two POIs", two_poi_works))
except Exception as e:
    print(f"   Two POIs: ❌ {e}")
    test_results.append(("Two POIs", False))

print()

# Restore original function
planner_utils.get_shortest_path = original_get_shortest_path

# Summary
print("="*70)
print("TEST SUMMARY")
print("="*70)
passed = sum(1 for _, result in test_results if result)
total = len(test_results)

for name, result in test_results:
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status}: {name}")

print()
print(f"Total: {total} tests")
print(f"Passed: {passed}/{total}")
print(f"Success rate: {100*passed/total:.1f}%")
print()

# Save results
output = {
    "test_results": [
        {"name": name, "passed": result}
        for name, result in test_results
    ],
    "summary": {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "success_rate": 100 * passed / total
    }
}

output_file = Path("benchmarks/results/tsp_offline_tests.json")
output_file.parent.mkdir(parents=True, exist_ok=True)
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"📊 Results saved to: {output_file}")
print()

if passed == total:
    print("✅ ALL TESTS PASSED! TSP algorithm is working correctly.")
    print("   Ready for repository restructuring.")
    exit(0)
else:
    print(f"❌ {total - passed} test(s) failed. Review errors above.")
    exit(1)
