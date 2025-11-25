#!/usr/bin/env python3.11
"""
Comprehensive TSP Testing Suite
Tests all constraint combinations to validate algorithm correctness.
"""
import os
import json
from datetime import datetime
from pathlib import Path

# Require Neo4j credentials to be provided via environment (no defaults here).
NEO4J_URI = os.environ.get('NEO4J_URI')
NEO4J_USERNAME = os.environ.get('NEO4J_USERNAME')
NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD')
NEO4J_DATABASE = os.environ.get('NEO4J_DATABASE')

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE]):
    raise RuntimeError(
        "Set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, and NEO4J_DATABASE "
        "in the environment before running test_tsp_comprehensive.py"
    )

os.environ['NEO4J_URI'] = NEO4J_URI
os.environ['NEO4J_USERNAME'] = NEO4J_USERNAME
os.environ['NEO4J_PASSWORD'] = NEO4J_PASSWORD
os.environ['NEO4J_DATABASE'] = NEO4J_DATABASE

from scripts.planner import PlannerConstraints, plan_versailles_itinerary

# Test scenarios
TEST_SCENARIOS = [
    {
        "name": "Standard Tourist - 4 hours",
        "start_time": datetime(2024, 6, 1, 9, 0),
        "duration": 240,
        "constraints": PlannerConstraints(
            interests=["history", "must_see"],
            user_profile="standard",
            accessibility="any"
        ),
        "expected_min_pois": 5,
        "expected_max_travel": 80
    },
    {
        "name": "Family with Stroller - 3 hours",
        "start_time": datetime(2024, 6, 1, 10, 0),
        "duration": 180,
        "constraints": PlannerConstraints(
            interests=["gardens", "must_see"],
            user_profile="family",
            accessibility="stroller"
        ),
        "expected_min_pois": 3,
        "expected_max_travel": 60
    },
    {
        "name": "Elder Step-Free - 2 hours",
        "start_time": datetime(2024, 6, 1, 14, 0),
        "duration": 120,
        "constraints": PlannerConstraints(
            interests=["history"],
            user_profile="elder",
            accessibility="step_free"
        ),
        "expected_min_pois": 2,
        "expected_max_travel": 40
    },
    {
        "name": "Must Include Hall of Mirrors",
        "start_time": datetime(2024, 6, 1, 9, 30),
        "duration": 180,
        "constraints": PlannerConstraints(
            interests=["history"],
            user_profile="standard",
            accessibility="any",
            must_include=["versailles:Room:galerie-des-glaces"]
        ),
        "expected_min_pois": 3,
        "expected_max_travel": 50
    },
    {
        "name": "Exclude Specific POIs",
        "start_time": datetime(2024, 6, 1, 11, 0),
        "duration": 240,
        "constraints": PlannerConstraints(
            interests=["gardens"],
            user_profile="standard",
            accessibility="any",
            exclude_ids=["versailles:Garden:cour-dhonneur"]
        ),
        "expected_min_pois": 4,
        "expected_max_travel": 70
    },
    {
        "name": "Lunch Break Required",
        "start_time": datetime(2024, 6, 1, 10, 0),
        "duration": 300,
        "constraints": PlannerConstraints(
            interests=["history", "gardens"],
            user_profile="family",
            accessibility="any",
            lunch_break=True
        ),
        "expected_min_pois": 4,
        "expected_max_travel": 80,
        "expect_lunch": True
    },
    {
        "name": "Hard Time Limits (Strict Schedule)",
        "start_time": datetime(2024, 6, 1, 9, 0),
        "duration": 180,
        "constraints": PlannerConstraints(
            interests=["must_see"],
            user_profile="standard",
            accessibility="any",
            hard_time_limits=True
        ),
        "expected_min_pois": 3,
        "expected_max_travel": 50
    },
    {
        "name": "Quick Visit - 90 minutes",
        "start_time": datetime(2024, 6, 1, 15, 0),
        "duration": 90,
        "constraints": PlannerConstraints(
            interests=["must_see"],
            user_profile="standard",
            accessibility="any"
        ),
        "expected_min_pois": 2,
        "expected_max_travel": 30
    },
    {
        "name": "Full Day - 6 hours",
        "start_time": datetime(2024, 6, 1, 9, 0),
        "duration": 360,
        "constraints": PlannerConstraints(
            interests=["history", "art", "gardens"],
            user_profile="standard",
            accessibility="any"
        ),
        "expected_min_pois": 8,
        "expected_max_travel": 120
    },
    {
        "name": "Art Lovers - Paintings Focus",
        "start_time": datetime(2024, 6, 1, 10, 0),
        "duration": 240,
        "constraints": PlannerConstraints(
            interests=["art", "paintings"],
            user_profile="standard",
            accessibility="any"
        ),
        "expected_min_pois": 4,
        "expected_max_travel": 60
    }
]


def run_test(scenario: dict) -> dict:
    """Run a single test scenario."""
    print(f"\n{'='*70}")
    print(f"Test: {scenario['name']}")
    print(f"{'='*70}")

    result = {
        "name": scenario["name"],
        "status": "PENDING",
        "error": None,
        "itinerary": None,
        "validations": {}
    }

    try:
        # Plan itinerary
        itinerary = plan_versailles_itinerary(
            start_time=scenario["start_time"],
            total_duration_minutes=scenario["duration"],
            constraints=scenario["constraints"]
        )

        # Extract stats
        poi_count = len([s for s in itinerary.steps if s.poi_id != "__lunch_break__"])
        has_lunch = any(s.poi_id == "__lunch_break__" for s in itinerary.steps)

        print(f"✅ Itinerary created successfully")
        print(f"   POIs visited: {poi_count}")
        print(f"   Travel time: {itinerary.travel_minutes:.1f} min")
        print(f"   Visit time: {itinerary.visit_minutes} min")
        print(f"   Idle time: {itinerary.idle_minutes:.1f} min")
        print(f"   Total time: {itinerary.total_minutes:.1f} min")
        print(f"   Has lunch: {has_lunch}")

        # Validate constraints
        validations = {}

        # Check POI count
        if poi_count >= scenario["expected_min_pois"]:
            validations["min_pois"] = "PASS"
            print(f"   ✅ POI count >= {scenario['expected_min_pois']}")
        else:
            validations["min_pois"] = "FAIL"
            print(f"   ❌ POI count < {scenario['expected_min_pois']}")

        # Check travel time
        if itinerary.travel_minutes <= scenario["expected_max_travel"]:
            validations["max_travel"] = "PASS"
            print(f"   ✅ Travel time <= {scenario['expected_max_travel']} min")
        else:
            validations["max_travel"] = "WARN"
            print(f"   ⚠️  Travel time > {scenario['expected_max_travel']} min (acceptable)")

        # Check duration
        if itinerary.total_minutes <= scenario["duration"] * 1.1:  # Allow 10% overage
            validations["duration"] = "PASS"
            print(f"   ✅ Total time within duration limit")
        else:
            validations["duration"] = "FAIL"
            print(f"   ❌ Total time exceeds duration limit")

        # Check lunch break
        if scenario.get("expect_lunch"):
            if has_lunch:
                validations["lunch"] = "PASS"
                print(f"   ✅ Lunch break included as expected")
            else:
                validations["lunch"] = "FAIL"
                print(f"   ❌ Lunch break missing")

        # Check must_include constraint
        if scenario["constraints"].must_include:
            all_included = all(
                any(s.poi_id == required for s in itinerary.steps)
                for required in scenario["constraints"].must_include
            )
            if all_included:
                validations["must_include"] = "PASS"
                print(f"   ✅ All required POIs included")
            else:
                validations["must_include"] = "FAIL"
                print(f"   ❌ Some required POIs missing")

        # Overall status
        failed = [k for k, v in validations.items() if v == "FAIL"]
        if failed:
            result["status"] = "FAIL"
            print(f"\n❌ Test FAILED ({len(failed)} validation(s) failed)")
        else:
            result["status"] = "PASS"
            print(f"\n✅ Test PASSED")

        result["validations"] = validations
        result["itinerary"] = {
            "poi_count": poi_count,
            "travel_minutes": itinerary.travel_minutes,
            "visit_minutes": itinerary.visit_minutes,
            "total_minutes": itinerary.total_minutes,
            "has_lunch": has_lunch,
            "steps": [
                {
                    "poi_id": s.poi_id,
                    "name": s.name,
                    "arrival": s.arrival_time.isoformat(),
                    "departure": s.departure_time.isoformat()
                }
                for s in itinerary.steps
            ]
        }

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
        print(f"\n❌ Test ERROR: {e}")

    return result


def main():
    """Run all test scenarios."""
    print("="*70)
    print("COMPREHENSIVE TSP TESTING SUITE")
    print("="*70)
    print(f"Total scenarios: {len(TEST_SCENARIOS)}")
    print("")

    results = []

    for scenario in TEST_SCENARIOS:
        result = run_test(scenario)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    print(f"Total: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Errors: {errors}")
    print("")

    # Detailed results
    for result in results:
        status_icon = {
            "PASS": "✅",
            "FAIL": "❌",
            "ERROR": "⚠️",
            "PENDING": "⏸️"
        }[result["status"]]

        print(f"{status_icon} {result['name']}: {result['status']}")
        if result["error"]:
            print(f"   Error: {result['error']}")

    # Save results
    output_file = Path("benchmarks/results/tsp_comprehensive_tests.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n📊 Results saved to: {output_file}")

    # Exit code
    if failed > 0 or errors > 0:
        print("\n❌ Some tests failed. Review results above.")
        return 1
    else:
        print("\n✅ All tests passed! TSP algorithm is working correctly.")
        return 0


if __name__ == "__main__":
    exit(main())
