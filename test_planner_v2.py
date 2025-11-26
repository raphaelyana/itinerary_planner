#!/usr/bin/env python3
"""Quick test of the new planner_v2 architecture."""

import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent))

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.planner_v2 import create_itinerary_v2, PlannerConstraints

def main():
    print("=" * 70)
    print("TESTING PLANNER V2 - NEW ARCHITECTURE")
    print("=" * 70)
    print()

    # Test 1: Castle + Gardens (2 hours)
    print("Test 1: Castle + Gardens (120 minutes, art/history interests)")
    print("-" * 70)

    constraints = PlannerConstraints(
        interests=["art", "history"],
        user_profile="standard",
        accessibility="any",
        must_include=[],
        exclude_ids=[],
    )

    start_time = datetime.now() + timedelta(days=1)
    start_time = start_time.replace(hour=9, minute=0, second=0, microsecond=0)

    try:
        itinerary = create_itinerary_v2(
            start_time=start_time,
            total_duration_minutes=120,
            constraints=constraints,
        )

        print(f"✅ SUCCESS!")
        print(f"   Total POIs: {len(itinerary.steps)}")
        print(f"   Visit time: {itinerary.visit_minutes} min")
        print(f"   Travel time: {itinerary.travel_minutes:.1f} min")
        print(f"   Total time: {itinerary.total_minutes:.1f} min")
        print()
        print("   First 5 POIs:")
        for i, step in enumerate(itinerary.steps[:5], 1):
            print(f"      {i}. {step.name} ({step.stay_minutes} min)")
        print()

    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("=" * 70)

if __name__ == "__main__":
    main()
