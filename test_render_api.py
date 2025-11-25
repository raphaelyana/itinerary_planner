#!/usr/bin/env python3.11
"""
Test the Render API deployment with real scenarios.
"""

import requests
import json
from datetime import datetime, timedelta

API_BASE_URL = "https://itinerary-planner-mkss.onrender.com"


def test_api_scenario(name: str, payload: dict):
    """Test a single scenario via the API."""
    print(f"\n{'='*80}")
    print(f"🧪 TESTING: {name}")
    print(f"{'='*80}")

    try:
        print(f"📤 Sending request to {API_BASE_URL}/itinerary")
        response = requests.post(
            f"{API_BASE_URL}/itinerary",
            json=payload,
            timeout=120
        )

        print(f"📥 Response status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS!")
            print(f"📍 POIs: {len(result['steps'])} stops")
            print(f"⏱️  Total time: {result['total_minutes']:.1f} minutes")
            print(f"🚶 Travel time: {result['travel_minutes']:.1f} minutes")

            print(f"\n📋 Itinerary:")
            for i, step in enumerate(result['steps'], 1):
                print(f"   {i}. {step['name']}")
                print(f"      Arrival: {step['arrival_time']} | Stay: {step['stay_minutes']}min")

            return True
        else:
            print(f"\n❌ FAILED: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"\n❌ FAILED: {str(e)}")
        return False


def main():
    print(f"{'='*80}")
    print(f"🧪 TESTING RENDER API: {API_BASE_URL}")
    print(f"{'='*80}")

    # Test 1: Standard tourist with Castle POIs (tests Castle ordering)
    tomorrow = datetime.now() + timedelta(days=1)
    start_time = datetime.combine(tomorrow.date(), datetime.min.time().replace(hour=9))

    test_1 = {
        "start_time": start_time.isoformat(),
        "total_duration_minutes": 240,
        "constraints": {
            "interests": ["history", "must_see"],
            "user_profile": "standard",
            "accessibility": "any",
        }
    }

    result_1 = test_api_scenario("Standard Tourist - 4 hours (Castle focus)", test_1)

    # Test 2: Garden enthusiast (tests garden paths)
    test_2 = {
        "start_time": start_time.isoformat(),
        "total_duration_minutes": 150,
        "constraints": {
            "interests": ["garden", "scenic", "fountains"],
            "user_profile": "standard",
            "accessibility": "any",
        }
    }

    result_2 = test_api_scenario("Garden Enthusiast - 2.5 hours", test_2)

    # Test 3: Budget-constrained family (tests budget filtering)
    test_3 = {
        "start_time": start_time.isoformat(),
        "total_duration_minutes": 240,
        "constraints": {
            "interests": ["history", "garden"],
            "user_profile": "family",
            "accessibility": "stroller",
            "lunch_break": True,
            "budget": {
                "total_budget": 50.0,
                "num_adults": 2,
                "num_children_under_18": 2,
                "all_eu_residents": True,
            }
        }
    }

    result_3 = test_api_scenario("Family on Budget - €50", test_3)

    # Summary
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    successes = sum([result_1, result_2, result_3])
    print(f"✅ Successes: {successes}/3")
    print(f"❌ Failures: {3 - successes}/3")

    return 0 if successes == 3 else 1


if __name__ == "__main__":
    exit(main())
