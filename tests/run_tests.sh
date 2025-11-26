#!/bin/bash

# Versailles Itinerary Planner - Test Runner
# Usage: ./run_tests.sh [test_number]
# Example: ./run_tests.sh 1.1  (run specific test)
#          ./run_tests.sh       (run all tests)

set -e

# Configuration
API_URL="${API_URL:-https://versailles-planner-api.onrender.com}"
RESULTS_DIR="test_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create results directory
mkdir -p "$RESULTS_DIR"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}Versailles Itinerary Planner - Test Suite${NC}"
echo -e "${BLUE}API: $API_URL${NC}"
echo -e "${BLUE}Timestamp: $TIMESTAMP${NC}"
echo -e "${BLUE}================================================${NC}\n"

# Function to run a test
run_test() {
    local test_id=$1
    local test_name=$2
    local json_payload=$3
    local expected_status=${4:-200}

    echo -e "${YELLOW}Running Test $test_id: $test_name${NC}"

    local result_file="$RESULTS_DIR/test_${test_id}_${TIMESTAMP}.json"
    local http_code

    # Make request and capture response + HTTP code
    http_code=$(curl -s -w "%{http_code}" -o "$result_file" \
        -X POST "$API_URL/itinerary" \
        -H "Content-Type: application/json" \
        -d "$json_payload")

    # Check HTTP status
    if [ "$http_code" -eq "$expected_status" ]; then
        echo -e "${GREEN}✓ Status: $http_code (expected $expected_status)${NC}"
    else
        echo -e "${RED}✗ Status: $http_code (expected $expected_status)${NC}"
        cat "$result_file"
        return 1
    fi

    # Check response is valid JSON
    if jq empty "$result_file" 2>/dev/null; then
        echo -e "${GREEN}✓ Valid JSON response${NC}"
    else
        echo -e "${RED}✗ Invalid JSON response${NC}"
        return 1
    fi

    # Display summary
    if [ "$http_code" -eq 200 ]; then
        local num_steps=$(jq '.steps | length' "$result_file")
        local total_time=$(jq '.total_minutes' "$result_file")
        local first_poi=$(jq -r '.steps[0].poi_id' "$result_file")
        local last_poi=$(jq -r '.steps[-1].poi_id' "$result_file")

        echo -e "${GREEN}✓ POIs: $num_steps${NC}"
        echo -e "${GREEN}✓ Total time: $total_time minutes${NC}"
        echo -e "${GREEN}✓ Route: $first_poi -> ... -> $last_poi${NC}"
    fi

    echo -e "${GREEN}✓ Test $test_id passed${NC}\n"
    return 0
}

# Test definitions

test_1_1() {
    run_test "1.1" "Simple 2-hour Castle visit (Morning)" '{
        "start_time": "2024-12-01T09:00:00",
        "total_duration_minutes": 120,
        "constraints": {
            "interests": ["must_see", "history"],
            "user_profile": "standard",
            "accessibility": "any"
        }
    }' 200
}

test_1_2() {
    run_test "1.2" "Garden-only visit (3 hours)" '{
        "start_time": "2024-12-01T10:00:00",
        "total_duration_minutes": 180,
        "constraints": {
            "interests": ["nature", "photo_spot"],
            "user_profile": "standard",
            "accessibility": "any",
            "exclude_ids": [
                "versailles:Castle:cour-dhonneur",
                "versailles:Castle:acces-grands-appartements"
            ]
        }
    }' 200
}

test_1_3() {
    run_test "1.3" "Midday long visit starting at Trianon" '{
        "start_time": "2024-12-01T12:30:00",
        "total_duration_minutes": 240,
        "constraints": {
            "interests": ["history", "architecture", "nature"],
            "user_profile": "standard",
            "accessibility": "any"
        }
    }' 200
}

test_2_1() {
    run_test "2.1" "Castle + Garden transition" '{
        "start_time": "2024-12-01T09:00:00",
        "total_duration_minutes": 180,
        "constraints": {
            "interests": ["must_see", "architecture", "nature"],
            "user_profile": "standard",
            "accessibility": "any",
            "must_include": [
                "versailles:Room:galerie-des-glaces",
                "versailles:Garden:bassin-de-latone"
            ]
        }
    }' 200
}

test_3_1() {
    run_test "3.1" "Castle interior with detours" '{
        "start_time": "2024-12-01T09:30:00",
        "total_duration_minutes": 150,
        "constraints": {
            "interests": ["must_see", "art", "history"],
            "user_profile": "standard",
            "accessibility": "any",
            "must_include": [
                "versailles:Room:galeries-de-lhistoire",
                "versailles:Room:chambre-du-roi",
                "versailles:Room:cabinet-du-conseil",
                "versailles:Room:galerie-des-glaces"
            ]
        }
    }' 200
}

test_4_1() {
    run_test "4.1" "Valid custom start POI" '{
        "start_time": "2024-12-01T09:00:00",
        "total_duration_minutes": 120,
        "constraints": {
            "interests": ["history"],
            "user_profile": "standard",
            "accessibility": "any",
            "start_poi": "versailles:Garden:acces-jardins-cour-des-princes"
        }
    }' 200
}

test_4_2() {
    run_test "4.2" "Invalid start POI (error handling)" '{
        "start_time": "2024-12-01T09:00:00",
        "total_duration_minutes": 120,
        "constraints": {
            "interests": ["history"],
            "user_profile": "standard",
            "accessibility": "any",
            "start_poi": "versailles:InvalidZone:nonexistent-poi"
        }
    }' 400
}

test_5_1() {
    run_test "5.1" "Step-free accessibility" '{
        "start_time": "2024-12-01T09:00:00",
        "total_duration_minutes": 120,
        "constraints": {
            "interests": ["must_see"],
            "user_profile": "standard",
            "accessibility": "step_free"
        }
    }' 200
}

test_6_1() {
    run_test "6.1" "Very short duration (30 min)" '{
        "start_time": "2024-12-01T09:00:00",
        "total_duration_minutes": 30,
        "constraints": {
            "interests": ["must_see"],
            "user_profile": "standard",
            "accessibility": "any"
        }
    }' 200
}

test_6_4() {
    run_test "6.4" "No POIs match filters" '{
        "start_time": "2024-12-01T09:00:00",
        "total_duration_minutes": 120,
        "constraints": {
            "interests": ["nonexistent_tag", "invalid_interest"],
            "user_profile": "standard",
            "accessibility": "any"
        }
    }' 400
}

test_9_1() {
    run_test "9.1" "Regression: Previously hanging pathfinding" '{
        "start_time": "2024-12-01T09:00:00",
        "total_duration_minutes": 120,
        "constraints": {
            "interests": ["history", "art"],
            "user_profile": "standard",
            "accessibility": "any",
            "must_include": [
                "versailles:Room:galeries-de-lhistoire",
                "versailles:Room:cabinet-du-conseil"
            ]
        }
    }' 200
}

# Main execution
PASSED=0
FAILED=0
FAILED_TESTS=()

if [ -n "$1" ]; then
    # Run specific test
    test_func="test_$(echo $1 | tr '.' '_')"
    if declare -f "$test_func" > /dev/null; then
        if $test_func; then
            PASSED=$((PASSED + 1))
        else
            FAILED=$((FAILED + 1))
            FAILED_TESTS+=("$1")
        fi
    else
        echo -e "${RED}Test $1 not found${NC}"
        exit 1
    fi
else
    # Run all tests
    echo -e "${BLUE}Running all tests...${NC}\n"

    for test_func in $(declare -F | grep "test_" | awk '{print $3}'); do
        if $test_func; then
            PASSED=$((PASSED + 1))
        else
            FAILED=$((FAILED + 1))
            test_id=$(echo "$test_func" | sed 's/test_//' | tr '_' '.')
            FAILED_TESTS+=("$test_id")
        fi
    done
fi

# Summary
echo -e "\n${BLUE}================================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo -e "\n${RED}Failed tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "${RED}  - Test $test${NC}"
    done
    exit 1
else
    echo -e "\n${GREEN}All tests passed! 🎉${NC}"
    exit 0
fi
