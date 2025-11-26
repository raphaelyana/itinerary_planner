#!/bin/bash

# Quick Test - Run a single itinerary request for rapid testing
# Usage: ./quick_test.sh

API_URL="${API_URL:-https://versailles-planner-api.onrender.com}"

echo "Testing Versailles Itinerary Planner API..."
echo "API URL: $API_URL"
echo ""

# Simple morning Castle visit
echo "Requesting 2-hour Castle visit (morning start)..."
response=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 120,
    "constraints": {
      "interests": ["must_see", "history"],
      "user_profile": "standard",
      "accessibility": "any"
    }
  }')

# Extract HTTP code (last line)
http_code=$(echo "$response" | tail -n1)
# Extract body (all but last line)
body=$(echo "$response" | sed '$d')

echo "HTTP Status: $http_code"
echo ""

if [ "$http_code" -eq 200 ]; then
    echo "✓ Success! Itinerary generated."
    echo ""

    # Pretty print using jq if available
    if command -v jq &> /dev/null; then
        echo "Summary:"
        echo "--------"
        echo "Number of POIs: $(echo "$body" | jq '.steps | length')"
        echo "Total duration: $(echo "$body" | jq '.total_minutes') minutes"
        echo "Travel time: $(echo "$body" | jq '.travel_minutes') minutes"
        echo "Visit time: $(echo "$body" | jq '.visit_minutes') minutes"
        echo ""
        echo "Route:"
        echo "$body" | jq -r '.steps[] | "  \(.arrival_time[11:16]) - \(.departure_time[11:16])  \(.name)"'
        echo ""
        echo "Full response saved to: quick_test_result.json"
        echo "$body" | jq '.' > quick_test_result.json
    else
        echo "$body"
        echo ""
        echo "Install 'jq' for formatted output: brew install jq"
    fi
else
    echo "✗ Request failed with status $http_code"
    echo ""
    echo "Response:"
    echo "$body"
fi
