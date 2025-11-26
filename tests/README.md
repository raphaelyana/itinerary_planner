# Testing Guide - Versailles Itinerary Planner

This directory contains comprehensive test scenarios for validating the itinerary planning system.

## Quick Start

### Run a Single Quick Test
```bash
cd tests
./quick_test.sh
```

This runs a simple 2-hour Castle visit and displays the results in a readable format.

### Run Specific Test
```bash
./run_tests.sh 1.1  # Run Test 1.1 (Simple 2-hour Castle visit)
./run_tests.sh 3.1  # Run Test 3.1 (Castle interior with detours)
```

### Run All Tests
```bash
./run_tests.sh
```

This runs the complete test suite and generates a summary report.

## Test Files

- **`test_scenarios.md`**: Complete documentation of all 30+ test scenarios
- **`run_tests.sh`**: Automated test runner with pass/fail reporting
- **`quick_test.sh`**: Simple single-test runner for rapid validation
- **`test_results/`**: Directory where test results are saved (JSON responses)

## Test Categories

### 1. Basic Functionality (Tests 1.x)
- Morning Castle visits
- Garden-only visits
- Midday Trianon starts
- Basic routing validation

### 2. Smart Routing (Tests 2.x)
- Castle → Garden transitions
- Time-based start point selection
- Zone-aware routing

### 3. Pre-computed Castle Paths (Tests 3.x)
- Castle interior routing performance
- Optional detour paths (Chapelle, Chambre du Roi, etc.)
- Historical rooms sequences

### 4. User Overrides (Tests 4.x)
- Custom start/finish POIs
- Error handling for invalid POIs
- Incompatible path validation

### 5. Accessibility & Profiles (Tests 5.x)
- Step-free routing
- Stroller-friendly paths
- Elder/family profiles

### 6. Edge Cases (Tests 6.x)
- Very short/long durations
- Impossible constraints
- Empty result sets

### 7. Performance (Tests 7.x)
- Large itineraries
- Castle-heavy routing
- Response time validation

### 8. Budget Constraints (Tests 8.x)
- Low budget (Gardens only)
- High budget (all zones)

### 9. Regression Tests (Tests 9.x)
- Previously failing scenarios
- Bug fixes validation

## Expected Results

All tests validate:
- ✅ **HTTP Status**: Correct status codes (200, 400, 500)
- ✅ **Response Format**: Valid JSON structure
- ✅ **Start/Finish Logic**: Smart routing rules applied correctly
- ✅ **POI Sequences**: Logical order with valid paths
- ✅ **Time Constraints**: Duration matches request
- ✅ **Performance**: Reasonable response times (< 30s)
- ✅ **Error Messages**: Clear, actionable errors

## Environment Variables

Configure the API URL:
```bash
export API_URL="https://your-api-url.onrender.com"
./run_tests.sh
```

Default: `https://versailles-planner-api.onrender.com`

## Test Results

Results are saved to `test_results/` with timestamps:
```
test_results/
├── test_1.1_20241201_143022.json
├── test_1.2_20241201_143025.json
└── ...
```

## Example Output

### Quick Test
```bash
$ ./quick_test.sh

Testing Versailles Itinerary Planner API...
API URL: https://versailles-planner-api.onrender.com

Requesting 2-hour Castle visit (morning start)...
HTTP Status: 200

✓ Success! Itinerary generated.

Summary:
--------
Number of POIs: 8
Total duration: 118.5 minutes
Travel time: 23.5 minutes
Visit time: 95 minutes

Route:
  09:00 - 09:05  Cour d'Honneur
  09:10 - 09:50  Galerie des Glaces
  09:52 - 10:12  Chambre du Roi
  ...
```

### Full Test Suite
```bash
$ ./run_tests.sh

================================================
Versailles Itinerary Planner - Test Suite
API: https://versailles-planner-api.onrender.com
Timestamp: 20241201_143000
================================================

Running Test 1.1: Simple 2-hour Castle visit
✓ Status: 200 (expected 200)
✓ Valid JSON response
✓ POIs: 8
✓ Total time: 118.5 minutes
✓ Route: versailles:Castle:cour-dhonneur -> ... -> versailles:Garden:pdv-parterre-du-midi-orangerie
✓ Test 1.1 passed

...

================================================
Test Summary
================================================
Passed: 11
Failed: 0

All tests passed! 🎉
```

## Adding New Tests

To add a new test to `run_tests.sh`:

```bash
test_X_Y() {
    run_test "X.Y" "Test Description" '{
        "start_time": "2024-12-01T09:00:00",
        "total_duration_minutes": 120,
        "constraints": {
            "interests": ["your_interests"],
            "user_profile": "standard",
            "accessibility": "any"
        }
    }' 200  # Expected HTTP status
}
```

## Troubleshooting

### Test hangs or times out
- Check API is accessible: `curl $API_URL/health`
- Increase curl timeout in scripts
- Check Render logs for errors

### Invalid JSON errors
- Verify API is returning proper JSON responses
- Check for API-level errors or exceptions

### All tests fail with 503
- API might be sleeping (Render free tier)
- Make health check request first to wake it up

## CI/CD Integration

Run tests in CI pipeline:
```bash
#!/bin/bash
export API_URL="https://your-staging-api.onrender.com"
cd tests
./run_tests.sh
exit $?  # Exit with test suite status code
```

## Coverage

Current test coverage:
- **Basic routing**: ✅ Covered
- **Smart start/finish**: ✅ Covered
- **Pre-computed paths**: ✅ Covered
- **Accessibility**: ✅ Covered
- **Error handling**: ✅ Covered
- **Performance**: ✅ Covered
- **Budget constraints**: ✅ Covered

Total: **30+ test scenarios** across **9 categories**
