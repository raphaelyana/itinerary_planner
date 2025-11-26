# Versailles Itinerary Planner - Test Scenarios

This document contains test cases for validating the itinerary planning system.

## How to Run Tests

Use `curl` to test the API endpoints:

```bash
API_URL="https://versailles-planner-api.onrender.com"

# Test template
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{...json payload...}'
```

---

## 1. BASIC FUNCTIONALITY TESTS

### Test 1.1: Simple 2-hour Castle visit (Morning)
**Purpose**: Verify basic Castle itinerary generation with morning start time

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 120,
    "constraints": {
      "interests": ["must_see", "history"],
      "user_profile": "standard",
      "accessibility": "any"
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ Start POI: `versailles:Castle:cour-dhonneur` (morning rule)
- ✅ Contains must-see Castle POIs (galerie-des-glaces, etc.)
- ✅ Total duration ≈ 120 minutes
- ✅ No gaps or impossible paths

---

### Test 1.2: Garden-only visit (3 hours)
**Purpose**: Verify garden-only routing without Castle

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ Start POI: `versailles:Garden:acces-jardins-cour-des-princes` (garden-only rule)
- ✅ Contains garden/fountain POIs
- ✅ Exit through: `versailles:Garden:pdv-parterre-du-midi-orangerie`
- ✅ No Castle POIs included

---

### Test 1.3: Midday long visit starting at Trianon
**Purpose**: Verify Trianon start for midday+ long visits

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T12:30:00",
    "total_duration_minutes": 240,
    "constraints": {
      "interests": ["history", "architecture", "nature"],
      "user_profile": "standard",
      "accessibility": "any"
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ Start POI: `versailles:Trianon:entree-petit-trianon` (midday+ long visit rule)
- ✅ Duration ≈ 240 minutes
- ✅ Mix of Trianon, Garden, and possibly Castle POIs

---

## 2. SMART ROUTING TESTS

### Test 2.1: Castle + Garden transition
**Purpose**: Verify proper routing from Castle interior to Gardens

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ Start at: `versailles:Castle:cour-dhonneur`
- ✅ Path includes Castle → (pavillon-dufour-sortie OR acces-jardins-cour-des-princes) → pdv-parterre-du-midi-orangerie → Gardens
- ✅ Exit through: `versailles:Garden:pdv-parterre-du-midi-orangerie`
- ✅ Both required POIs visited

---

### Test 2.2: Morning vs Afternoon start time
**Purpose**: Compare routing decisions for different start times

**Morning Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 120,
    "constraints": {
      "interests": ["history"],
      "user_profile": "standard",
      "accessibility": "any"
    }
  }'
```

**Afternoon Request** (change start_time to 14:00):
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T14:00:00",
    "total_duration_minutes": 120,
    "constraints": {
      "interests": ["history"],
      "user_profile": "standard",
      "accessibility": "any"
    }
  }'
```

**Expected Results**:
- Morning: Start at `cour-dhonneur`
- Afternoon (short visit): Start at `cour-dhonneur` (not long enough for Trianon)
- Different POI selections based on opening hours

---

## 3. PRE-COMPUTED CASTLE PATHS TESTS

### Test 3.1: Castle interior routing with detours
**Purpose**: Verify pre-computed paths work for Castle POIs including optional detours

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ All 4 required Castle POIs included
- ✅ Path uses pre-computed lookup (fast response < 5 seconds)
- ✅ Logical sequence: galeries → ... → galerie-des-glaces → chambre-du-roi → cabinet-du-conseil
- ✅ No pathfinding timeouts

---

### Test 3.2: Castle detour - Chapelle Royale
**Purpose**: Verify optional detour path (Chapelle Royale)

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 120,
    "constraints": {
      "interests": ["art", "architecture"],
      "user_profile": "standard",
      "accessibility": "any",
      "must_include": [
        "versailles:Room:chapelle-royale",
        "versailles:Room:salon-dhercule"
      ]
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ Chapelle Royale visited as detour off main trunk
- ✅ Path: salles-louis-xiv → chapelle-royale → salon-dhercule
- ✅ Fast response (pre-computed path)

---

### Test 3.3: Historical rooms path (Napoleonic galleries)
**Purpose**: Verify long sequential path through historical rooms

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 180,
    "constraints": {
      "interests": ["napoleon_i", "art_militaire", "history"],
      "user_profile": "standard",
      "accessibility": "any"
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ Includes salle-1792, salle-1796-1797, ..., salle-marengo, galerie-basse
- ✅ Sequential path through Napoleonic rooms
- ✅ No pathfinding errors

---

## 4. USER-SPECIFIED START/FINISH TESTS

### Test 4.1: Valid custom start POI
**Purpose**: Verify user can override smart routing with valid custom start

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 120,
    "constraints": {
      "interests": ["history"],
      "user_profile": "standard",
      "accessibility": "any",
      "start_poi": "versailles:Garden:acces-jardins-cour-des-princes"
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ Start POI: `versailles:Garden:acces-jardins-cour-des-princes` (user override)
- ✅ Itinerary starts from specified POI despite being morning

---

### Test 4.2: Invalid start POI (doesn't exist)
**Purpose**: Verify helpful error when user specifies non-existent POI

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 120,
    "constraints": {
      "interests": ["history"],
      "user_profile": "standard",
      "accessibility": "any",
      "start_poi": "versailles:InvalidZone:nonexistent-poi"
    }
  }'
```

**Expected Results**:
- ✅ Status: 400 Bad Request
- ✅ Error message: "Specified start POI 'versailles:InvalidZone:nonexistent-poi' does not exist in database"
- ✅ Helpful suggestion to check POI ID

---

### Test 4.3: Incompatible start/finish combination
**Purpose**: Verify error when user specifies start/finish with no valid path

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 60,
    "constraints": {
      "interests": ["history"],
      "user_profile": "standard",
      "accessibility": "any",
      "start_poi": "versailles:Trianon:entree-grand-trianon",
      "finish_poi": "versailles:Castle:cour-dhonneur-sortie",
      "must_include": ["versailles:Room:galerie-des-glaces"]
    }
  }'
```

**Expected Results**:
- ✅ Status: 400 or 500
- ✅ Error message explains path not found
- ✅ Suggestion to use different start/finish or let system choose

---

## 5. ACCESSIBILITY & USER PROFILE TESTS

### Test 5.1: Step-free accessibility
**Purpose**: Verify step-free routing

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 120,
    "constraints": {
      "interests": ["must_see"],
      "user_profile": "standard",
      "accessibility": "step_free"
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ All paths are step-free
- ✅ May exclude some POIs that require stairs
- ✅ Travel segments all have `is_step_free: true`

---

### Test 5.2: Stroller-friendly accessibility
**Purpose**: Verify stroller-friendly routing (most restrictive)

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T10:00:00",
    "total_duration_minutes": 120,
    "constraints": {
      "interests": ["nature", "photo_spot"],
      "user_profile": "family",
      "accessibility": "stroller"
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ All paths are step-free AND stroller-friendly
- ✅ Likely focuses on gardens/outdoor areas
- ✅ Travel segments all have `stroller_friendly: true`

---

### Test 5.3: Elder profile (slower pace)
**Purpose**: Verify elder profile adjusts travel times

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 180,
    "constraints": {
      "interests": ["must_see", "art"],
      "user_profile": "elder",
      "accessibility": "step_free"
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ Fewer POIs than "standard" profile for same duration
- ✅ Uses `elder_walk_min` property for travel time calculations
- ✅ More conservative routing

---

## 6. EDGE CASES & ERROR HANDLING

### Test 6.1: Very short duration (30 minutes)
**Purpose**: Verify system handles minimal time constraints

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 30,
    "constraints": {
      "interests": ["must_see"],
      "user_profile": "standard",
      "accessibility": "any"
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK OR 400 if truly impossible
- ✅ 1-2 POIs maximum
- ✅ Realistic itinerary or clear error message

---

### Test 6.2: Very long duration (8 hours)
**Purpose**: Verify system handles full-day visits

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 480,
    "constraints": {
      "interests": ["history", "art", "nature", "architecture"],
      "user_profile": "standard",
      "accessibility": "any",
      "lunch_break": true
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ Includes lunch break (45 minutes)
- ✅ Covers multiple zones (Castle, Garden, possibly Trianon)
- ✅ 12-20 POIs
- ✅ Total time ≈ 480 minutes including lunch

---

### Test 6.3: Impossible constraints (must_include contradictions)
**Purpose**: Verify error handling for impossible requirements

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 30,
    "constraints": {
      "interests": ["history"],
      "user_profile": "standard",
      "accessibility": "any",
      "must_include": [
        "versailles:Room:galerie-des-glaces",
        "versailles:Trianon:hameau-de-la-reine",
        "versailles:Garden:bassin-de-neptune",
        "versailles:Room:galerie-des-batailles"
      ]
    }
  }'
```

**Expected Results**:
- ✅ Status: 400 Bad Request
- ✅ Clear error: "Unable to fit required POIs within duration"
- ✅ No crash or timeout

---

### Test 6.4: No POIs match filters
**Purpose**: Verify error when interests don't match any POIs

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 120,
    "constraints": {
      "interests": ["nonexistent_tag", "invalid_interest"],
      "user_profile": "standard",
      "accessibility": "any"
    }
  }'
```

**Expected Results**:
- ✅ Status: 400 Bad Request
- ✅ Error: "No POIs match the specified constraints"
- ✅ No crash

---

## 7. PERFORMANCE & STRESS TESTS

### Test 7.1: Large itinerary (many POIs)
**Purpose**: Verify routing performance with many POIs

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T08:00:00",
    "total_duration_minutes": 600,
    "constraints": {
      "interests": ["history", "art", "architecture", "nature", "photo_spot"],
      "user_profile": "standard",
      "accessibility": "any"
    }
  }'
```

**Expected Results**:
- ✅ Response time < 30 seconds
- ✅ Status: 200 OK
- ✅ Logical routing (no backtracking)
- ✅ Uses greedy algorithm for large POI count

---

### Test 7.2: Castle-heavy itinerary (tests pre-computed paths)
**Purpose**: Verify pre-computed Castle paths improve performance

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 180,
    "constraints": {
      "interests": ["louis_xiv", "art_classique", "art_baroque", "must_see"],
      "user_profile": "standard",
      "accessibility": "any",
      "must_include": [
        "versailles:Room:galeries-de-lhistoire",
        "versailles:Room:chapelle-royale",
        "versailles:Room:galerie-des-glaces",
        "versailles:Room:chambre-du-roi",
        "versailles:Room:galerie-des-batailles"
      ]
    }
  }'
```

**Expected Results**:
- ✅ Response time < 5 seconds (pre-computed paths)
- ✅ Status: 200 OK
- ✅ All required Castle POIs included
- ✅ Logical Castle tour sequence

---

## 8. BUDGET CONSTRAINT TESTS

### Test 8.1: Low budget (Garden only)
**Purpose**: Verify budget limits zone access

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T10:00:00",
    "total_duration_minutes": 120,
    "constraints": {
      "interests": ["nature", "photo_spot"],
      "user_profile": "standard",
      "accessibility": "any",
      "budget": {
        "total_budget": 10,
        "num_adults": 2,
        "num_children_under_18": 0,
        "all_eu_residents": true
      }
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ Only includes free Garden/Park POIs
- ✅ No Castle or Trianon POIs (require tickets)

---

### Test 8.2: High budget (all zones)
**Purpose**: Verify high budget allows all zones

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 240,
    "constraints": {
      "interests": ["must_see", "history", "nature"],
      "user_profile": "standard",
      "accessibility": "any",
      "budget": {
        "total_budget": 100,
        "num_adults": 2,
        "num_children_under_18": 1,
        "all_eu_residents": false
      }
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ Includes Castle, Garden, Trianon POIs
- ✅ Maximizes value within budget

---

## 9. REGRESSION TESTS

### Test 9.1: Previously hanging Castle pathfinding
**Purpose**: Verify Castle pathfinding no longer hangs (regression test)

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Expected Results**:
- ✅ Response time < 5 seconds (NO HANG)
- ✅ Status: 200 OK
- ✅ Both POIs included with valid path

---

## 10. VALIDATION CHECKLIST

For each test, verify:

- [ ] **HTTP Status**: Correct status code (200, 400, 500)
- [ ] **Response Structure**: Valid JSON with expected fields
- [ ] **Start/Finish POIs**: Match smart routing rules or user overrides
- [ ] **POI Sequence**: Logical order, no impossible jumps
- [ ] **Time Constraints**: Total duration approximately matches request
- [ ] **Travel Segments**: All segments have valid from/to POIs
- [ ] **Accessibility**: Paths meet accessibility requirements
- [ ] **Performance**: Response time reasonable (< 30s for complex routes)
- [ ] **Error Messages**: Clear, actionable error messages

---

## Summary

**Total Tests**: 30+ scenarios covering:
- ✅ Basic functionality (3 tests)
- ✅ Smart routing (2 tests)
- ✅ Pre-computed Castle paths (3 tests)
- ✅ Custom start/finish (3 tests)
- ✅ Accessibility profiles (3 tests)
- ✅ Edge cases (4 tests)
- ✅ Performance (2 tests)
- ✅ Budget constraints (2 tests)
- ✅ Regression tests (1 test)

Run these tests after each deployment to ensure system stability and correctness.
