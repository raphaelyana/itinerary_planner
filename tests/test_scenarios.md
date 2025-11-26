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
- ✅ Contains must-see Castle POIs (galerie-des-glaces, chapelle-royale, etc.)
- ✅ Total duration ≈ 120-140 minutes (slight overage allowed)
- ✅ No backtracking through gateway nodes

---

### Test 1.2: Garden-only visit (3 hours)
**Purpose**: Verify garden-only routing skips Castle entirely

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T10:00:00",
    "total_duration_minutes": 180,
    "constraints": {
      "interests": ["garden", "water", "fountains", "photo_spot"],
      "user_profile": "standard",
      "accessibility": "any"
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ Start POI: `versailles:Garden:pdv-parterre-du-midi-orangerie` (direct garden start)
- ✅ Contains only Garden/Park POIs (bassin-de-latone, bassin-de-neptune, etc.)
- ✅ Exit through: `versailles:Garden:pdv-parterre-du-midi-orangerie`
- ✅ No Castle POIs or Castle entry points in route
- ✅ No backtracking through PDV

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
      "interests": ["history", "architecture", "marie_antoinette"],
      "user_profile": "standard",
      "accessibility": "any"
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ Start POI: `versailles:Trianon:entree-petit-trianon` (midday+ long visit rule)
- ✅ Duration ≈ 240 minutes
- ✅ Focus on Trianon POIs (petit-trianon, hameau-de-la-reine, etc.)

---

## 2. SMART ROUTING TESTS

### Test 2.1: Castle + Garden transition (No Backtracking)
**Purpose**: Verify proper one-way routing from Castle to Gardens

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 180,
    "constraints": {
      "interests": ["must_see", "architecture", "water"],
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
- ✅ Path: Castle POIs → (gateway node) → pdv-parterre-du-midi-orangerie → Garden POIs
- ✅ **NO backtracking** through `acces-jardins-cour-des-princes` or `pavillon-dufour-sortie`
- ✅ Exit through: `versailles:Garden:pdv-parterre-du-midi-orangerie`
- ✅ Both required POIs visited

---

### Test 2.2: Morning vs Midday start time
**Purpose**: Compare routing decisions for different start times

**Morning Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 120,
    "constraints": {
      "interests": ["history", "must_see"],
      "user_profile": "standard",
      "accessibility": "any"
    }
  }'
```

**Midday Long Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T12:30:00",
    "total_duration_minutes": 200,
    "constraints": {
      "interests": ["history", "must_see"],
      "user_profile": "standard",
      "accessibility": "any"
    }
  }'
```

**Expected Results**:
- Morning: Start at `versailles:Castle:cour-dhonneur`
- Midday long: Start at `versailles:Trianon:entree-petit-trianon`
- Different POI selections optimized for entry point

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
- ✅ Fast response < 5 seconds (pre-computed paths)
- ✅ Logical sequence through Castle interior
- ✅ Path includes detour to chambre-du-roi/cabinet-du-conseil

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
      "interests": ["art", "architecture", "art_religieux"],
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
- ✅ Includes Napoleonic-era rooms (salle-1792, salle-du-sacre, etc.)
- ✅ Sequential path through historical galleries
- ✅ Fast routing (no timeouts)

---

## 4. USER-SPECIFIED START/FINISH TESTS

### Test 4.1: Valid custom start POI (Garden override)
**Purpose**: Verify user can override smart routing with valid custom start

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 120,
    "constraints": {
      "interests": ["history", "must_see"],
      "user_profile": "standard",
      "accessibility": "any",
      "start_poi": "versailles:Garden:pdv-parterre-du-midi-orangerie"
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ Start POI: `versailles:Garden:pdv-parterre-du-midi-orangerie` (user override)
- ✅ Itinerary respects user choice despite being morning

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
- ✅ All travel segments have `is_step_free: true`
- ✅ May exclude POIs requiring stairs (e.g., attique-chimay)

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
      "interests": ["garden", "photo_spot"],
      "user_profile": "family",
      "accessibility": "stroller"
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ All segments have `is_step_free: true` AND `stroller_friendly: true`
- ✅ Likely focuses on gardens/outdoor areas

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
- ✅ Uses `elder_walk_min` for travel calculations
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
- ✅ Status: 200 OK OR 400 if impossible
- ✅ 1-2 POIs maximum
- ✅ Realistic itinerary or clear error

---

### Test 6.2: Very long duration (8 hours with lunch)
**Purpose**: Verify system handles full-day visits

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 480,
    "constraints": {
      "interests": ["history", "art", "garden", "architecture"],
      "user_profile": "standard",
      "accessibility": "any",
      "lunch_break": true
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ Includes lunch break (45 minutes, 11:00-14:00 window)
- ✅ Covers multiple zones (Castle → Gardens or Trianon)
- ✅ 12-20 POIs

---

### Test 6.3: Impossible time constraint
**Purpose**: Verify error handling for impossible requirements

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 20,
    "constraints": {
      "interests": ["history"],
      "user_profile": "standard",
      "accessibility": "any",
      "must_include": [
        "versailles:Room:galerie-des-glaces",
        "versailles:Room:galerie-des-batailles",
        "versailles:Garden:bassin-de-neptune"
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

---

## 7. PERFORMANCE TESTS

### Test 7.1: Large itinerary (many POIs)
**Purpose**: Verify routing performance with many POIs

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T08:00:00",
    "total_duration_minutes": 540,
    "constraints": {
      "interests": ["history", "art", "architecture", "garden", "photo_spot"],
      "user_profile": "standard",
      "accessibility": "any"
    }
  }'
```

**Expected Results**:
- ✅ Response time < 30 seconds
- ✅ Status: 200 OK
- ✅ Uses greedy algorithm for large POI count
- ✅ Logical routing (no backtracking through gateways)

---

### Test 7.2: Castle-heavy itinerary (pre-computed paths)
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
      "accessibility": "any"
    }
  }'
```

**Expected Results**:
- ✅ Response time < 5 seconds (pre-computed paths)
- ✅ Status: 200 OK
- ✅ Logical Castle tour sequence
- ✅ No pathfinding timeouts

---

## 8. REGRESSION TESTS

### Test 8.1: Previously hanging Castle pathfinding
**Purpose**: Verify Castle pathfinding no longer hangs

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

### Test 8.2: Opening hours validation (Chapelle Royale)
**Purpose**: Verify chapelle-royale has opening hours set

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 120,
    "constraints": {
      "interests": ["art_religieux", "architecture"],
      "user_profile": "standard",
      "accessibility": "any",
      "must_include": ["versailles:Room:chapelle-royale"]
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK (not "missing opening hours" error)
- ✅ Chapelle Royale included in itinerary
- ✅ Valid arrival/departure times

---

## 9. GATEWAY ROUTING TESTS

### Test 9.1: No backtracking through garden entry
**Purpose**: Verify gateway blocking prevents backtracking

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 180,
    "constraints": {
      "interests": ["water", "fountains", "garden"],
      "user_profile": "standard",
      "accessibility": "any",
      "must_include": [
        "versailles:Garden:bassin-de-latone",
        "versailles:Garden:bassin-de-neptune",
        "versailles:Garden:bassin-dapollon"
      ]
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ All 3 fountain basins visited
- ✅ Path does NOT go back through `acces-jardins-cour-des-princes` or `pdv-parterre-du-midi-orangerie` after passing them
- ✅ Logical garden tour flow

---

### Test 9.2: Castle-only visit exits correctly
**Purpose**: Verify Castle-only visits exit through cour-dhonneur-sortie

**Request**:
```bash
curl -X POST "$API_URL/itinerary" \
  -H "Content-Type: application/json" \
  -d '{
    "start_time": "2024-12-01T09:00:00",
    "total_duration_minutes": 90,
    "constraints": {
      "interests": ["must_see"],
      "user_profile": "standard",
      "accessibility": "any"
    }
  }'
```

**Expected Results**:
- ✅ Status: 200 OK
- ✅ Only Castle POIs visited
- ✅ Finish POI: `versailles:Castle:cour-dhonneur-sortie` (indoor-only exit)

---

## 10. VALIDATION CHECKLIST

For each test, verify:

- [ ] **HTTP Status**: Correct status code (200, 400, 500)
- [ ] **Response Structure**: Valid JSON with `steps`, `travel_minutes`, `total_minutes`, `travel_segments`
- [ ] **Start/Finish POIs**: Match smart routing rules or user overrides
- [ ] **POI Sequence**: Logical order, no impossible jumps
- [ ] **Time Constraints**: Total duration approximately matches request (slight overage OK)
- [ ] **Travel Segments**: All segments have valid from/to POIs
- [ ] **Gateway Blocking**: No backtracking through passed gateway nodes
- [ ] **Accessibility**: Paths meet accessibility requirements
- [ ] **Performance**: Response time reasonable (< 30s for complex, < 5s for Castle)
- [ ] **Error Messages**: Clear, actionable error messages

---

## Summary

**Total Tests**: 25+ scenarios covering:
- ✅ Basic functionality (3 tests)
- ✅ Smart routing (2 tests)
- ✅ Pre-computed Castle paths (3 tests)
- ✅ Custom start/finish (2 tests)
- ✅ Accessibility profiles (3 tests)
- ✅ Edge cases (4 tests)
- ✅ Performance (2 tests)
- ✅ Regression tests (2 tests)
- ✅ Gateway routing (2 tests)

Run these tests after each deployment to ensure system stability and correctness.
