# Network Entry and Exit Points

## Valid Start Points (Entrances)

### Castle Zone
- **`versailles:Castle:cour-dhonneur`** ⭐ DEFAULT
  - Main Palace Entrance (Cour d'Honneur)
  - Best for: Palace tours, Hall of Mirrors, Royal Apartments

### Garden Zone
- **`versailles:Garden:acces-jardins-cour-des-princes`**
  - Gardens Entrance (Cour des Princes)
  - Best for: Garden-focused tours

- **`versailles:Garden:entree-parc-bassin-apollon`**
  - Park Entrance (Basin of Apollo)
  - Best for: Park tours, fountains

- **`versailles:Garden:grille-de-neptune`**
  - Neptune Gate
  - Best for: Northern garden access

### Trianon Zone
- **`versailles:Trianon:entree-grand-trianon`**
  - Grand Trianon Entrance
  - Best for: Trianon-focused tours

- **`versailles:Trianon:entree-petit-trianon`**
  - Petit Trianon Entrance
  - Best for: Marie-Antoinette tours

---

## Valid Finish Points (Exits)

### Castle Zone
- **`versailles:Castle:cour-dhonneur-sortie`** ⭐ DEFAULT
  - Palace Exit (Cour d'Honneur)
  - Returns visitors to main palace courtyard

### Garden Zone
- **`versailles:Garden:entree-parc-bassin-apollon`**
  - Basin of Apollo Exit
  - Exit via park/basin

- **`versailles:Garden:grille-de-neptune`**
  - Neptune Gate Exit
  - Northern garden exit

### Trianon Zone
- **`versailles:Trianon:entree-grand-trianon`**
  - Grand Trianon Exit

- **`versailles:Trianon:entree-petit-trianon`**
  - Petit Trianon Exit

---

## Automatic Selection Logic

The planner automatically selects the best entrance/exit based on the POIs in the itinerary:

### Zone-Based Selection

**Castle-focused itinerary** (majority Castle POIs)
- Start: `versailles:Castle:cour-dhonneur`
- Exit: `versailles:Castle:cour-dhonneur-sortie`

**Garden-focused itinerary** (majority Garden/Park POIs)
- Start: `versailles:Garden:acces-jardins-cour-des-princes`
- Exit: `versailles:Garden:grille-de-neptune`

**Trianon-focused itinerary** (majority Trianon POIs)
- Start: `versailles:Trianon:entree-grand-trianon`
- Exit: `versailles:Trianon:entree-grand-trianon`

### Override with Explicit Parameters

You can override the automatic selection using API parameters:

```json
{
  "constraints": {
    "start_poi": "versailles:Trianon:entree-petit-trianon",
    "finish_poi": "versailles:Garden:grille-de-neptune"
  }
}
```

---

## Implementation

See `configs/network_config.py` for:
- `ENTRANCE_POINTS` - All valid entrances
- `EXIT_POINTS` - All valid exits
- `get_entrance_for_pois()` - Auto-select best entrance
- `get_exit_for_pois()` - Auto-select best exit
- `validate_entrance()` - Validate entrance POI ID
- `validate_exit()` - Validate exit POI ID

---

## Usage Examples

### Example 1: Auto-detect entrance/exit
```json
POST /itinerary
{
  "start_time": "2025-11-26T09:00:00",
  "total_duration_minutes": 240,
  "constraints": {
    "interests": ["napoleon_i", "art_militaire"]
  }
}
```
→ Auto-selects Castle entrance (Napoleon rooms are in Castle)

### Example 2: Explicit Trianon tour
```json
POST /itinerary
{
  "start_time": "2025-11-26T10:00:00",
  "total_duration_minutes": 180,
  "constraints": {
    "interests": ["marie_antoinette"],
    "start_poi": "versailles:Trianon:entree-petit-trianon",
    "finish_poi": "versailles:Trianon:entree-petit-trianon"
  }
}
```
→ Starts and ends at Petit Trianon

### Example 3: Cross-zone tour
```json
POST /itinerary
{
  "start_time": "2025-11-26T09:00:00",
  "total_duration_minutes": 360,
  "constraints": {
    "interests": ["history", "garden"],
    "start_poi": "versailles:Castle:cour-dhonneur",
    "finish_poi": "versailles:Garden:grille-de-neptune"
  }
}
```
→ Starts at palace, ends at Neptune gate (garden exit)
