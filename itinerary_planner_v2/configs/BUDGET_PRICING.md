# Budget-Based Itinerary Planning

## Overview

The planner now supports budget constraints that automatically filter POIs based on affordable ticket types. This ensures visitors only see itineraries they can actually afford.

## Pricing Information

### Ticket Types

1. **Passeport** (Full Access)
   - Includes: Castle + Trianon + Gardens
   - Low season (Nov 1 - Mar 31): €24 (€10 reduced)
   - High season (Apr 1 - Oct 31): €32 (€10 reduced)

2. **Château** (Castle Only)
   - Includes: Castle + basic garden access
   - Year-round: €21 (€13 reduced)

3. **Domaine de Trianon**
   - Includes: Trianon only (opens 12h)
   - Year-round: €12 (€8 reduced)

4. **Free Access**
   - Gardens and Park only (non-event days)
   - Under 18: Free to all zones
   - Under 26 (EU residents): Free to all zones

### Free Access Eligibility

- **Under 18 years** (worldwide): Free access to Castle and Trianon
- **Under 26 years** (EU residents only): Free access to Castle and Trianon
- **Note:** Still requires Passeport (€10 reduced) for garden events

### Reduced Rate Eligibility

- Famille nombreuse card holders
- ANCV Chèques-Vacances
- Société des Amis de Versailles members

## API Usage

### Example 1: Family with 2 Adults + 2 Children

```json
POST /itinerary
{
  "start_time": "2025-11-26T09:00:00",
  "total_duration_minutes": 240,
  "constraints": {
    "interests": ["history", "must_see"],
    "budget": {
      "total_budget": 50.0,
      "num_adults": 2,
      "num_children_under_18": 2,
      "all_eu_residents": true
    }
  }
}
```

**Result:**
- Children are free
- 2 adults × €21 = €42 (Château ticket)
- Budget: €50 → Château ticket affordable
- Accessible zones: Castle, Garden, Park
- **Itinerary includes Castle POIs** ✅

---

### Example 2: Low Budget (Gardens Only)

```json
POST /itinerary
{
  "start_time": "2025-11-26T10:00:00",
  "total_duration_minutes": 180,
  "constraints": {
    "interests": ["garden", "scenic"],
    "budget": {
      "total_budget": 10.0,
      "num_adults": 1
    }
  }
}
```

**Result:**
- Budget too low for Castle ticket (€21)
- Accessible zones: Garden, Park only
- **Itinerary excludes Castle POIs** ✅
- Gardens and park are accessible

---

### Example 3: Budget Error (Wants Castle, Can't Afford)

```json
POST /itinerary
{
  "start_time": "2025-11-26T09:00:00",
  "total_duration_minutes": 240,
  "constraints": {
    "interests": ["must_see"],  // Hall of Mirrors requires Castle ticket
    "budget": {
      "total_budget": 15.0,
      "num_adults": 1
    }
  }
}
```

**Result:**
```json
{
  "error": "Budget insufficient. Your budget (€15.00) allows free ticket which includes: Garden, Park. However, your itinerary requires access to: Castle. Please increase budget or adjust interests to exclude Castle/Trianon POIs."
}
```

---

### Example 4: Young EU Visitors (Free Access)

```json
POST /itinerary
{
  "start_time": "2025-11-26T09:00:00",
  "total_duration_minutes": 300,
  "constraints": {
    "interests": ["history", "art", "marie_antoinette"],
    "budget": {
      "total_budget": 10.0,
      "num_adults": 0,
      "num_youth_18_25_eu": 2,
      "all_eu_residents": true
    }
  }
}
```

**Result:**
- Youth 18-25 (EU) are free for Castle + Trianon
- Budget €10 → Enough for any access
- Accessible zones: Castle, Garden, Park, Trianon
- **Full access itinerary** ✅

---

## How It Works

### 1. Budget Validation

When budget is specified, the planner:

1. Creates visitor group from age/residency parameters
2. Calculates ticket costs based on visit date (season)
3. Determines which zones are affordable
4. Filters POI candidates to only include affordable zones

### 2. Zone Filtering

POIs are mapped to zones:
- `versailles:Castle:*` → Castle zone
- `versailles:Garden:*` → Garden zone
- `versailles:Park:*` → Park zone
- `versailles:Trianon:*` → Trianon zone

### 3. Automatic Selection

The planner automatically selects the best affordable ticket:

| Budget | Visitor Type | Ticket | Zones Included |
|--------|-------------|--------|----------------|
| €0-10 | Adult | Free | Garden, Park |
| €12+ | Adult | Trianon | Trianon |
| €21+ | Adult | Château | Castle, Garden, Park |
| €24-32 | Adult | Passeport | All zones |
| Any | Under 18 | Free* | Castle, Garden, Park, Trianon |
| Any | 18-25 EU | Free* | Castle, Garden, Park, Trianon |

*Free access to ticketed areas; may need Passeport (€10 reduced) for garden events

### 4. Error Handling

If budget is insufficient for requested POIs:
- Returns clear error message
- Suggests affordable alternative
- Lists missing zones

## Configuration

See [pricing_config.py](pricing_config.py) for:
- `PRICING_TABLE` - Official pricing by season
- `is_free_access()` - Free access eligibility
- `get_ticket_price()` - Price calculation
- `find_affordable_ticket()` - Best ticket within budget
- `validate_budget_for_interests()` - Budget validation

## Pricing Source

Pricing based on official Versailles website:
https://www.chateauversailles.fr/preparer-ma-visite/billets-tarifs

**Note:** Prices updated as of 2025. Check official website for current rates.

## Live Pricing (Dynamic Fetch)

### Enable Live Pricing

Set environment variable to fetch current prices from the website:

```bash
export VERSAILLES_USE_LIVE_PRICING=true
```

When enabled, the system will:
1. Fetch current pricing from https://www.chateauversailles.fr/preparer-ma-visite/billets-tarifs
2. Use date-specific pricing based on visit date
3. Fall back to static pricing if fetch fails

### Manual Pricing Check

Check current pricing for a specific date:

```bash
python3.11 scripts/fetch_pricing.py --date 2025-12-09
```

Output:
```
📅 Fetching pricing for 09/12/2025
======================================================================
PRICING FETCHED FOR 09/12/2025
======================================================================

🎫 PASSEPORT (Basse saison)
   Plein tarif: €24.00
   Tarif réduit: €10.00

🏰 CHÂTEAU
   Plein tarif: €21.00
   Tarif réduit: €13.00

🌸 DOMAINE DE TRIANON
   Plein tarif: €12.00
   Tarif réduit: €8.00
```

### Production Deployment

For production with live pricing:

```yaml
# render.yaml
envVars:
  - key: VERSAILLES_USE_LIVE_PRICING
    value: "true"
```

**Note:** Live pricing adds ~1-2 seconds latency per request due to web scraping. Consider:
- Caching fetched prices (by date)
- Using static pricing for demos
- Enabling only in production

## Future Enhancements

Potential additions:
- [x] Auto-fetch current pricing from website ✅
- [ ] Cache fetched prices (daily cache)
- [ ] Handle garden event pricing (Grandes Eaux Musicales)
- [ ] Support audio guide pricing (€4-6)
- [ ] Group discount calculations
- [ ] Combined ticket offers (Passeport + lunch, etc.)
