# Interest Tags Vocabulary

## Overview

This document defines the complete vocabulary of interest tags used in the Versailles knowledge graph. Each POI can have multiple tags separated by semicolons (`;`).

## Complete Tag List (38 tags)

### **Core Interest Categories**

| Tag | Description | Example POIs |
|-----|-------------|--------------|
| `history` | Historical significance | Most palace rooms, monuments |
| `art` | General artistic value | Galleries, decorated rooms |
| `architecture` | Architectural interest | Buildings, structures |
| `garden` | Garden and landscape | All garden areas |
| `must_see` | Top attractions | Hall of Mirrors, Royal Apartments |

### **Art Subcategories**

| Tag | Description | Example POIs |
|-----|-------------|--------------|
| `art_baroque` | Baroque art style | Royal Chapel |
| `art_classique` | Classical art style | Hall of Mirrors |
| `art_militaire` | Military art | Battle galleries, Napoleon rooms |
| `art_portrait` | Portrait galleries | Gallery of Historic Figures |
| `art_religieux` | Religious art | Royal Chapel |
| `art_rocaille` | Rococo art style | Petits Appartements |

### **Architecture Subcategories**

| Tag | Description | Example POIs |
|-----|-------------|--------------|
| `architecture_classique` | Classical architecture | Main palace façade |

### **Historical Periods**

| Tag | Description | Example POIs |
|-----|-------------|--------------|
| `louis_xiv` | Louis XIV era (1643-1715) | Hall of Mirrors, Grand Apartments |
| `louis_xv` | Louis XV era (1715-1774) | Opera, Petits Appartements |
| `marie_antoinette` | Marie-Antoinette era | Petit Trianon, Hameau |
| `napoleon_i` | Napoleon I era (1804-1814) | Napoleon rooms, battle galleries |
| `monarchy_july` | July Monarchy (1830-1848) | Gallery of Battles, Crusades rooms |
| `revolution` | French Revolution period | Historical galleries |
| `pape_pie_vii` | Pope Pius VII related | Specific historical room |

### **Thematic Categories**

| Tag | Description | Example POIs |
|-----|-------------|--------------|
| `military` | Military history | Battle galleries |
| `politics` | Political significance | Council rooms |
| `mythology` | Mythological themes | Fountain of Apollo, bosquets |
| `music` | Musical heritage | Opera, concert halls |
| `lifestyle` | Royal lifestyle | Private apartments, Hameau |

### **Visitor Experience**

| Tag | Description | Example POIs |
|-----|-------------|--------------|
| `photo_spot` | Great for photos | Hall of Mirrors, gardens |
| `point_of_view` | Scenic viewpoint | Terraces, overlooks |
| `scenic` | Scenic beauty | Gardens, water features |
| `rain_safe` | Indoor/covered | Palace interiors, galleries |
| `walk` | Walking paths | Garden alleys, groves |

### **Natural Elements**

| Tag | Description | Example POIs |
|-----|-------------|--------------|
| `flowers` | Flower displays | Parterres, orangerie |
| `fountains` | Water fountains | Basin of Apollo, Neptune |
| `water` | Water features | Canals, basins, fountains |

### **Functional Tags**

| Tag | Description | Example POIs |
|-----|-------------|--------------|
| `entrance` | Entry points | Palace entrances, gate |
| `exit` | Exit points | Exit gates |
| `restroom` | Restroom facilities | Visitor facilities |
| `services` | Service areas | Information desks |
| `protection` | Weather protection | Covered areas |
| `transition` | Transition spaces | Corridors, vestibules |

## Usage Guidelines

### **Tag Formatting Rules**

✅ **Correct:**
```csv
interest_tags: history;art;rain_safe;napoleon_i
```

❌ **Incorrect:**
```csv
interest_tags: history;art;rain_safe:napoleon_i  # Wrong separator
interest_tags: history, art, must_see            # Wrong separator
interest_tags: HISTORY;ART                       # Wrong case
```

### **Combining Tags**

POIs should have multiple tags to enable flexible filtering:

**Example 1: Hall of Mirrors**
```csv
interest_tags: history;art;art_classique;louis_xiv;must_see;photo_spot;rain_safe
```

**Example 2: Basin of Apollo**
```csv
interest_tags: mythology;fountains;water;louis_xiv;photo_spot;scenic
```

**Example 3: Gallery of Battles**
```csv
interest_tags: history;art;art_militaire;military;monarchy_july;rain_safe
```

### **Tag Selection Best Practices**

1. **Be Specific AND General**: Include both broad tags (`art`) and specific ones (`art_militaire`)
2. **Historical Context**: Add era tags (`louis_xiv`, `napoleon_i`) for historical POIs
3. **Visitor Experience**: Include experience tags (`rain_safe`, `photo_spot`) for UX
4. **Natural Features**: Tag natural elements (`water`, `flowers`) for thematic filtering

## Tag Statistics

**Total unique tags:** 38

**Category breakdown:**
- Core interests: 5
- Art subcategories: 6
- Architecture subcategories: 1
- Historical periods: 7
- Thematic categories: 4
- Visitor experience: 6
- Natural elements: 3
- Functional tags: 6

## API Integration

### **Filtering by Interest Tags**

Example API request:
```json
{
  "constraints": {
    "interests": ["napoleon_i", "art_militaire"],
    "user_profile": "standard",
    "accessibility": "any"
  }
}
```

This will match POIs containing **ANY** of these tags:
- Napoleon rooms (tagged with `napoleon_i`)
- Battle galleries (tagged with `art_militaire`)

### **Tested Tags for Demo**

These tags have been verified to return results:
- ✅ `history` (most POIs)
- ✅ `art` (galleries, rooms)
- ✅ `garden` (outdoor areas)
- ✅ `must_see` (top attractions)
- ✅ `architecture` (buildings)
- ✅ `napoleon_i` (Napoleon-related)
- ✅ `louis_xiv` (Sun King era)
- ✅ `rain_safe` (indoor POIs)

## Validation

### **Current Data Status**

✅ All 160 POIs use valid tags from this vocabulary
✅ All tags use semicolon (`;`) separator
✅ No orphaned or undefined tags

### **Problematic Tags Found (Fixed)**

❌ **Before:** `rain_safe:napoleon_i` (colon separator)
✅ **After:** `rain_safe;napoleon_i` (semicolon separator)

## Future Enhancements

### **Potential New Tags**

Consider adding:
- `accessible_wheelchair` - Wheelchair accessible POIs
- `guided_tour` - POIs with guided tours
- `audio_guide` - Audio guide available
- `seasonal` - Seasonal attractions
- `special_events` - Event venues

### **Tag Hierarchies**

Consider implementing parent-child relationships:
```
art (parent)
├── art_baroque
├── art_classique
├── art_militaire
├── art_portrait
└── art_religieux
```

This would allow querying:
- `art` → Returns ALL art POIs
- `art_militaire` → Returns ONLY military art POIs

---

**Last Updated:** 2025-11-24
**Status:** ✅ Complete and validated
**POI Count:** 160
**Tag Count:** 38
