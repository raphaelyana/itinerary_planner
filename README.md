# Versailles Itinerary Planner V2

Intelligent itinerary planning system for Château de Versailles using TSP optimization and Neo4j knowledge graph.

## Architecture

- **Algorithm**: TSP optimization (3 solvers: permutation, greedy, Held-Karp)
- **Database**: Neo4j AuraDB (160 POIs, 324 connections)
- **API**: FastAPI (async, type-safe)
- **Deployment**: Render (free tier)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export NEO4J_URI=neo4j+s://4100001d.databases.neo4j.io
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=your_password
export NEO4J_DATABASE=neo4j

# Ingest data
python3.11 scripts/ingest.py

# Run API
uvicorn scripts.api:app --reload

# Run benchmarks
./run_benchmarks.sh offline
```

## Network Structure

### Start Points (Entrances)
- `versailles:Castle:cour-dhonneur` - Main palace entrance
- `versailles:Garden:acces-jardins-cour-des-princes` - Gardens entrance
- `versailles:Garden:entree-parc-bassin-apollon` - Park entrance (Basin of Apollo)
- `versailles:Garden:grille-de-neptune` - Neptune gate
- `versailles:Trianon:entree-grand-trianon` - Grand Trianon entrance
- `versailles:Trianon:entree-petit-trianon` - Petit Trianon entrance

### Finish Points (Exits)
- `versailles:Castle:cour-dhonneur-sortie` - Palace exit
- `versailles:Garden:entree-parc-bassin-apollon` - Basin of Apollo exit
- `versailles:Garden:grille-de-neptune` - Neptune gate exit
- `versailles:Trianon:entree-grand-trianon` - Grand Trianon exit
- `versailles:Trianon:entree-petit-trianon` - Petit Trianon exit

## Interest Tags (38 total)

See data for complete list. Most used:
- `history` (87 POIs)
- `art` (63 POIs)
- `scenic` (58 POIs)
- `rain_safe` (54 POIs)
- `architecture` (37 POIs)

## API Endpoints

### Plan Itinerary
```bash
POST /itinerary
{
  "start_time": "2025-11-26T09:00:00",
  "total_duration_minutes": 240,
  "constraints": {
    "interests": ["history", "must_see"],
    "user_profile": "standard",
    "accessibility": "any"
  }
}
```

## Performance

- 90-95% optimal routes
- 2000x faster than brute force
- Sub-second response times

## Deployment

See `render.yaml` for production deployment configuration.

Free tier handles:
- 1000s daily users
- 160 POIs (0.3% of 50k limit)
- 324 connections (0.16% of 200k limit)
