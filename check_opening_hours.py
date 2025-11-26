#!/usr/bin/env python3
"""Quick script to check if opening hours are set for POIs."""
import os
from neo4j import GraphDatabase
from scripts.planner_utils import normalize_neo4j_uri

def check_hours():
    uri = normalize_neo4j_uri(os.getenv("NEO4J_URI"))
    driver = GraphDatabase.driver(
        uri,
        auth=(os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "neo4j"))
    )
    
    with driver.session() as session:
        # Check chapelle-royale specifically
        result = session.run("""
            MATCH (poi:POI {id: 'versailles:Room:chapelle-royale'})
            RETURN poi.id AS id, 
                   poi.name AS name,
                   poi.opening_ruleset_id AS ruleset,
                   poi.opening_time AS opening,
                   poi.closing_time AS closing,
                   poi.is_open_today AS is_open
        """)
        
        record = result.single()
        if record:
            print(f"✓ Found: {record['name']}")
            print(f"  Ruleset: {record['ruleset']}")
            print(f"  Opening: {record['opening']}")
            print(f"  Closing: {record['closing']}")
            print(f"  Is open: {record['is_open']}")
            
            if not record['opening'] or not record['closing']:
                print("\n⚠️  Opening hours not set! Run the updater script.")
                return False
            else:
                print("\n✓ Opening hours are properly set!")
                return True
        else:
            print("✗ POI not found in database")
            return False
    
    driver.close()

if __name__ == "__main__":
    check_hours()
