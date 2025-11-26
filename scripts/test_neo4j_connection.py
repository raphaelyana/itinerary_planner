"""Test Neo4j connection to diagnose Render deployment issues."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j import GraphDatabase
from scripts.planner_utils import normalize_neo4j_uri

def test_connection():
    """Test Neo4j connection with detailed diagnostics."""

    # Get connection parameters from environment
    uri = normalize_neo4j_uri(os.getenv("NEO4J_URI", "neo4j://localhost:7687"))
    user = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "neo4j")
    database = os.getenv("NEO4J_DATABASE", "neo4j")

    print("="*70)
    print("NEO4J CONNECTION TEST")
    print("="*70)
    print(f"URI: {uri}")
    print(f"Username: {user}")
    print(f"Database: {database}")
    print(f"Password: {'*' * len(password) if password else 'NOT SET'}")
    print("="*70)
    print()

    try:
        print("Attempting to connect...")
        driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=10,
            connection_timeout=30.0,
        )

        print("✓ Driver created successfully")
        print()

        # Test query
        print("Testing database query...")
        with driver.session(database=database) as session:
            result = session.run("RETURN 1 AS test")
            record = result.single()
            if record and record["test"] == 1:
                print("✓ Query test successful")
            else:
                print("✗ Query returned unexpected result")

        print()

        # Count POIs
        print("Counting POIs in database...")
        with driver.session(database=database) as session:
            result = session.run("MATCH (p:POI) RETURN count(p) AS count")
            record = result.single()
            if record:
                count = record["count"]
                print(f"✓ Found {count} POIs in database")
                if count == 0:
                    print("⚠️  WARNING: No POIs found - database may not be ingested")
            else:
                print("✗ Could not count POIs")

        print()

        # Count connections
        print("Counting connections in database...")
        with driver.session(database=database) as session:
            result = session.run("MATCH ()-[r:CONNECTS_TO]->() RETURN count(r) AS count")
            record = result.single()
            if record:
                count = record["count"]
                print(f"✓ Found {count} connections in database")
                if count == 0:
                    print("⚠️  WARNING: No connections found - database may not be ingested")
            else:
                print("✗ Could not count connections")

        print()
        print("="*70)
        print("✅ ALL TESTS PASSED - Neo4j connection is working")
        print("="*70)

        driver.close()
        return True

    except Exception as e:
        print()
        print("="*70)
        print("❌ CONNECTION FAILED")
        print("="*70)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print()
        print("Possible causes:")
        print("1. Neo4j service is not running")
        print("2. Incorrect URI/credentials")
        print("3. Network connectivity issues")
        print("4. Database not initialized")
        print("="*70)
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
