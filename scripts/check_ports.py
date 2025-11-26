#!/usr/bin/env python3
"""Quick diagnostic script to check Neo4j connectivity and ports."""

import os
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file if it exists (development mode)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars

from scripts.planner_utils import normalize_neo4j_uri

def check_port(host: str, port: int, timeout: float = 5.0) -> bool:
    """Test if a port is reachable."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except socket.gaierror:
        return False
    except Exception as e:
        print(f"Error testing {host}:{port} - {e}")
        return False

def main():
    print("="*70)
    print("NEO4J PORT CONNECTIVITY CHECK")
    print("="*70)
    print()

    # Get URI from environment
    raw_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    normalized_uri = normalize_neo4j_uri(raw_uri)

    print(f"Raw URI from env: {raw_uri}")
    print(f"Normalized URI: {normalized_uri}")
    print()

    # Parse hostname and port
    if "://" in normalized_uri:
        scheme, rest = normalized_uri.split("://", 1)
        if ":" in rest:
            host, port_str = rest.rsplit(":", 1)
            try:
                port = int(port_str)
            except ValueError:
                host = rest
                port = 7687
        else:
            host = rest
            port = 7687
    else:
        host = "localhost"
        port = 7687

    print(f"Extracted hostname: {host}")
    print(f"Extracted port: {port}")
    print()

    # Test common Neo4j ports
    ports_to_test = [
        (port, "Bolt (from URI)"),
        (7687, "Bolt (default)"),
        (7474, "HTTP"),
    ]

    for test_port, description in ports_to_test:
        print(f"Testing {host}:{test_port} ({description})... ", end="", flush=True)
        if check_port(host, test_port):
            print("✅ OPEN")
        else:
            print("❌ CLOSED/UNREACHABLE")

    print()
    print("="*70)
    print("Environment Variables:")
    print("="*70)
    print(f"NEO4J_URI: {os.getenv('NEO4J_URI', 'NOT SET')}")
    print(f"NEO4J_USERNAME: {os.getenv('NEO4J_USERNAME', 'NOT SET')}")
    print(f"NEO4J_PASSWORD: {'***' if os.getenv('NEO4J_PASSWORD') else 'NOT SET'}")
    print(f"NEO4J_DATABASE: {os.getenv('NEO4J_DATABASE', 'NOT SET')}")
    print("="*70)

    # Now try actual Neo4j connection
    print()
    print("Testing actual Neo4j driver connection...")
    print()

    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            normalized_uri,
            auth=(
                os.getenv("NEO4J_USERNAME", "neo4j"),
                os.getenv("NEO4J_PASSWORD", "neo4j")
            ),
            connection_timeout=10.0,
        )

        with driver.session() as session:
            result = session.run("RETURN 1 AS test")
            record = result.single()
            if record and record["test"] == 1:
                print("✅ Neo4j driver connection SUCCESSFUL")
            else:
                print("❌ Query returned unexpected result")

        driver.close()

    except Exception as e:
        print(f"❌ Neo4j driver connection FAILED")
        print(f"   Error: {type(e).__name__}: {e}")

    print()
    print("="*70)

if __name__ == "__main__":
    main()
