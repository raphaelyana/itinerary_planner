"""Centralized configuration module that loads .env file."""

import os
from pathlib import Path

# Load .env file if it exists (development only)
try:
    from dotenv import load_dotenv

    # Look for .env file in project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip .env loading
    # Environment variables from system/Render will still work via os.getenv()
    pass


def get_neo4j_uri() -> str:
    """Get Neo4j URI from environment."""
    return os.getenv("NEO4J_URI", "neo4j://localhost:7687")


def get_neo4j_username() -> str:
    """Get Neo4j username from environment."""
    return os.getenv("NEO4J_USERNAME", "neo4j")


def get_neo4j_password() -> str:
    """Get Neo4j password from environment."""
    return os.getenv("NEO4J_PASSWORD", "neo4j")


def get_neo4j_database() -> str:
    """Get Neo4j database name from environment."""
    return os.getenv("NEO4J_DATABASE", "neo4j")
