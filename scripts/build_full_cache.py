from __future__ import annotations

import csv
import os
from itertools import combinations
from pathlib import Path
from typing import Iterable, List

from scripts.path_cache import PathCache
from scripts.planner_utils import get_shortest_path

PROFILES: List[str] = ["standard", "family", "elder"]
ACCESSIBILITY_MODES: List[str] = ["any", "step_free", "stroller"]

profiles_env = os.getenv("CACHE_PROFILES")
if profiles_env:
    PROFILES = [p.strip() for p in profiles_env.split(",") if p.strip()] or PROFILES

access_env = os.getenv("CACHE_ACCESSIBILITY")
if access_env:
    ACCESSIBILITY_MODES = [a.strip() for a in access_env.split(",") if a.strip()] or ACCESSIBILITY_MODES

CACHE_FILE = os.getenv("PLANNER_DISTANCE_CACHE", "cache/full_path_cache.json")
POIS_CSV = Path("data/main_data/pois.csv")


def _load_poi_ids() -> Iterable[str]:
    with POIS_CSV.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            poi_id = (row.get("id") or "").strip()
            if poi_id:
                yield poi_id


def precompute_all_paths() -> None:
    poi_ids = list(dict.fromkeys(_load_poi_ids()))
    cache = PathCache(CACHE_FILE, auto_save=False)

    total_combinations = len(PROFILES) * len(ACCESSIBILITY_MODES)
    pair_count = len(poi_ids) * (len(poi_ids) - 1) // 2

    print(f"[path-cache] Precomputing {pair_count} pairs across {total_combinations} contexts…")

    for profile in PROFILES:
        for accessibility in ACCESSIBILITY_MODES:
            print(f"  Context profile={profile}, accessibility={accessibility}")
            for start_id, end_id in combinations(poi_ids, 2):
                if cache.contains(profile, accessibility, start_id, end_id):
                    continue
                try:
                    result = get_shortest_path(
                        start_id,
                        end_id,
                        user_profile=profile,
                        accessibility=accessibility,
                    )
                except ValueError:
                    continue

                cache.store(profile, accessibility, start_id, end_id, result, persist=False)
                cache.store(profile, accessibility, end_id, start_id, result, persist=False)
            cache.save()

    cache.save()
    print(f"[path-cache] Cache saved to {CACHE_FILE}")


if __name__ == "__main__":
    precompute_all_paths()
