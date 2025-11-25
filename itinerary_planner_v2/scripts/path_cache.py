from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from scripts.planner_utils import PathSegment, ShortestPathResult


class PathCache:
    """
    Persistent, context-aware cache storing shortest-path results keyed by
    (profile, accessibility, start_id, end_id).
    """

    def __init__(self, filepath: str, *, auto_save: bool = False) -> None:
        self.filepath = Path(filepath)
        self.auto_save = auto_save
        self._cache: Dict[str, Dict[str, object]] = {}
        self._load()

    @staticmethod
    def _make_key(profile: str, accessibility: str, start_id: str, end_id: str) -> str:
        return f"{profile}|{accessibility}|{start_id}|{end_id}"

    @staticmethod
    def _serialize(result: ShortestPathResult) -> Dict[str, object]:
        return {
            "total_minutes": result.total_minutes,
            "node_ids": result.node_ids,
            "segments": [
                {
                    "from_id": segment.from_id,
                    "to_id": segment.to_id,
                    "distance_min": segment.distance_min,
                    "is_step_free": segment.is_step_free,
                    "stroller_friendly": segment.stroller_friendly,
                    "path_type": segment.path_type,
                    "notes": segment.notes,
                }
                for segment in result.segments
            ],
        }

    @staticmethod
    def _deserialize(payload: Dict[str, object]) -> Optional[ShortestPathResult]:
        try:
            total = float(payload["total_minutes"])
            node_ids = list(payload.get("node_ids", []))
            segments_data = payload.get("segments", []) or []
        except (KeyError, TypeError, ValueError):
            return None

        segments: list[PathSegment] = []
        try:
            for segment in segments_data:
                segments.append(
                    PathSegment(
                        from_id=segment["from_id"],
                        to_id=segment["to_id"],
                        distance_min=float(segment.get("distance_min", 0.0)),
                        is_step_free=bool(segment.get("is_step_free", False)),
                        stroller_friendly=bool(segment.get("stroller_friendly", False)),
                        path_type=segment.get("path_type"),
                        notes=segment.get("notes"),
                    )
                )
        except (KeyError, TypeError, ValueError):
            return None

        return ShortestPathResult(node_ids=node_ids, total_minutes=total, segments=segments)

    def _load(self) -> None:
        if not self.filepath.exists():
            self._cache = {}
            return
        with self.filepath.open() as handle:
            self._cache = json.load(handle)

    def save(self) -> None:
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        with self.filepath.open("w") as handle:
            json.dump(self._cache, handle)

    def contains(self, profile: str, accessibility: str, start_id: str, end_id: str) -> bool:
        key = self._make_key(profile, accessibility, start_id, end_id)
        return key in self._cache

    def get(
        self,
        profile: str,
        accessibility: str,
        start_id: str,
        end_id: str,
    ) -> Optional[ShortestPathResult]:
        key = self._make_key(profile, accessibility, start_id, end_id)
        payload = self._cache.get(key)
        if payload is None:
            return None
        return self._deserialize(payload)

    def store(
        self,
        profile: str,
        accessibility: str,
        start_id: str,
        end_id: str,
        result: ShortestPathResult,
        *,
        persist: bool = False,
    ) -> None:
        key = self._make_key(profile, accessibility, start_id, end_id)
        self._cache[key] = self._serialize(result)
        if persist or self.auto_save:
            self.save()
