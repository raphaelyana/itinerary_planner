from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, Optional, Tuple, Literal, Protocol

import requests
from bs4 import BeautifulSoup

from neo4j import GraphDatabase, Session

Season = Literal["High", "Low"]
Weekday = Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TimeWindow = Optional[Tuple[str, str]]


@dataclass(frozen=True)
class RulesetDailyHours:
    """Represents the opening window to apply to a ruleset for a specific day."""

    ruleset_id: str
    opening_time: Optional[str]
    closing_time: Optional[str]
    season: Season
    target_date: date


class OpeningHoursSource(Protocol):
    """Protocol for pluggable data sources (scraper, API, mocked data)."""

    def fetch_daily_hours(
        self, target_date: date, *, season: Optional[Season] = None
    ) -> Iterable[RulesetDailyHours]:
        ...


class StaticScheduleSource:
    """Simple in-memory schedule used until a live scraper is plugged in."""

    def __init__(
        self,
        schedule: Dict[str, Dict[Season, Dict[Weekday, TimeWindow]]],
    ) -> None:
        self._schedule = schedule

    def fetch_daily_hours(
        self, target_date: date, *, season: Optional[Season] = None
    ) -> Iterable[RulesetDailyHours]:
        resolved_season = season or determine_season(target_date)
        weekday = cast_weekday(target_date.strftime("%a"))

        for ruleset_id, season_windows in self._schedule.items():
            current_season_windows = season_windows.get(resolved_season, {})
            time_window = current_season_windows.get(weekday)
            opening, closing = time_window if time_window else (None, None)
            yield RulesetDailyHours(
                ruleset_id=ruleset_id,
                opening_time=opening,
                closing_time=closing,
                season=resolved_season,
                target_date=target_date,
            )


class VersaillesWebSource:
    """Scrapes the official Versailles opening-hours page for dynamic updates."""

    BASE_URL = "https://www.chateauversailles.fr"
    OPENING_PATH = "/planifier/infos-pratiques/horaires"

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self._session = session or requests.Session()

    def fetch_daily_hours(
        self, target_date: date, *, season: Optional[Season] = None
    ) -> Iterable[RulesetDailyHours]:
        html = self._get_opening_page(target_date)
        soup = BeautifulSoup(html, "html.parser")

        parsed = self._parse_sections(soup)
        resolved_season = season or determine_season(target_date)
        weekday = cast_weekday(target_date.strftime("%a"))

        for ruleset_id, hours_by_season in parsed.items():
            season_hours = hours_by_season.get(resolved_season, {})
            opening, closing = season_hours.get(weekday, (None, None))
            yield RulesetDailyHours(
                ruleset_id=ruleset_id,
                opening_time=opening,
                closing_time=closing,
                season=resolved_season,
                target_date=target_date,
            )

    def _get_opening_page(self, target_date: date) -> str:
        # Agenda pages follow pattern /actualites/agenda-chateau-versailles/fr-YYYY-MM-DD
        agenda_path = f"/actualites/agenda-chateau-versailles/fr-{target_date.strftime('%Y-%m-%d')}"
        url = f"{self.BASE_URL}{agenda_path}"
        response = self._session.get(url, timeout=10)
        if response.status_code == 404:
            # fallback to general schedule page if agenda entry missing
            fallback_url = f"{self.BASE_URL}{self.OPENING_PATH}"
            fallback_resp = self._session.get(fallback_url, timeout=10)
            fallback_resp.raise_for_status()
            return fallback_resp.text
        response.raise_for_status()
        return response.text

    @staticmethod
    def _parse_sections(soup: BeautifulSoup) -> Dict[str, Dict[Season, Dict[Weekday, TimeWindow]]]:
        schedule: Dict[str, Dict[Season, Dict[Weekday, TimeWindow]]] = {
            "PALACE_DEFAULT": {"High": {}, "Low": {}},
            "TRIANON_DEFAULT": {"High": {}, "Low": {}},
            "GARDENS_DEFAULT": {"High": {}, "Low": {}},
            "PARK_DEFAULT": {"High": {}, "Low": {}},
        }

        mapping = {
            "palace": "PALACE_DEFAULT",
            "trianon": "TRIANON_DEFAULT",
            "gardens": "GARDENS_DEFAULT",
            "park": "PARK_DEFAULT",
        }

        sections = soup.select("section[id^='horaires-']")
        for section in sections:
            section_id = section.get("id", "")
            key = next((v for k, v in mapping.items() if k in section_id), None)
            if not key:
                continue

            season_label = "High" if "haute" in section_id else "Low"
            table = section.find("table")
            if not table:
                continue

            for row in table.select("tbody tr"):
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue
                day_text = cells[0].get_text(strip=True)
                hours_text = cells[1].get_text(strip=True)
                weekday = _normalize_weekday(day_text)
                opening, closing = _parse_hours(hours_text)
                schedule[key].setdefault(season_label, {})[weekday] = (opening, closing)

        return schedule


def cast_weekday(value: str) -> Weekday:
    days: Tuple[Weekday, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
    for day in days:
        if value == day:
            return day
    raise ValueError(f"Unsupported weekday abbreviation: {value!r}")


def _normalize_weekday(label: str) -> Weekday:
    label_lower = label.lower()
    mapping = {
        "lundi": "Mon",
        "mardi": "Tue",
        "mercredi": "Wed",
        "jeudi": "Thu",
        "vendredi": "Fri",
        "samedi": "Sat",
        "dimanche": "Sun",
    }
    if label_lower in mapping:
        return cast_weekday(mapping[label_lower])
    normalized = label[:3].capitalize()
    return cast_weekday(normalized)


def _parse_hours(text: str) -> TimeWindow:
    if not text or "fermé" in text.lower():
        return None
    parts = text.replace("h", ":").split("-")
    if len(parts) != 2:
        return None
    start = parts[0].strip()
    end = parts[1].strip()
    return start, end


def determine_season(target_date: date) -> Season:
    """Infer high/low season based on historical operating calendar."""
    high_start = date(target_date.year, 4, 1)
    high_end = date(target_date.year, 10, 31)
    return "High" if high_start <= target_date <= high_end else "Low"


DEFAULT_SCHEDULE: Dict[str, Dict[Season, Dict[Weekday, TimeWindow]]] = {
    "PALACE_DEFAULT": {
        "High": {
            "Mon": None,
            "Tue": ("09:00", "18:30"),
            "Wed": ("09:00", "18:30"),
            "Thu": ("09:00", "18:30"),
            "Fri": ("09:00", "18:30"),
            "Sat": ("09:00", "18:30"),
            "Sun": ("09:00", "18:30"),
        },
        "Low": {
            "Mon": None,
            "Tue": ("09:00", "17:30"),
            "Wed": ("09:00", "17:30"),
            "Thu": ("09:00", "17:30"),
            "Fri": ("09:00", "17:30"),
            "Sat": ("09:00", "17:30"),
            "Sun": ("09:00", "17:30"),
        },
    },
    "GARDENS_DEFAULT": {
        "High": {
            "Mon": ("07:00", "20:30"),
            "Tue": ("07:00", "20:30"),
            "Wed": ("07:00", "20:30"),
            "Thu": ("07:00", "20:30"),
            "Fri": ("07:00", "20:30"),
            "Sat": ("07:00", "20:30"),
            "Sun": ("07:00", "20:30"),
        },
        "Low": {
            "Mon": ("08:00", "18:00"),
            "Tue": ("08:00", "18:00"),
            "Wed": ("08:00", "18:00"),
            "Thu": ("08:00", "18:00"),
            "Fri": ("08:00", "18:00"),
            "Sat": ("08:00", "18:00"),
            "Sun": ("08:00", "18:00"),
        },
    },
    "TRIANON_DEFAULT": {
        "High": {
            "Mon": None,
            "Tue": ("12:00", "18:30"),
            "Wed": ("12:00", "18:30"),
            "Thu": ("12:00", "18:30"),
            "Fri": ("12:00", "18:30"),
            "Sat": ("12:00", "18:30"),
            "Sun": ("12:00", "18:30"),
        },
        "Low": {
            "Mon": None,
            "Tue": ("12:00", "17:30"),
            "Wed": ("12:00", "17:30"),
            "Thu": ("12:00", "17:30"),
            "Fri": ("12:00", "17:30"),
            "Sat": ("12:00", "17:30"),
            "Sun": ("12:00", "17:30"),
        },
    },
    "PARK_DEFAULT": {
        "High": {
            "Mon": ("07:00", "20:30"),
            "Tue": ("07:00", "20:30"),
            "Wed": ("07:00", "20:30"),
            "Thu": ("07:00", "20:30"),
            "Fri": ("07:00", "20:30"),
            "Sat": ("07:00", "20:30"),
            "Sun": ("07:00", "20:30"),
        },
        "Low": {
            "Mon": ("08:00", "18:00"),
            "Tue": ("08:00", "18:00"),
            "Wed": ("08:00", "18:00"),
            "Thu": ("08:00", "18:00"),
            "Fri": ("08:00", "18:00"),
            "Sat": ("08:00", "18:00"),
            "Sun": ("08:00", "18:00"),
        },
    },
}


def apply_update(session: Session, update: RulesetDailyHours) -> int:
    """Apply a single ruleset update; returns number of affected POIs."""
    query = """
    MATCH (poi:POI {opening_ruleset_id: $ruleset_id})
    SET poi.opening_time = $opening_time,
        poi.closing_time = $closing_time,
        poi.is_open_today = $is_open_today,
        poi.current_season = $season,
        poi.opening_time_last_updated = date($date)
    RETURN count(poi) AS affected
    """
    params = {
        "ruleset_id": update.ruleset_id,
        "opening_time": update.opening_time,
        "closing_time": update.closing_time,
        "is_open_today": update.opening_time is not None,
        "season": update.season,
        "date": update.target_date.isoformat(),
    }
    result = session.run(query, params)
    record = result.single()
    return record["affected"] if record else 0


def run_daily_update(
    target_date: Optional[date] = None,
    *,
    season: Optional[Season] = None,
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    database: Optional[str] = None,
    source: Optional[OpeningHoursSource] = None,
) -> int:
    """
    Update today's opening/closing times on POIs sharing the same ruleset.

    Parameters
    ----------
    target_date:
        Date to compute the schedule for. Defaults to ``date.today()``.
    season:
        Override for the peak season label; if omitted it is inferred from the date.
    uri, user, password:
        Neo4j connection details. Defaults fall back to the environment variables
        ``NEO4J_URI``, ``NEO4J_USERNAME``, ``NEO4J_PASSWORD`` or to Neo4j defaults.
    source:
        Optional data source implementation. If omitted, uses the static fallback.

    Returns
    -------
    int
        Total number of POI nodes touched by the update.
    """

    target_date = target_date or date.today()
    effective_source = source or VersaillesWebSource()
    updates = list(effective_source.fetch_daily_hours(target_date, season=season))

    if not updates:
        fallback = StaticScheduleSource(DEFAULT_SCHEDULE)
        updates = list(fallback.fetch_daily_hours(target_date, season=season))
        print("[updater] Falling back to static schedule; live scrape returned no data.")

    uri = uri or os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    user = user or os.getenv("NEO4J_USERNAME", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "neo4j")
    database = database or os.getenv("NEO4J_DATABASE", "neo4j")

    total = 0
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session(database=database) as session:
            for update in updates:
                total += apply_update(session, update)
    finally:
        driver.close()

    print(
        f"Updated opening hours for {total} POIs "
        f"({updates[0].season if updates else 'n/a'} season, date {target_date.isoformat()})."
    )
    return total


if __name__ == "__main__":
    run_daily_update()
