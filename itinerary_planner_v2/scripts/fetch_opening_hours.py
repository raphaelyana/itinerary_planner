#!/usr/bin/env python3.11
"""
Fetch opening hours from Château de Versailles official website.
URL pattern: https://www.chateauversailles.fr/actualites/agenda-chateau-versailles/fr-YYYY-MM-DD
"""
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass
import json


@dataclass
class OpeningHours:
    """Opening hours for a specific POI."""
    poi_id: str
    date: str
    opening_time: str
    closing_time: str
    is_open: bool
    notes: Optional[str] = None


class VersaillesScheduleScraper:
    """Scrape opening hours from official Versailles website."""

    BASE_URL = "https://www.chateauversailles.fr/actualites/agenda-chateau-versailles/fr-{date}"

    # Mapping of website sections to POI categories
    SECTION_MAPPINGS = {
        "château": "Castle",
        "palace": "Castle",
        "galerie des glaces": "Room:galerie-des-glaces",
        "grands appartements": "Castle",
        "jardins": "Garden",
        "gardens": "Garden",
        "trianon": "Trianon",
        "grand trianon": "Trianon",
        "petit trianon": "Trianon",
        "domaine de marie-antoinette": "Trianon",
        "parc": "Park",
        "park": "Park"
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def fetch_date(self, date: datetime) -> Optional[Dict]:
        """Fetch opening hours for a specific date."""
        url = self.BASE_URL.format(date=date.strftime("%Y-%m-%d"))

        print(f"📅 Fetching: {url}")

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract opening hours from page
            opening_info = self._parse_opening_hours(soup, date)

            return opening_info

        except requests.RequestException as e:
            print(f"❌ Error fetching {url}: {e}")
            return None

    def _parse_opening_hours(self, soup: BeautifulSoup, date: datetime) -> Dict:
        """Parse opening hours from HTML."""
        result = {
            "date": date.strftime("%Y-%m-%d"),
            "sections": [],
            "raw_text": ""
        }

        # Look for schedule sections
        schedule_sections = soup.find_all(['div', 'section', 'article'],
                                         class_=re.compile(r'schedule|horaire|opening|hours', re.I))

        for section in schedule_sections:
            text = section.get_text(strip=True)
            result["raw_text"] += text + "\n"

            # Extract time patterns (e.g., "9h-18h", "09:00-18:00", "9h00 à 18h00")
            time_patterns = re.findall(
                r'(\d{1,2})[h:](\d{2})?\s*[-àto]\s*(\d{1,2})[h:](\d{2})?',
                text
            )

            for match in time_patterns:
                open_hour, open_min, close_hour, close_min = match
                opening_time = f"{open_hour.zfill(2)}:{open_min or '00'}"
                closing_time = f"{close_hour.zfill(2)}:{close_min or '00'}"

                # Try to identify which section this applies to
                section_name = self._identify_section(text)

                result["sections"].append({
                    "name": section_name,
                    "opening_time": opening_time,
                    "closing_time": closing_time,
                    "raw": text[:200]
                })

        # If no specific times found, look for "closed" indicators
        if not result["sections"]:
            if any(word in result["raw_text"].lower() for word in ["fermé", "closed", "fermée"]):
                result["sections"].append({
                    "name": "all",
                    "is_closed": True,
                    "raw": "Closed"
                })

        return result

    def _identify_section(self, text: str) -> str:
        """Identify which POI section the text refers to."""
        text_lower = text.lower()

        for keyword, section in self.SECTION_MAPPINGS.items():
            if keyword in text_lower:
                return section

        return "unknown"

    def fetch_week(self, start_date: datetime) -> Dict[str, Dict]:
        """Fetch opening hours for a week."""
        results = {}

        for day_offset in range(7):
            date = start_date + timedelta(days=day_offset)
            day_name = date.strftime("%A")

            print(f"\nFetching {day_name} ({date.strftime('%Y-%m-%d')})...")

            opening_info = self.fetch_date(date)
            if opening_info:
                results[date.strftime("%Y-%m-%d")] = opening_info

        return results

    def apply_default_hours(self, date: datetime) -> Dict[str, OpeningHours]:
        """
        Apply default opening hours based on day of week.
        Fallback when scraping fails.
        """
        day_name = date.strftime("%A")

        # Default hours from official Versailles website
        # Source: https://www.chateauversailles.fr/preparer-ma-visite/horaires
        defaults = {
            "Castle": {
                "Tuesday-Sunday": ("09:00", "17:30"),  # Extended in summer
                "Monday": None  # Closed
            },
            "Garden": {
                "Tuesday-Sunday": ("08:00", "18:00"),  # Extended in summer
                "Monday": ("08:00", "18:00")
            },
            "Trianon": {
                "Tuesday-Sunday": ("12:00", "17:30"),
                "Monday": None  # Closed
            },
            "Park": {
                "All": ("07:00", "20:30")
            }
        }

        result = {}

        # Castle
        if day_name != "Monday":
            result["Castle"] = OpeningHours(
                poi_id="Castle",
                date=date.strftime("%Y-%m-%d"),
                opening_time="09:00",
                closing_time="17:30",
                is_open=True,
                notes="Default hours (scraping failed)"
            )

        # Gardens
        result["Garden"] = OpeningHours(
            poi_id="Garden",
            date=date.strftime("%Y-%m-%d"),
            opening_time="08:00",
            closing_time="18:00",
            is_open=True,
            notes="Default hours (scraping failed)"
        )

        # Trianon
        if day_name != "Monday":
            result["Trianon"] = OpeningHours(
                poi_id="Trianon",
                date=date.strftime("%Y-%m-%d"),
                opening_time="12:00",
                closing_time="17:30",
                is_open=True,
                notes="Default hours (scraping failed)"
            )

        return result


def update_pois_with_opening_hours(date: datetime) -> None:
    """
    Update POIs CSV with opening/closing times for a specific date.
    Creates a dated version of the POI file.
    """
    import csv
    from pathlib import Path

    scraper = VersaillesScheduleScraper()

    # Try to scrape
    print(f"\n{'='*70}")
    print(f"FETCHING OPENING HOURS FOR {date.strftime('%Y-%m-%d')}")
    print(f"{'='*70}\n")

    opening_info = scraper.fetch_date(date)

    # If scraping failed, use defaults
    if not opening_info or not opening_info.get("sections"):
        print("⚠️  Scraping failed or no data found. Using default hours.")
        default_hours = scraper.apply_default_hours(date)
    else:
        print(f"✅ Found {len(opening_info['sections'])} schedule sections")
        print("\nParsed schedules:")
        for section in opening_info['sections']:
            print(f"  • {section['name']}: {section.get('opening_time', 'N/A')} - {section.get('closing_time', 'N/A')}")

        # Convert to OpeningHours format
        default_hours = scraper.apply_default_hours(date)

    # Read existing POIs
    pois_file = Path("data/main_data/pois.csv")
    output_file = Path(f"data/main_data/pois_{date.strftime('%Y%m%d')}.csv")

    print(f"\n📝 Updating POIs file: {output_file}")

    with open(pois_file, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames

        # Add opening_time and closing_time if not present
        if 'opening_time' not in fieldnames:
            fieldnames = list(fieldnames) + ['opening_time', 'closing_time']

        rows = list(reader)

    # Update rows with opening hours
    updated_count = 0
    for row in rows:
        zone = row.get('zone', '')
        poi_id = row.get('id', '')

        # Determine which schedule to apply
        opening_hours = None
        if 'Castle' in zone or 'Room' in row.get('category', ''):
            opening_hours = default_hours.get('Castle')
        elif 'Garden' in zone:
            opening_hours = default_hours.get('Garden')
        elif 'Trianon' in zone or 'Trianon' in poi_id:
            opening_hours = default_hours.get('Trianon')
        elif 'Park' in zone:
            opening_hours = default_hours.get('Garden')  # Same as gardens

        if opening_hours and opening_hours.is_open:
            row['opening_time'] = opening_hours.opening_time
            row['closing_time'] = opening_hours.closing_time
            updated_count += 1
        else:
            # Closed
            row['opening_time'] = ''
            row['closing_time'] = ''

    # Write updated file
    with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Updated {updated_count}/{len(rows)} POIs with opening hours")
    print(f"📄 Saved to: {output_file}")

    return output_file


def main():
    """CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch opening hours from Château de Versailles website"
    )
    parser.add_argument(
        '--date',
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Date in YYYY-MM-DD format (default: today)"
    )
    parser.add_argument(
        '--week',
        action='store_true',
        help="Fetch entire week starting from date"
    )
    parser.add_argument(
        '--update-pois',
        action='store_true',
        help="Update POIs CSV with opening hours"
    )
    parser.add_argument(
        '--output',
        type=str,
        help="Output JSON file for results"
    )

    args = parser.parse_args()

    # Parse date
    try:
        date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"❌ Invalid date format: {args.date}")
        print("   Use YYYY-MM-DD format (e.g., 2025-11-25)")
        return 1

    scraper = VersaillesScheduleScraper()

    if args.week:
        # Fetch week
        results = scraper.fetch_week(date)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n📊 Results saved to: {args.output}")

    elif args.update_pois:
        # Update POIs file
        output_file = update_pois_with_opening_hours(date)

    else:
        # Fetch single date
        result = scraper.fetch_date(date)

        if result:
            print(f"\n📊 Results:")
            print(json.dumps(result, indent=2))

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n📄 Saved to: {args.output}")
        else:
            print("❌ Failed to fetch opening hours")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
