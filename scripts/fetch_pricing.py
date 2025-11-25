#!/usr/bin/env python3.11
"""
Fetch current ticket pricing from Château de Versailles website.
URL: https://www.chateauversailles.fr/preparer-ma-visite/billets-tarifs?date_filter[value][date]=DD/MM/YYYY
"""

import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class TicketPricing:
    """Current ticket pricing from website."""
    passeport_full: Optional[float] = None
    passeport_reduced: Optional[float] = None
    chateau_full: Optional[float] = None
    chateau_reduced: Optional[float] = None
    trianon_full: Optional[float] = None
    trianon_reduced: Optional[float] = None
    fetch_date: Optional[str] = None
    season: Optional[str] = None  # "Basse saison" or "Haute saison"


class VersaillesPricingScraper:
    """Scrape current ticket pricing from official website."""

    BASE_URL = "https://www.chateauversailles.fr/preparer-ma-visite/billets-tarifs"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def fetch_pricing(self, date: datetime) -> Optional[TicketPricing]:
        """
        Fetch current pricing for a specific date.

        Args:
            date: Visit date

        Returns:
            TicketPricing object with current prices, or None if fetch fails
        """
        # Format date as DD/MM/YYYY for French website
        date_str = date.strftime("%d/%m/%Y")

        # Add date filter parameter
        params = {
            'date_filter[value][date]': date_str,
            'visite-tid': '1'  # Filter for visit tickets
        }

        url = self.BASE_URL
        print(f"📅 Fetching pricing for {date_str}")
        print(f"🔗 URL: {url}")

        try:
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            pricing = TicketPricing(fetch_date=date_str)

            # Parse Passeport pricing
            self._parse_passeport(soup, pricing)

            # Parse Château pricing
            self._parse_chateau(soup, pricing)

            # Parse Trianon pricing
            self._parse_trianon(soup, pricing)

            return pricing

        except requests.RequestException as e:
            print(f"❌ Error fetching pricing: {e}")
            return None

    def _parse_passeport(self, soup: BeautifulSoup, pricing: TicketPricing):
        """Parse Passeport ticket pricing."""
        # Look for "Passeport" section
        passeport_sections = soup.find_all(['div', 'article', 'section'],
                                          string=re.compile(r'Passeport', re.I))

        for section in passeport_sections:
            # Search parent container for pricing
            parent = section.find_parent(['div', 'article', 'section'])
            if not parent:
                continue

            text = parent.get_text()

            # Look for "Tarif Basse saison" or "Tarif Haute saison"
            if 'Basse saison' in text:
                pricing.season = "Basse saison"
            elif 'Haute saison' in text:
                pricing.season = "Haute saison"

            # Extract prices using regex
            # Pattern: "Plein tarif : XX €" or "XX €"
            full_price_match = re.search(r'Plein tarif\s*:?\s*(\d+)\s*€', text)
            if full_price_match:
                pricing.passeport_full = float(full_price_match.group(1))

            # Pattern: "Tarif réduit" followed by price
            reduced_match = re.search(r'Tarif réduit.*?(\d+)\s*€', text, re.DOTALL)
            if reduced_match:
                pricing.passeport_reduced = float(reduced_match.group(1))

            if pricing.passeport_full:
                break  # Found Passeport pricing

    def _parse_chateau(self, soup: BeautifulSoup, pricing: TicketPricing):
        """Parse Château ticket pricing."""
        chateau_sections = soup.find_all(['div', 'article', 'section'],
                                        string=re.compile(r'Billet Château', re.I))

        for section in chateau_sections:
            parent = section.find_parent(['div', 'article', 'section'])
            if not parent:
                continue

            text = parent.get_text()

            # Extract full price
            full_price_match = re.search(r'Plein tarif\s*:?\s*(\d+)\s*€', text)
            if full_price_match:
                pricing.chateau_full = float(full_price_match.group(1))

            # Extract reduced price
            reduced_match = re.search(r'Tarif réduit\s*:?\s*(\d+)\s*€', text)
            if reduced_match:
                pricing.chateau_reduced = float(reduced_match.group(1))

            if pricing.chateau_full:
                break

    def _parse_trianon(self, soup: BeautifulSoup, pricing: TicketPricing):
        """Parse Domaine de Trianon ticket pricing."""
        trianon_sections = soup.find_all(['div', 'article', 'section'],
                                        string=re.compile(r'Billet Domaine de Trianon|Trianon', re.I))

        for section in trianon_sections:
            parent = section.find_parent(['div', 'article', 'section'])
            if not parent:
                continue

            text = parent.get_text()

            # Skip if this is just the Passeport description mentioning Trianon
            if 'Passeport' in text:
                continue

            # Extract full price
            full_price_match = re.search(r'Plein tarif\s*:?\s*(\d+)\s*€', text)
            if full_price_match:
                pricing.trianon_full = float(full_price_match.group(1))

            # Extract reduced price
            reduced_match = re.search(r'Tarif réduit\s*:?\s*(\d+)\s*€', text)
            if reduced_match:
                pricing.trianon_reduced = float(reduced_match.group(1))

            if pricing.trianon_full:
                break


def update_pricing_config(date: datetime, dry_run: bool = True) -> Optional[TicketPricing]:
    """
    Fetch current pricing and optionally update pricing_config.py.

    Args:
        date: Visit date
        dry_run: If True, only display prices without updating config

    Returns:
        TicketPricing object with fetched prices
    """
    scraper = VersaillesPricingScraper()
    pricing = scraper.fetch_pricing(date)

    if not pricing:
        print("❌ Failed to fetch pricing")
        return None

    print(f"\n{'='*70}")
    print(f"PRICING FETCHED FOR {pricing.fetch_date}")
    print(f"{'='*70}")

    print(f"\n🎫 PASSEPORT ({pricing.season or 'Unknown season'})")
    if pricing.passeport_full:
        print(f"   Plein tarif: €{pricing.passeport_full:.2f}")
    if pricing.passeport_reduced:
        print(f"   Tarif réduit: €{pricing.passeport_reduced:.2f}")

    print(f"\n🏰 CHÂTEAU")
    if pricing.chateau_full:
        print(f"   Plein tarif: €{pricing.chateau_full:.2f}")
    if pricing.chateau_reduced:
        print(f"   Tarif réduit: €{pricing.chateau_reduced:.2f}")

    print(f"\n🌸 DOMAINE DE TRIANON")
    if pricing.trianon_full:
        print(f"   Plein tarif: €{pricing.trianon_full:.2f}")
    if pricing.trianon_reduced:
        print(f"   Tarif réduit: €{pricing.trianon_reduced:.2f}")

    print(f"\n{'='*70}")

    if dry_run:
        print("\n💡 Dry run mode - config not updated")
        print("   Run with --update to modify pricing_config.py")
    else:
        print("\n⚠️  Auto-update not yet implemented")
        print("   Manually update configs/pricing_config.py with above values")

    return pricing


def main():
    """CLI interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch current ticket pricing from Versailles website"
    )
    parser.add_argument(
        '--date',
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Visit date in YYYY-MM-DD format (default: today)"
    )
    parser.add_argument(
        '--update',
        action='store_true',
        help="Update pricing_config.py with fetched prices (default: dry run)"
    )

    args = parser.parse_args()

    # Parse date
    try:
        date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"❌ Invalid date format: {args.date}")
        print("   Use YYYY-MM-DD format (e.g., 2025-11-25)")
        return 1

    pricing = update_pricing_config(date, dry_run=not args.update)

    if pricing:
        # Check if all expected prices were found
        missing = []
        if not pricing.passeport_full:
            missing.append("Passeport full price")
        if not pricing.chateau_full:
            missing.append("Château full price")
        if not pricing.trianon_full:
            missing.append("Trianon full price")

        if missing:
            print(f"\n⚠️  Warning: Could not fetch {', '.join(missing)}")
            print("   Website structure may have changed")
            return 1

        print("\n✅ All pricing fetched successfully")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
