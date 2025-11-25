"""
Versailles pricing configuration.
Based on official pricing from https://www.chateauversailles.fr/preparer-ma-visite/billets-tarifs

Environment variables:
- VERSAILLES_USE_LIVE_PRICING: Set to "true" to fetch current pricing from website (default: false)
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Set
from enum import Enum

# Check if live pricing should be used (from environment variable)
USE_LIVE_PRICING = os.getenv("VERSAILLES_USE_LIVE_PRICING", "false").lower() == "true"


class Season(Enum):
    """Pricing seasons."""
    LOW = "low"   # November 1 - March 31
    HIGH = "high"  # April 1 - October 31


class TicketType(Enum):
    """Available ticket types."""
    PASSEPORT = "passeport"          # Full access: Castle + Trianon + Gardens
    CHATEAU = "chateau"              # Castle only
    TRIANON = "trianon"              # Trianon only (opens 12h)
    GARDENS = "gardens"              # Gardens only (special events)
    FREE = "free"                    # Free access


@dataclass
class Visitor:
    """Visitor information for pricing."""
    age: int
    is_eu_resident: bool = False
    has_reduced_rate_card: bool = False  # famille nombreuse, ANCV, etc.


@dataclass
class TicketPrice:
    """Ticket pricing information."""
    ticket_type: TicketType
    season: Season
    full_price: float
    reduced_price: Optional[float] = None
    zones_included: Set[str] = None

    def __post_init__(self):
        if self.zones_included is None:
            self.zones_included = self._get_default_zones()

    def _get_default_zones(self) -> Set[str]:
        """Get zones included in ticket type."""
        if self.ticket_type == TicketType.PASSEPORT:
            return {"Castle", "Garden", "Park", "Trianon"}
        elif self.ticket_type == TicketType.CHATEAU:
            return {"Castle", "Garden", "Park"}  # Castle ticket includes basic garden access
        elif self.ticket_type == TicketType.TRIANON:
            return {"Trianon"}
        elif self.ticket_type == TicketType.GARDENS:
            return {"Garden", "Park"}
        elif self.ticket_type == TicketType.FREE:
            return {"Garden", "Park"}  # Free access to gardens/park only
        return set()


# Official pricing (2025)
PRICING_TABLE = {
    (TicketType.PASSEPORT, Season.LOW): TicketPrice(
        ticket_type=TicketType.PASSEPORT,
        season=Season.LOW,
        full_price=24.0,
        reduced_price=10.0,  # With free access entitlement
    ),
    (TicketType.PASSEPORT, Season.HIGH): TicketPrice(
        ticket_type=TicketType.PASSEPORT,
        season=Season.HIGH,
        full_price=32.0,
        reduced_price=10.0,
    ),
    (TicketType.CHATEAU, Season.LOW): TicketPrice(
        ticket_type=TicketType.CHATEAU,
        season=Season.LOW,
        full_price=21.0,
        reduced_price=13.0,
    ),
    (TicketType.CHATEAU, Season.HIGH): TicketPrice(
        ticket_type=TicketType.CHATEAU,
        season=Season.HIGH,
        full_price=21.0,
        reduced_price=13.0,
    ),
    (TicketType.TRIANON, Season.LOW): TicketPrice(
        ticket_type=TicketType.TRIANON,
        season=Season.LOW,
        full_price=12.0,
        reduced_price=8.0,
    ),
    (TicketType.TRIANON, Season.HIGH): TicketPrice(
        ticket_type=TicketType.TRIANON,
        season=Season.HIGH,
        full_price=12.0,
        reduced_price=8.0,
    ),
    (TicketType.FREE, Season.LOW): TicketPrice(
        ticket_type=TicketType.FREE,
        season=Season.LOW,
        full_price=0.0,
    ),
    (TicketType.FREE, Season.HIGH): TicketPrice(
        ticket_type=TicketType.FREE,
        season=Season.HIGH,
        full_price=0.0,
    ),
}


def get_season(date: datetime) -> Season:
    """
    Determine pricing season for a given date.

    Low season: November 1 - March 31
    High season: April 1 - October 31
    """
    month = date.month
    if 4 <= month <= 10:
        return Season.HIGH
    else:
        return Season.LOW


def is_free_access(visitor: Visitor) -> bool:
    """
    Check if visitor qualifies for free access to Castle and Trianon.

    Free access:
    - Under 18 years old (worldwide)
    - Under 26 years old (EU residents only)
    """
    if visitor.age < 18:
        return True
    if visitor.age < 26 and visitor.is_eu_resident:
        return True
    return False


def get_ticket_price(
    visitor: Visitor,
    ticket_type: TicketType,
    date: datetime,
    use_live_pricing: bool = False
) -> float:
    """
    Calculate ticket price for a visitor.

    Args:
        visitor: Visitor information
        ticket_type: Type of ticket
        date: Visit date
        use_live_pricing: If True, attempts to fetch current pricing from website

    Returns:
        Price in euros
    """
    # Check free access eligibility
    if is_free_access(visitor) and ticket_type in [TicketType.CHATEAU, TicketType.TRIANON, TicketType.PASSEPORT]:
        # Free access to Castle and Trianon
        # Note: Still need Passeport (10€ reduced) for garden events
        if ticket_type == TicketType.PASSEPORT:
            return PRICING_TABLE[(TicketType.PASSEPORT, get_season(date))].reduced_price
        return 0.0

    season = get_season(date)

    # Try to fetch live pricing if requested or enabled via environment variable
    if use_live_pricing or USE_LIVE_PRICING:
        try:
            live_price = _fetch_live_price(visitor, ticket_type, date)
            if live_price is not None:
                return live_price
        except Exception:
            pass  # Fall back to static pricing

    # Use static pricing from table
    pricing = PRICING_TABLE.get((ticket_type, season))

    if not pricing:
        raise ValueError(f"No pricing found for {ticket_type} in {season} season")

    # Check reduced rate eligibility
    if visitor.has_reduced_rate_card and pricing.reduced_price:
        return pricing.reduced_price

    return pricing.full_price


def _fetch_live_price(
    visitor: Visitor,
    ticket_type: TicketType,
    date: datetime
) -> Optional[float]:
    """
    Fetch live pricing from Versailles website.

    Args:
        visitor: Visitor information
        ticket_type: Type of ticket
        date: Visit date

    Returns:
        Price in euros, or None if fetch fails
    """
    try:
        from scripts.fetch_pricing import VersaillesPricingScraper

        scraper = VersaillesPricingScraper()
        pricing = scraper.fetch_pricing(date)

        if not pricing:
            return None

        # Select appropriate price based on ticket type and visitor
        is_reduced = visitor.has_reduced_rate_card

        if ticket_type == TicketType.PASSEPORT:
            return pricing.passeport_reduced if is_reduced else pricing.passeport_full
        elif ticket_type == TicketType.CHATEAU:
            return pricing.chateau_reduced if is_reduced else pricing.chateau_full
        elif ticket_type == TicketType.TRIANON:
            return pricing.trianon_reduced if is_reduced else pricing.trianon_full

        return None

    except Exception:
        return None  # Fall back to static pricing


def calculate_group_cost(
    visitors: List[Visitor],
    ticket_type: TicketType,
    date: datetime
) -> float:
    """
    Calculate total cost for a group of visitors.

    Args:
        visitors: List of visitors
        ticket_type: Type of ticket
        date: Visit date

    Returns:
        Total cost in euros
    """
    total = 0.0
    for visitor in visitors:
        total += get_ticket_price(visitor, ticket_type, date)
    return total


def find_affordable_ticket(
    visitors: List[Visitor],
    budget: float,
    date: datetime,
    preferred_zones: Optional[Set[str]] = None
) -> Optional[TicketPrice]:
    """
    Find the best ticket type within budget.

    Args:
        visitors: List of visitors
        budget: Total budget in euros
        date: Visit date
        preferred_zones: Zones the visitor wants to access (optional)

    Returns:
        Best affordable ticket, or None if budget insufficient
    """
    season = get_season(date)

    # Try tickets in order of coverage (most comprehensive first)
    ticket_priority = [
        TicketType.PASSEPORT,
        TicketType.CHATEAU,
        TicketType.TRIANON,
        TicketType.FREE,
    ]

    for ticket_type in ticket_priority:
        cost = calculate_group_cost(visitors, ticket_type, date)

        if cost <= budget:
            pricing = PRICING_TABLE[(ticket_type, season)]

            # Check if ticket covers preferred zones
            if preferred_zones:
                if not preferred_zones.issubset(pricing.zones_included):
                    continue  # Ticket doesn't cover all requested zones

            return pricing

    return None


def get_accessible_zones(
    visitors: List[Visitor],
    budget: float,
    date: datetime
) -> Set[str]:
    """
    Determine which zones are accessible within budget.

    Args:
        visitors: List of visitors
        budget: Total budget in euros
        date: Visit date

    Returns:
        Set of accessible zone names
    """
    ticket = find_affordable_ticket(visitors, budget, date)

    if ticket:
        return ticket.zones_included

    # If no ticket affordable, only free areas (gardens/park on non-event days)
    return {"Garden", "Park"}


def validate_budget_for_interests(
    visitors: List[Visitor],
    budget: float,
    date: datetime,
    poi_zones: Set[str]
) -> tuple[bool, Optional[str], Optional[TicketPrice]]:
    """
    Validate if budget is sufficient for requested POI zones.

    Args:
        visitors: List of visitors
        budget: Total budget in euros
        date: Visit date
        poi_zones: Zones that POIs are located in

    Returns:
        Tuple of (is_valid, error_message, recommended_ticket)
    """
    # Find ticket that covers requested zones
    ticket = find_affordable_ticket(visitors, budget, date, poi_zones)

    if ticket:
        cost = calculate_group_cost(visitors, ticket.ticket_type, date)
        return (True, None, ticket)

    # Budget insufficient - find what they CAN afford
    affordable_ticket = find_affordable_ticket(visitors, budget, date)

    if affordable_ticket:
        cost = calculate_group_cost(visitors, affordable_ticket.ticket_type, date)
        missing_zones = poi_zones - affordable_ticket.zones_included

        error_msg = (
            f"Budget insufficient. Your budget (€{budget:.2f}) allows {affordable_ticket.ticket_type.value} "
            f"ticket (€{cost:.2f}) which includes: {', '.join(sorted(affordable_ticket.zones_included))}. "
            f"However, your itinerary requires access to: {', '.join(sorted(missing_zones))}. "
            f"Please increase budget or adjust interests to exclude Castle/Trianon POIs."
        )
        return (False, error_msg, affordable_ticket)

    # Can't afford any ticket
    error_msg = (
        f"Budget insufficient (€{budget:.2f}). Minimum ticket cost: "
        f"€{get_ticket_price(visitors[0], TicketType.TRIANON, date):.2f} per person for Trianon. "
        f"Consider visiting only the free gardens and park."
    )
    return (False, error_msg, None)


# Helper function to create visitor list from simple parameters
def create_visitor_group(
    num_adults: int = 1,
    num_children_under_18: int = 0,
    num_youth_18_25_eu: int = 0,
    all_eu_residents: bool = True,
    has_reduced_rate_cards: int = 0
) -> List[Visitor]:
    """
    Create a list of visitors from simple parameters.

    Args:
        num_adults: Number of adults (26+)
        num_children_under_18: Number of children under 18 (free)
        num_youth_18_25_eu: Number of EU residents aged 18-25 (free)
        all_eu_residents: Whether all adults are EU residents
        has_reduced_rate_cards: Number of adults with reduced rate cards

    Returns:
        List of Visitor objects
    """
    visitors = []

    # Children under 18 (free)
    for _ in range(num_children_under_18):
        visitors.append(Visitor(age=15, is_eu_resident=True))

    # Youth 18-25 EU (free)
    for _ in range(num_youth_18_25_eu):
        visitors.append(Visitor(age=20, is_eu_resident=True))

    # Adults
    for i in range(num_adults):
        has_card = i < has_reduced_rate_cards
        visitors.append(Visitor(
            age=30,
            is_eu_resident=all_eu_residents,
            has_reduced_rate_card=has_card
        ))

    return visitors
