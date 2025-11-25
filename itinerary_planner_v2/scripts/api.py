from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from scripts.planner import Itinerary, ItineraryStep, PlannerConstraints, BudgetConstraint, plan_versailles_itinerary
from scripts.planner_utils import ShortestPathResult

app = FastAPI(title="Versailles Itinerary Planner")

logger = logging.getLogger(__name__)


UserProfileLiteral = Literal["standard", "family", "elder"]
AccessibilityLiteral = Literal["any", "step_free", "stroller"]


class BudgetModel(BaseModel):
    """Budget information for ticket pricing."""
    total_budget: float = Field(..., gt=0, description="Total budget in euros")
    num_adults: int = Field(1, ge=0, description="Number of adults (26+ years)")
    num_children_under_18: int = Field(0, ge=0, description="Number of children under 18 (free)")
    num_youth_18_25_eu: int = Field(0, ge=0, description="Number of EU residents aged 18-25 (free)")
    all_eu_residents: bool = Field(True, description="Whether all adults are EU residents")
    has_reduced_rate_cards: int = Field(0, ge=0, description="Number of adults with reduced rate cards")


class ConstraintsModel(BaseModel):
    interests: List[str] = Field(default_factory=list)
    user_profile: UserProfileLiteral = "standard"
    accessibility: AccessibilityLiteral = "any"
    must_include: List[str] = Field(default_factory=list)
    exclude_ids: List[str] = Field(default_factory=list)
    lunch_break: Optional[bool] = None
    hard_time_limits: bool = False
    start_poi: Optional[str] = Field(None, description="Specific entrance POI ID to start from")
    finish_poi: Optional[str] = Field(None, description="Specific exit POI ID to finish at")
    budget: Optional[BudgetModel] = Field(None, description="Budget constraints for ticket pricing")

    @validator("interests", "must_include", "exclude_ids", each_item=True)
    def no_empty_strings(cls, value: str) -> str:
        if not value:
            raise ValueError("values must be non-empty strings")
        return value


class ItineraryRequest(BaseModel):
    start_time: datetime
    total_duration_minutes: int = Field(..., gt=0, le=12 * 60)
    constraints: ConstraintsModel


class ItineraryStepResponse(BaseModel):
    poi_id: str
    name: str
    arrival_time: datetime
    departure_time: datetime
    stay_minutes: int

    @classmethod
    def from_model(cls, step: ItineraryStep) -> "ItineraryStepResponse":
        return cls(
            poi_id=step.poi_id,
            name=step.name,
            arrival_time=step.arrival_time,
            departure_time=step.departure_time,
            stay_minutes=step.stay_minutes,
        )


class TravelSegmentResponse(BaseModel):
    from_id: str
    to_id: str
    distance_minutes: float
    is_step_free: bool
    stroller_friendly: bool
    path_type: Optional[str]
    notes: Optional[str]

    @classmethod
    def from_model(cls, segment) -> "TravelSegmentResponse":
        return cls(
            from_id=segment.from_id,
            to_id=segment.to_id,
            distance_minutes=segment.distance_min,
            is_step_free=segment.is_step_free,
            stroller_friendly=segment.stroller_friendly,
            path_type=segment.path_type,
            notes=segment.notes,
        )


class TravelSummaryResponse(BaseModel):
    node_ids: List[str]
    total_minutes: float
    segments: List[TravelSegmentResponse]

    @classmethod
    def from_model(cls, result: ShortestPathResult) -> "TravelSummaryResponse":
        return cls(
            node_ids=result.node_ids,
            total_minutes=result.total_minutes,
            segments=[TravelSegmentResponse.from_model(segment) for segment in result.segments],
        )


class ItineraryResponse(BaseModel):
    steps: List[ItineraryStepResponse]
    travel_minutes: float
    visit_minutes: int
    idle_minutes: float
    total_minutes: float
    travel_segments: TravelSummaryResponse

    @classmethod
    def from_model(cls, itinerary: Itinerary) -> "ItineraryResponse":
        return cls(
            steps=[ItineraryStepResponse.from_model(step) for step in itinerary.steps],
            travel_minutes=itinerary.travel_minutes,
            visit_minutes=itinerary.visit_minutes,
            idle_minutes=itinerary.idle_minutes,
            total_minutes=itinerary.total_minutes,
            travel_segments=TravelSummaryResponse.from_model(itinerary.travel_segments),
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/itinerary", response_model=ItineraryResponse)
def create_itinerary(request: ItineraryRequest) -> ItineraryResponse:
    # Convert budget model to budget constraint
    budget_constraint = None
    if request.constraints.budget:
        budget_constraint = BudgetConstraint(
            total_budget=request.constraints.budget.total_budget,
            num_adults=request.constraints.budget.num_adults,
            num_children_under_18=request.constraints.budget.num_children_under_18,
            num_youth_18_25_eu=request.constraints.budget.num_youth_18_25_eu,
            all_eu_residents=request.constraints.budget.all_eu_residents,
            has_reduced_rate_cards=request.constraints.budget.has_reduced_rate_cards,
        )

    constraints = PlannerConstraints(
        interests=request.constraints.interests,
        user_profile=request.constraints.user_profile,
        accessibility=request.constraints.accessibility,
        must_include=request.constraints.must_include,
        exclude_ids=request.constraints.exclude_ids,
        lunch_break=request.constraints.lunch_break,
        hard_time_limits=request.constraints.hard_time_limits,
        start_poi=request.constraints.start_poi,
        finish_poi=request.constraints.finish_poi,
        budget=budget_constraint,
    )

    try:
        itinerary = plan_versailles_itinerary(
            start_time=request.start_time,
            total_duration_minutes=request.total_duration_minutes,
            constraints=constraints,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - safety net
        logger.exception("Planner failed")
        raise HTTPException(status_code=500, detail="Unexpected error during itinerary planning") from exc

    return ItineraryResponse.from_model(itinerary)
