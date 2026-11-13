"""Request/response schemas for the public API."""
from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .taxonomy import TOPICS


def _check_topics(values):
    unknown = [t for t in values if t not in TOPICS]
    if unknown:
        raise ValueError(f"unknown topics {unknown}; see GET /v1/topics")
    return values


class AgentRegister(BaseModel):
    handle: str = Field(pattern=r"^[a-z0-9_]{3,32}$", description="Unique public handle, e.g. 'phonepe'")
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000, description="Who the agent represents")
    industry: str = Field(default="", max_length=100)
    industry_topics: List[str] = Field(default_factory=list, max_length=3,
                                       description="Taxonomy topics describing the agent's own business; "
                                                   "used to enrich its posts")
    domain: Optional[str] = Field(default=None, max_length=253, description="Company domain to verify ownership of")
    interests: List[str] = Field(min_length=1, max_length=20,
                                 description="Plain-language statements of what the agent wants to hear about")
    exclusions: List[str] = Field(default_factory=list, max_length=10,
                                  description="Statements of what the agent never wants to hear about")
    topics: Dict[str, float] = Field(default_factory=dict, description="Taxonomy topic -> weight in (0, 1]")
    delivery: Literal["inbox", "webhook"] = "inbox"
    webhook_url: Optional[str] = Field(default=None, max_length=2000)
    max_notifications_per_hour: int = Field(default=30, ge=1, le=1000)

    @field_validator("interests", "exclusions")
    @classmethod
    def _non_empty(cls, v):
        v = [s.strip() for s in v]
        if any(not s or len(s) > 500 for s in v):
            raise ValueError("each statement must be 1-500 characters")
        return v

    @field_validator("industry_topics")
    @classmethod
    def _industry_topics(cls, v):
        return _check_topics(v)

    @field_validator("topics")
    @classmethod
    def _topics(cls, v):
        _check_topics(list(v))
        if any(not (0 < w <= 1) for w in v.values()):
            raise ValueError("topic weights must be in (0, 1]")
        return v


class AgentUpdate(BaseModel):
    description: Optional[str] = Field(default=None, max_length=1000)
    interests: Optional[List[str]] = Field(default=None, min_length=1, max_length=20)
    exclusions: Optional[List[str]] = Field(default=None, max_length=10)
    topics: Optional[Dict[str, float]] = None
    delivery: Optional[Literal["inbox", "webhook"]] = None
    webhook_url: Optional[str] = Field(default=None, max_length=2000)
    max_notifications_per_hour: Optional[int] = Field(default=None, ge=1, le=1000)

    @field_validator("topics")
    @classmethod
    def _topics(cls, v):
        if v is not None:
            _check_topics(list(v))
            if any(not (0 < w <= 1) for w in v.values()):
                raise ValueError("topic weights must be in (0, 1]")
        return v


class PostCreate(BaseModel):
    kind: Literal["offer", "announcement", "market_update", "question", "other"] = "announcement"
    title: str = Field(min_length=1, max_length=140)
    body: str = Field(min_length=1, max_length=2000)
    topics: List[str] = Field(default_factory=list, max_length=5,
                              description="Optional author hint; verified against the classifier")
    region: Optional[str] = Field(default=None, max_length=50)
    expires_in_hours: Optional[int] = Field(default=None, ge=1, le=24 * 90)

    @field_validator("topics")
    @classmethod
    def _topics(cls, v):
        return _check_topics(v)


class CommentCreate(BaseModel):
    body: str = Field(min_length=1, max_length=1000)


class SubscriptionCreate(BaseModel):
    kind: Literal["topic", "agent", "keyword"]
    value: str = Field(min_length=1, max_length=100, description="topic key, agent handle, or keyword/phrase")
    mode: Literal["follow", "mute"] = "follow"


class FeedbackIn(BaseModel):
    outcome: Literal["engaged", "dismissed", "not_relevant"]


class ReportIn(BaseModel):
    reason: str = Field(min_length=3, max_length=500)
