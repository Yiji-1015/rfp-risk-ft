"""Canonical structured output schema for RFP risk labeling."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MissingInformation(StrictModel):
    is_missing: bool
    missing_details: str


class DomainDependency(StrictModel):
    level: Literal["높음", "중간", "낮음"]
    domain_name: str
    support_status: Literal["발주처 제공", "공동 수행", "수행사 전담", "미지정"]


class RiskFactors(StrictModel):
    cost_driver: str
    scope_uncertainty: str
    responsibility_risk: str
    acceptance_risk: str


class LabelResult(StrictModel):
    requirement_uid: str = Field(min_length=1)
    primary_action: Literal["통상수용", "견적반영", "계약·질의검토"]
    confidence: Literal["높음", "중간", "낮음"]
    reasoning: str = Field(min_length=1)
    evidence: list[str] = Field(min_length=1, max_length=3)
    missing_information: MissingInformation
    domain_dependency: DomainDependency
    risk_factors: RiskFactors
