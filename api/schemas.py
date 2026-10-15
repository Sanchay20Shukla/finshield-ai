"""Versioned fraud-scoring API contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(ge=0)
    oldbalanceOrg: float = Field(ge=0)
    newbalanceOrig: float = Field(ge=0)
    oldbalanceDest: float = Field(ge=0)
    newbalanceDest: float = Field(ge=0)
    nameOrig: str = Field(default="C_API_ORIGIN", min_length=1, max_length=32)
    nameDest: str = Field(default="C_API_DESTINATION", min_length=1, max_length=32)
    step: int = Field(default=1, ge=1)
    transaction_velocity: int = Field(default=1, ge=1)
    account_transaction_count: int = Field(default=0, ge=0)
    average_account_amount: float = Field(default=0, ge=0)
    destination_frequency: int = Field(default=0, ge=0)

    @field_validator("nameDest")
    @classmethod
    def accounts_must_differ(cls, value: str, info):
        if value == info.data.get("nameOrig"):
            raise ValueError("origin and destination accounts must differ")
        return value


class RuleSummary(BaseModel):
    rule_id: str
    rule: str
    severity: str
    score: int
    reason: str
    evidence: dict[str, Any]


class TransactionScoreResponse(BaseModel):
    transaction_id: str
    risk_score: float
    fraud_probability: float
    decision: Literal["APPROVE", "MONITOR", "MANUAL_REVIEW", "BLOCK"]
    triggered_rules: list[str]
    risk_factors: list[str]
    rule_details: list[RuleSummary]
    score_breakdown: dict[str, float]
    model_threshold: float
    explanation_method: str
    model_version: str = "unknown"
    explanation_units: str = "model output"
    shap_factors: list[dict[str, Any]] = Field(default_factory=list)
