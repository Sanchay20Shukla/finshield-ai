"""Fraud scoring routes."""

from fastapi import APIRouter, Depends

from api.schemas import TransactionScoreRequest, TransactionScoreResponse
from api.service import FraudScoringService, get_scoring_service


router = APIRouter(prefix="/api/v1/transactions", tags=["Fraud scoring"])


@router.post("/score", response_model=TransactionScoreResponse, summary="Score a transaction for fraud risk")
def score_transaction(
    transaction: TransactionScoreRequest,
    service: FraudScoringService = Depends(get_scoring_service),
) -> TransactionScoreResponse:
    return service.score(transaction)
