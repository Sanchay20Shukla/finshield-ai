"""FinShield API application entry point."""

from fastapi import FastAPI, Request
from time import perf_counter

from api.routes.scoring import router as scoring_router
from src.monitoring.telemetry import MonitoringRepository
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path
from api.routes.platform import router as platform_router


app = FastAPI(
    title="FinShield AI Fraud API",
    description="Explainable hybrid fraud scoring using ML, rules, and behavioral risk.",
    version="1.0.0",
)
app.include_router(scoring_router)
app.include_router(platform_router)


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/platform/")


app.mount("/platform", StaticFiles(directory=Path(__file__).resolve().parents[1] / "frontend", html=True), name="platform")


@app.middleware("http")
async def record_api_telemetry(request: Request, call_next):
    started = perf_counter()
    status_code, error_type = 500, None
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as error:
        error_type = type(error).__name__
        raise
    finally:
        MonitoringRepository.default().record_api(
            request.method, request.url.path, status_code,
            (perf_counter() - started) * 1000, error_type,
        )


@app.get("/health", tags=["Operations"])
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/ready", tags=["Operations"])
def readiness():
    from api.service import get_scoring_service
    try:
        service = get_scoring_service()
    except Exception as error:
        raise HTTPException(status_code=503, detail="Scoring model unavailable") from error
    return {"status": "ready", "model_version": service.model_version}
