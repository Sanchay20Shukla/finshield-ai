"""Container orchestration contract tests."""

from pathlib import Path

import pytest
import yaml


pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]


def test_compose_launches_complete_platform_with_health_dependencies():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert set(services) == {"finshield-api", "finshield-dashboard", "postgres"}
    assert services["finshield-api"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert services["finshield-dashboard"]["depends_on"]["finshield-api"]["condition"] == "service_healthy"
    assert "healthcheck" in services["postgres"]
    assert "healthcheck" in services["finshield-api"]
    assert "healthcheck" in services["finshield-dashboard"]


def test_compose_preserves_large_data_and_runtime_state_outside_images():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    dashboard_volumes = compose["services"]["finshield-dashboard"]["volumes"]
    api_volumes = compose["services"]["finshield-api"]["volumes"]
    assert "./data/raw:/app/data/raw:ro" in dashboard_volumes
    assert "./data/processed:/app/data/processed" in dashboard_volumes
    assert "./models:/app/models:ro" in dashboard_volumes
    assert "./models:/app/models:ro" in api_volumes
    assert "finshield_postgres" in compose["volumes"]
