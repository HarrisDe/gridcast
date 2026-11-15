from fastapi.testclient import TestClient

from gridcast import __version__
from gridcast.api import app

client = TestClient(app)


def test_health_reports_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
