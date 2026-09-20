# Shared fixtures: a temporary history database and a test client for the API

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SAFE_SAMPLE = {
    "ph": 7.2, "Hardness": 190.0, "Solids": 800.0, "Chloramines": 3.5, "Sulfate": 220.0,
    "Conductivity": 380.0, "Organic_carbon": 3.5, "Trihalomethanes": 60.0, "Turbidity": 3.9,
}
EXTREME_SAMPLE = {
    "ph": 3.0, "Hardness": 900.0, "Solids": 50000.0, "Chloramines": 12.0,
    "Sulfate": 480.0, "Conductivity": 750.0, "Organic_carbon": 28.0, "Trihalomethanes": 120.0, "Turbidity": 6.5,
}


@pytest.fixture(scope="session")
def history_db(tmp_path_factory):
    return str(tmp_path_factory.mktemp("db") / "history_test.db")


@pytest.fixture(scope="session")
def client(history_db):
    os.environ["WATER_HISTORY_DB"] = history_db
    from fastapi.testclient import TestClient
    from api import main
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def safe_sample():
    return dict(SAFE_SAMPLE)


@pytest.fixture
def extreme_sample():
    return dict(EXTREME_SAMPLE)
