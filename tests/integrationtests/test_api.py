import httpx
import pytest
from fastapi.testclient import TestClient

"""
Test APIs written in FastAPI
"""

BASE_URL = "https://frontend-638730968773.us-central1.run.app"


@pytest.mark.asyncio
async def test_read_root():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}")
        assert response.status_code == 200
