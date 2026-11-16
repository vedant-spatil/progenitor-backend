from pathlib import Path
from typing import Any

import pytest

CASSETTES = Path(__file__).parent / "cassettes"


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    # Never store credentials or contact details in cassettes.
    return {
        "filter_query_parameters": [("api_key", "REDACTED"), ("email", "REDACTED")],
        "filter_headers": ["authorization", "cookie", "set-cookie"],
        "decode_compressed_response": True,
    }


@pytest.fixture(scope="module")
def vcr_cassette_dir(request: pytest.FixtureRequest) -> str:
    return str(CASSETTES / request.module.__name__.split(".")[-1])
