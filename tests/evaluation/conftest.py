from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def prohibit_live_benchmark_calls():
    with patch("ollama.Client.embed", side_effect=AssertionError("Real embedding prohibited in pytest")), \
         patch("crewai.crew.Crew.kickoff", side_effect=AssertionError("Real generation prohibited in pytest")), \
         patch("ddgs.ddgs.DDGS.text", side_effect=AssertionError("Real DDGS prohibited in pytest")):
        yield
