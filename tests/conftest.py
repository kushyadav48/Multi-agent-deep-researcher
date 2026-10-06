"""Keep default operational metrics out of the real application history."""

import pytest

from observability.store import MetricsStore


@pytest.fixture(autouse=True)
def isolated_default_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr('observability.store.get_default_metrics_store',
                        lambda: MetricsStore(tmp_path / 'default_metrics.db'))
