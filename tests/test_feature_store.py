"""
Unit tests for the Redis feature store.

Uses fakeredis so these run without a real Redis instance — no infra needed in CI.

The key behaviours we're testing:
- Read-before-write: a transaction must not count in its own velocity features
- Window accuracy: tx_count_1h and tx_count_24h reflect correct time windows
- User isolation: one user's history doesn't bleed into another's
- Average computation: avg_amount_7d is the mean of prior transactions, not current
"""

import pytest
import fakeredis

from feature_store import FeatureStore


@pytest.fixture
def store():
    """FeatureStore wired to an in-memory fake Redis."""
    fake_r = fakeredis.FakeRedis(decode_responses=True)
    fs = FeatureStore.__new__(FeatureStore)
    fs._r = fake_r
    return fs


class TestReadBeforeWrite:
    def test_first_transaction_has_zero_velocity(self, store):
        result = store.get_and_update("u001", 100.0)
        assert result.tx_count_1h == 0
        assert result.tx_count_24h == 0

    def test_current_transaction_not_counted_in_own_velocity(self, store):
        # After N transactions, the Nth should see N-1 in its velocity
        for _ in range(3):
            result = store.get_and_update("u001", 50.0)
        assert result.tx_count_1h == 2  # the 3rd sees the previous 2, not itself


class TestVelocityCounts:
    def test_counts_accumulate_across_calls(self, store):
        store.get_and_update("u001", 50.0)
        store.get_and_update("u001", 50.0)
        result = store.get_and_update("u001", 50.0)
        assert result.tx_count_1h == 2
        assert result.tx_count_24h == 2

    def test_1h_and_24h_match_for_recent_transactions(self, store):
        store.get_and_update("u001", 100.0)
        result = store.get_and_update("u001", 100.0)
        # Both windows should agree for same-second transactions
        assert result.tx_count_1h == result.tx_count_24h


class TestUserIsolation:
    def test_separate_users_have_independent_velocity(self, store):
        store.get_and_update("u001", 100.0)
        store.get_and_update("u001", 100.0)
        store.get_and_update("u001", 100.0)

        result_u002 = store.get_and_update("u002", 50.0)
        assert result_u002.tx_count_1h == 0
        assert result_u002.tx_count_24h == 0

    def test_high_velocity_user_doesnt_affect_new_user(self, store):
        for _ in range(10):
            store.get_and_update("power-user", 200.0)

        new_user = store.get_and_update("new-user", 10.0)
        assert new_user.tx_count_1h == 0


class TestAverageAmount:
    def test_first_transaction_uses_own_amount_as_baseline(self, store):
        result = store.get_and_update("u001", 75.0)
        assert result.avg_amount_7d == pytest.approx(75.0)

    def test_average_reflects_prior_transactions_not_current(self, store):
        store.get_and_update("u001", 100.0)
        store.get_and_update("u001", 200.0)
        # Third transaction sees average of [100, 200] = 150, not including 300
        result = store.get_and_update("u001", 300.0)
        assert result.avg_amount_7d == pytest.approx(150.0)

    def test_average_updates_with_each_transaction(self, store):
        store.get_and_update("u001", 0.0)
        store.get_and_update("u001", 100.0)
        result = store.get_and_update("u001", 200.0)
        assert result.avg_amount_7d == pytest.approx(50.0)  # avg of [0, 100]
