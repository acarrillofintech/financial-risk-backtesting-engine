"""Tests for Christoffersen VaR backtesting."""

import numpy as np
import pytest

from src.christoffersen import (
    ChristoffersenTestResult,
    ConditionalCoverageTestResult,
    TransitionCounts,
    christoffersen_independence_test,
    conditional_coverage_test,
    count_exception_transitions,
)


def create_clustered_exceptions() -> np.ndarray:
    """Create five exceptions with three grouped together."""
    exceptions = np.zeros(250, dtype=bool)
    exceptions[[40, 100, 101, 102, 200]] = True
    return exceptions


def create_separated_exceptions() -> np.ndarray:
    """Create five exceptions distributed through time."""
    exceptions = np.zeros(250, dtype=bool)
    exceptions[[40, 80, 120, 160, 200]] = True
    return exceptions


def test_transition_counts() -> None:
    """Transition counts should identify all four states."""
    exceptions = np.array(
        [False, False, True, True, False],
        dtype=bool,
    )

    result = count_exception_transitions(exceptions)

    assert isinstance(result, TransitionCounts)
    assert result.no_exception_to_no_exception == 1
    assert result.no_exception_to_exception == 1
    assert result.exception_to_no_exception == 1
    assert result.exception_to_exception == 1


def test_transition_counts_cover_all_transitions() -> None:
    """The four counts should total observations minus one."""
    exceptions = create_clustered_exceptions()

    result = count_exception_transitions(exceptions)

    total_transitions = (
        result.no_exception_to_no_exception
        + result.no_exception_to_exception
        + result.exception_to_no_exception
        + result.exception_to_exception
    )

    assert total_transitions == len(exceptions) - 1


def test_clustered_transition_values() -> None:
    """Clustered example should produce known transitions."""
    result = count_exception_transitions(
        create_clustered_exceptions()
    )

    assert result.no_exception_to_no_exception == 241
    assert result.no_exception_to_exception == 3
    assert result.exception_to_no_exception == 3
    assert result.exception_to_exception == 2


def test_independence_result_dataclass() -> None:
    """The function should return its result dataclass."""
    result = christoffersen_independence_test(
        create_clustered_exceptions()
    )

    assert isinstance(
        result,
        ChristoffersenTestResult,
    )


def test_clustered_exception_probabilities() -> None:
    """An exception should be more likely after another one."""
    result = christoffersen_independence_test(
        create_clustered_exceptions()
    )

    assert (
        result.probability_after_no_exception
        == pytest.approx(3 / 244)
    )
    assert (
        result.probability_after_exception
        == pytest.approx(2 / 5)
    )
    assert (
        result.probability_after_exception
        > result.probability_after_no_exception
    )


def test_clustered_likelihood_ratio() -> None:
    """Clustered example should match the known statistic."""
    result = christoffersen_independence_test(
        create_clustered_exceptions()
    )

    assert result.likelihood_ratio == pytest.approx(
        9.894654,
        rel=1e-6,
    )
    assert result.p_value == pytest.approx(
        0.001658,
        rel=1e-3,
    )


def test_clustered_exceptions_reject_independence() -> None:
    """Strong exception clustering should be rejected."""
    result = christoffersen_independence_test(
        create_clustered_exceptions(),
        significance_level=0.05,
    )

    assert result.rejected is True


def test_separated_exceptions_are_less_dependent() -> None:
    """Separated exceptions should show less dependence."""
    clustered_result = (
        christoffersen_independence_test(
            create_clustered_exceptions()
        )
    )

    separated_result = (
        christoffersen_independence_test(
            create_separated_exceptions()
        )
    )

    assert (
        separated_result.likelihood_ratio
        < clustered_result.likelihood_ratio
    )


def test_conditional_coverage_result_dataclass() -> None:
    """Combined test should return its result dataclass."""
    result = conditional_coverage_test(
        exceptions=create_clustered_exceptions(),
        confidence_level=0.99,
    )

    assert isinstance(
        result,
        ConditionalCoverageTestResult,
    )


def test_conditional_coverage_combines_statistics() -> None:
    """Conditional coverage should combine both LR tests."""
    result = conditional_coverage_test(
        exceptions=create_clustered_exceptions(),
        confidence_level=0.99,
    )

    expected_likelihood_ratio = (
        result.kupiec_test.likelihood_ratio
        + result.independence_test.likelihood_ratio
    )

    assert result.likelihood_ratio == pytest.approx(
        expected_likelihood_ratio
    )
    assert result.likelihood_ratio == pytest.approx(
        11.851464,
        rel=1e-6,
    )


def test_conditional_coverage_rejects_clustered_example() -> None:
    """The combined test should reject the example."""
    result = conditional_coverage_test(
        exceptions=create_clustered_exceptions(),
        confidence_level=0.99,
    )

    assert result.p_value == pytest.approx(
        0.002670,
        rel=1e-3,
    )
    assert result.rejected is True


def test_two_dimensional_exceptions_raise_value_error() -> None:
    """Exception indicators must be one-dimensional."""
    exceptions = np.array(
        [[False, True], [True, False]]
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        christoffersen_independence_test(exceptions)


@pytest.mark.parametrize(
    "exceptions",
    [
        np.array([], dtype=bool),
        np.array([False], dtype=bool),
    ],
)
def test_insufficient_observations_raise_value_error(
    exceptions: np.ndarray,
) -> None:
    """At least two observations should be required."""
    with pytest.raises(
        ValueError,
        match="At least two",
    ):
        christoffersen_independence_test(exceptions)


def test_invalid_exception_value_raises_value_error() -> None:
    """Indicators other than zero and one are invalid."""
    exceptions = np.array([0, 1, 2, 0])

    with pytest.raises(
        ValueError,
        match="boolean values",
    ):
        christoffersen_independence_test(exceptions)


@pytest.mark.parametrize(
    "significance_level",
    [0.0, 1.0, -0.1, 1.1],
)
def test_invalid_significance_level_raises_value_error(
    significance_level: float,
) -> None:
    """Significance must be strictly between zero and one."""
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        christoffersen_independence_test(
            exceptions=create_clustered_exceptions(),
            significance_level=significance_level,
        )


def test_non_numeric_significance_raises_type_error() -> None:
    """Significance must be a real number."""
    with pytest.raises(
        TypeError,
        match="real number",
    ):
        christoffersen_independence_test(
            exceptions=create_clustered_exceptions(),
            significance_level="0.05",  # type: ignore[arg-type]
        )