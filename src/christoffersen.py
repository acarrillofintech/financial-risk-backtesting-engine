"""Christoffersen tests for Value at Risk exceptions."""

from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import NDArray
from scipy.special import xlogy
from scipy.stats import chi2

from src.backtesting import (
    KupiecTestResult,
    kupiec_proportion_of_failures_test,
)


BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class TransitionCounts:
    """Transition counts between consecutive VaR exceptions."""

    no_exception_to_no_exception: int
    no_exception_to_exception: int
    exception_to_no_exception: int
    exception_to_exception: int


@dataclass(frozen=True)
class ChristoffersenTestResult:
    """Result of the Christoffersen independence test."""

    transitions: TransitionCounts
    probability_after_no_exception: float
    probability_after_exception: float
    likelihood_ratio: float
    p_value: float
    significance_level: float
    rejected: bool


@dataclass(frozen=True)
class ConditionalCoverageTestResult:
    """Combined Kupiec and Christoffersen test result."""

    kupiec_test: KupiecTestResult
    independence_test: ChristoffersenTestResult
    likelihood_ratio: float
    p_value: float
    significance_level: float
    rejected: bool


def _validate_exceptions(
    exceptions: BoolArray,
) -> BoolArray:
    """Validate a sequence of VaR exception indicators."""
    normalized_exceptions = np.asarray(exceptions)

    if normalized_exceptions.ndim != 1:
        raise ValueError(
            "Exceptions must be a one-dimensional array."
        )

    if normalized_exceptions.size < 2:
        raise ValueError(
            "At least two exception observations are required."
        )

    if not np.all(
        np.isin(normalized_exceptions, [False, True, 0, 1])
    ):
        raise ValueError(
            "Exceptions must contain only boolean values."
        )

    return normalized_exceptions.astype(bool)


def _validate_significance_level(
    significance_level: float,
) -> float:
    """Validate the statistical significance level."""
    if (
        isinstance(significance_level, bool)
        or not isinstance(significance_level, Real)
    ):
        raise TypeError(
            "Significance level must be a real number."
        )

    normalized_significance = float(significance_level)

    if not 0.0 < normalized_significance < 1.0:
        raise ValueError(
            "Significance level must be between 0 and 1."
        )

    return normalized_significance


def count_exception_transitions(
    exceptions: BoolArray,
) -> TransitionCounts:
    """Count transitions between consecutive exception states."""
    normalized_exceptions = _validate_exceptions(
        exceptions
    )

    previous_states = normalized_exceptions[:-1]
    current_states = normalized_exceptions[1:]

    count_00 = int(
        np.sum(~previous_states & ~current_states)
    )
    count_01 = int(
        np.sum(~previous_states & current_states)
    )
    count_10 = int(
        np.sum(previous_states & ~current_states)
    )
    count_11 = int(
        np.sum(previous_states & current_states)
    )

    return TransitionCounts(
        no_exception_to_no_exception=count_00,
        no_exception_to_exception=count_01,
        exception_to_no_exception=count_10,
        exception_to_exception=count_11,
    )


def _safe_probability(
    successes: int,
    observations: int,
) -> float:
    """Calculate a probability while handling empty states."""
    if observations == 0:
        return 0.0

    return successes / observations


def christoffersen_independence_test(
    exceptions: BoolArray,
    significance_level: float = 0.05,
) -> ChristoffersenTestResult:
    """Test whether VaR exceptions occur independently."""
    normalized_exceptions = _validate_exceptions(
        exceptions
    )

    normalized_significance = (
        _validate_significance_level(
            significance_level
        )
    )

    transitions = count_exception_transitions(
        normalized_exceptions
    )

    count_00 = (
        transitions.no_exception_to_no_exception
    )
    count_01 = (
        transitions.no_exception_to_exception
    )
    count_10 = (
        transitions.exception_to_no_exception
    )
    count_11 = (
        transitions.exception_to_exception
    )

    transitions_after_no_exception = (
        count_00 + count_01
    )
    transitions_after_exception = (
        count_10 + count_11
    )
    total_transitions = (
        transitions_after_no_exception
        + transitions_after_exception
    )

    probability_after_no_exception = (
        _safe_probability(
            successes=count_01,
            observations=transitions_after_no_exception,
        )
    )

    probability_after_exception = (
        _safe_probability(
            successes=count_11,
            observations=transitions_after_exception,
        )
    )

    unconditional_probability = (
        (count_01 + count_11) / total_transitions
    )

    independent_log_likelihood = (
        xlogy(
            count_00 + count_10,
            1.0 - unconditional_probability,
        )
        + xlogy(
            count_01 + count_11,
            unconditional_probability,
        )
    )

    dependent_log_likelihood = (
        xlogy(
            count_00,
            1.0 - probability_after_no_exception,
        )
        + xlogy(
            count_01,
            probability_after_no_exception,
        )
        + xlogy(
            count_10,
            1.0 - probability_after_exception,
        )
        + xlogy(
            count_11,
            probability_after_exception,
        )
    )

    likelihood_ratio = float(
        -2.0
        * (
            independent_log_likelihood
            - dependent_log_likelihood
        )
    )

    likelihood_ratio = max(
        likelihood_ratio,
        0.0,
    )

    p_value = float(
        chi2.sf(likelihood_ratio, df=1)
    )

    return ChristoffersenTestResult(
        transitions=transitions,
        probability_after_no_exception=(
            probability_after_no_exception
        ),
        probability_after_exception=(
            probability_after_exception
        ),
        likelihood_ratio=likelihood_ratio,
        p_value=p_value,
        significance_level=normalized_significance,
        rejected=bool(
            p_value < normalized_significance
        ),
    )


def conditional_coverage_test(
    exceptions: BoolArray,
    confidence_level: float = 0.99,
    significance_level: float = 0.05,
) -> ConditionalCoverageTestResult:
    """Combine Kupiec coverage and Christoffersen independence."""
    normalized_exceptions = _validate_exceptions(
        exceptions
    )

    normalized_significance = (
        _validate_significance_level(
            significance_level
        )
    )

    kupiec_result = (
        kupiec_proportion_of_failures_test(
            exceptions=normalized_exceptions,
            confidence_level=confidence_level,
            significance_level=normalized_significance,
        )
    )

    independence_result = (
        christoffersen_independence_test(
            exceptions=normalized_exceptions,
            significance_level=normalized_significance,
        )
    )

    likelihood_ratio = (
        kupiec_result.likelihood_ratio
        + independence_result.likelihood_ratio
    )

    p_value = float(
        chi2.sf(likelihood_ratio, df=2)
    )

    return ConditionalCoverageTestResult(
        kupiec_test=kupiec_result,
        independence_test=independence_result,
        likelihood_ratio=likelihood_ratio,
        p_value=p_value,
        significance_level=normalized_significance,
        rejected=bool(
            p_value < normalized_significance
        ),
    )


def main() -> None:
    """Run a demonstration with clustered VaR exceptions."""
    exceptions = np.zeros(250, dtype=bool)

    # Five exceptions, three of them grouped together.
    exceptions[[40, 100, 101, 102, 200]] = True

    independence_result = (
        christoffersen_independence_test(
            exceptions=exceptions,
        )
    )

    coverage_result = conditional_coverage_test(
        exceptions=exceptions,
        confidence_level=0.99,
    )

    transitions = independence_result.transitions

    print("Christoffersen independence test")
    print(f"Observations: {len(exceptions)}")
    print(f"Exceptions: {int(np.sum(exceptions))}")

    print("\nTransition counts")
    print(
        "No exception -> No exception: "
        f"{transitions.no_exception_to_no_exception}"
    )
    print(
        "No exception -> Exception: "
        f"{transitions.no_exception_to_exception}"
    )
    print(
        "Exception -> No exception: "
        f"{transitions.exception_to_no_exception}"
    )
    print(
        "Exception -> Exception: "
        f"{transitions.exception_to_exception}"
    )

    print("\nIndependence test")
    print(
        "Exception probability after no exception: "
        f"{independence_result.probability_after_no_exception:.2%}"
    )
    print(
        "Exception probability after exception: "
        f"{independence_result.probability_after_exception:.2%}"
    )
    print(
        "Likelihood-ratio statistic: "
        f"{independence_result.likelihood_ratio:.6f}"
    )
    print(
        f"P-value: {independence_result.p_value:.6f}"
    )
    print(
        f"Model rejected: {independence_result.rejected}"
    )

    print("\nConditional coverage test")
    print(
        "Likelihood-ratio statistic: "
        f"{coverage_result.likelihood_ratio:.6f}"
    )
    print(
        f"P-value: {coverage_result.p_value:.6f}"
    )
    print(
        f"Model rejected: {coverage_result.rejected}"
    )


if __name__ == "__main__":
    main()