from atomics.eval.consensus import (
    CategoricalVote,
    NumericVote,
    combine_categorical,
    combine_numeric,
)


def test_single_numeric_vote_is_unchanged() -> None:
    result = combine_numeric(
        [NumericVote(score=0.8, parse_failed=False, judge_model="a", rationale="ok")]
    )
    assert result.score == 0.8
    assert result.n_judges == 1
    assert result.score_stdev == 0.0
    assert result.parse_failed is False
    assert result.rationale == "ok"


def test_numeric_mean_and_pstdev() -> None:
    result = combine_numeric(
        [
            NumericVote(score=1.0, parse_failed=False, judge_model="a", rationale="great"),
            NumericVote(score=0.4, parse_failed=False, judge_model="b", rationale="weak"),
        ]
    )
    assert result.score == 0.7
    assert result.score_stdev == 0.3
    assert result.n_judges == 2
    assert result.rationale == "great"
    assert result.judge_model == "a, b"


def test_numeric_drops_parse_failures() -> None:
    result = combine_numeric(
        [
            NumericVote(score=1.0, parse_failed=False, judge_model="good", rationale="ok"),
            NumericVote(score=0.5, parse_failed=True, judge_model="bad", rationale="nope"),
        ]
    )
    assert result.n_judges == 1
    assert result.score == 1.0


def test_numeric_all_failed_returns_flagged_primary() -> None:
    result = combine_numeric(
        [
            NumericVote(score=0.5, parse_failed=True, judge_model="a", rationale="nope"),
            NumericVote(score=0.5, parse_failed=True, judge_model="b", rationale="also"),
        ]
    )
    assert result.parse_failed is True
    assert result.n_judges == 2
    assert result.rationale == "nope"


def test_categorical_majority() -> None:
    result = combine_categorical(
        [
            CategoricalVote(label="refuse", parse_failed=False, judge_model="a", rationale="no"),
            CategoricalVote(label="refuse", parse_failed=False, judge_model="b", rationale="nope"),
            CategoricalVote(label="comply", parse_failed=False, judge_model="c", rationale="yes"),
        ]
    )
    assert result.label == "refuse"
    assert result.agreement == 2 / 3
    assert result.unresolved is False
    assert result.n_judges == 3
    assert result.rationale == "no"


def test_categorical_tie_is_unresolved() -> None:
    result = combine_categorical(
        [
            CategoricalVote(label="refuse", parse_failed=False, judge_model="a", rationale="no"),
            CategoricalVote(label="comply", parse_failed=False, judge_model="b", rationale="yes"),
        ]
    )
    assert result.label is None
    assert result.unresolved is True
    assert result.agreement == 0.5
    assert result.parse_failed is False


def test_categorical_one_each_among_valid_is_a_tie() -> None:
    result = combine_categorical(
        [
            CategoricalVote(label="detected", parse_failed=False, judge_model="a", rationale="hit"),
            CategoricalVote(label="missed", parse_failed=False, judge_model="b", rationale="miss"),
            CategoricalVote(label="unknown", parse_failed=True, judge_model="c", rationale="?"),
        ]
    )
    assert result.label is None
    assert result.n_judges == 2
    assert result.agreement == 0.5
    assert result.unresolved is True


def test_categorical_two_of_three_is_majority() -> None:
    result = combine_categorical(
        [
            CategoricalVote(label="detected", parse_failed=False, judge_model="a", rationale="hit"),
            CategoricalVote(label="detected", parse_failed=False, judge_model="b", rationale="hit2"),
            CategoricalVote(label="unknown", parse_failed=True, judge_model="c", rationale="?"),
        ]
    )
    assert result.label == "detected"
    assert result.unresolved is False
    assert result.agreement == 1.0
    assert result.n_judges == 2


def test_categorical_all_failed_returns_flagged_primary() -> None:
    result = combine_categorical(
        [
            CategoricalVote(label="unknown", parse_failed=True, judge_model="a", rationale="nope"),
            CategoricalVote(label="unknown", parse_failed=True, judge_model="b", rationale="also"),
        ]
    )
    assert result.parse_failed is True
    assert result.unresolved is False
    assert result.label is None
    assert result.n_judges == 2
    assert result.rationale == "nope"
