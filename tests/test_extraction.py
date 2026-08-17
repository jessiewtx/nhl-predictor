import json

import pytest

from nhl_predictor.extraction import (
    ExtractionFormatError,
    parse_extraction,
    run_hard_rules,
    score_against_expected,
)
from nhl_predictor.extraction.assertions import failures, passed
from nhl_predictor.schema import SourceTier

SOURCE = (
    "Injury Report. Last Updated: March 6, 2026. "
    "Kevin Bahl Lower Body out. Connor Zary is day-to-day."
)
CUTOFF = "2026-03-07T23:30:00+00:00"
PAGE_AS_OF = "2026-03-06T00:00:00+00:00"


def _output(**overrides) -> dict:
    base = {
        "team": "CGY",
        "as_of_utc": PAGE_AS_OF,
        "player_statuses": [
            {
                "player_name": "Kevin Bahl",
                "status": "confirmed_out",
                "evidence": "Kevin Bahl Lower Body out",
            }
        ],
        "confirmed_starting_goalie": None,
        "source_tier": "official",
    }
    base.update(overrides)
    return base


def _check(output, **kwargs):
    settings = {
        "source_text": SOURCE,
        "source_tier": SourceTier.OFFICIAL,
        "cutoff_utc": CUTOFF,
        "page_as_of_utc": PAGE_AS_OF,
    }
    settings.update(kwargs)
    return run_hard_rules(output, **settings)


def test_well_formed_grounded_output_passes_every_rule() -> None:
    assert passed(_check(_output()))


def test_output_outside_the_contract_is_rejected() -> None:
    with pytest.raises(ExtractionFormatError):
        parse_extraction('{"team": "CGY"}')
    with pytest.raises(ExtractionFormatError):
        parse_extraction(_output(player_statuses=[{"player_name": "X", "status": "probably_out"}]))


def test_fenced_json_is_recovered_rather_than_failed() -> None:
    fenced = "```json\n" + json.dumps(_output(player_statuses=[])) + "\n```"

    assert parse_extraction(fenced).team == "CGY"


def test_invented_player_is_caught_even_with_plausible_evidence() -> None:
    checks = _check(
        _output(
            player_statuses=[
                {
                    "player_name": "Sidney Crosby",
                    "status": "confirmed_out",
                    "evidence": "Sidney Crosby is out",
                }
            ]
        )
    )
    names = {check.name for check in failures(checks)}

    assert "grounded_players" in names
    assert "verbatim_evidence" in names


def test_rumor_cannot_be_promoted_to_a_confirmation() -> None:
    checks = _check(
        _output(source_tier="social_unconfirmed"),
        source_tier=SourceTier.SOCIAL_UNCONFIRMED,
    )

    assert "tier_discipline" in {check.name for check in failures(checks)}


def test_stale_page_may_not_be_stamped_with_the_fetch_time() -> None:
    checks = _check(_output(as_of_utc="2026-08-14T21:30:00+00:00"))
    names = {check.name for check in failures(checks)}

    assert "stated_as_of" in names
    assert "no_future_dating" in names


def test_claims_require_an_availability_signal_in_the_source() -> None:
    silent = "Team store hours and parking information for tonight's visit."
    checks = _check(_output(), source_text=silent, page_as_of_utc=None)

    assert "abstains_on_silence" in {check.name for check in failures(checks)}


def test_scoring_rewards_matching_claims_and_flags_a_wrong_status() -> None:
    expected = parse_extraction(_output())
    wrong_status = parse_extraction(
        _output(
            player_statuses=[
                {
                    "player_name": "Kevin Bahl",
                    "status": "questionable",
                    "evidence": "Kevin Bahl Lower Body out",
                }
            ]
        )
    )

    assert score_against_expected(expected, expected)["f1"] == 1.0
    assert score_against_expected(wrong_status, expected)["f1"] == 0.0
    assert score_against_expected(wrong_status, expected)["starter_exact_match"] is True
