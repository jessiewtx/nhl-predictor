"""Deterministic hard rules for one extraction output.

These are cheap, reproducible, and cannot hallucinate a verdict, so they carry
the failures we refuse to ship: inventing a player, citing evidence the source
never contained, promoting a rumor to a confirmation, or stamping a stale page
with today's timestamp. Every rule maps to a line in SPEC.md.

The same primitives audit the golden set and evaluate a trained model. A label
that violates a rule teaches the model to violate it.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime

from nhl_predictor.extraction.contract import (
    ExtractionFormatError,
    ExtractionResult,
    parse_extraction,
)
from nhl_predictor.schema import PlayerStatus, SourceTier

CONFIRMING_STATUSES = frozenset({PlayerStatus.CONFIRMED_OUT, PlayerStatus.CONFIRMED_STARTER})

_AVAILABILITY_SIGNAL = re.compile(
    r"injur|injured reserve|\bir\b|day[- ]to[- ]day|week[- ]to[- ]week|questionable|doubtful"
    r"|game[- ]time decision|scratch|ruled out|\bout\b|miss(?:es|ed|ing)?\b|return"
    r"|activat|surgery|illness|upper body|lower body|\bstart(?:s|er|ing)?\b|in net"
    r"|between the pipes|line ?up|available",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def fold(text: str) -> str:
    """Lowercase, strip accents, and collapse whitespace for robust matching."""

    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", stripped).strip().lower()


def surname(player_name: str) -> str:
    tokens = re.findall(r"[^\W\d_]+", fold(player_name), flags=re.UNICODE)
    return tokens[-1] if tokens else ""


def _parse_timestamp(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} is not an ISO-8601 timestamp: {value!r}") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone: {value!r}")
    return parsed


def check_grounded_players(result: ExtractionResult, source_text: str) -> Check:
    folded_source = fold(source_text)
    ungrounded = [
        claim.player_name
        for claim in result.player_statuses
        if surname(claim.player_name) not in folded_source
    ]
    if ungrounded:
        return Check("grounded_players", False, f"players absent from source: {ungrounded}")
    return Check("grounded_players", True, "every player appears in the source")


def check_verbatim_evidence(result: ExtractionResult, source_text: str) -> Check:
    folded_source = fold(source_text)
    invented = [
        claim.player_name
        for claim in result.player_statuses
        if fold(claim.evidence) not in folded_source
    ]
    if invented:
        return Check("verbatim_evidence", False, f"evidence not found in source: {invented}")
    return Check("verbatim_evidence", True, "all evidence quoted from the source")


def check_evidence_names_player(result: ExtractionResult) -> Check:
    mismatched = [
        claim.player_name
        for claim in result.player_statuses
        if surname(claim.player_name) not in fold(claim.evidence)
    ]
    if mismatched:
        return Check("evidence_names_player", False, f"evidence omits the player: {mismatched}")
    return Check("evidence_names_player", True, "each evidence span names its player")


def check_no_duplicate_players(result: ExtractionResult) -> Check:
    seen: set[str] = set()
    duplicates: list[str] = []
    for claim in result.player_statuses:
        key = fold(claim.player_name)
        if key in seen:
            duplicates.append(claim.player_name)
        seen.add(key)
    if duplicates:
        return Check("no_duplicate_players", False, f"repeated players: {duplicates}")
    return Check("no_duplicate_players", True, "one claim per player")


def check_tier_discipline(result: ExtractionResult, source_tier: SourceTier) -> Check:
    if result.source_tier != source_tier:
        return Check(
            "tier_discipline",
            False,
            f"declared tier {result.source_tier} but source is {source_tier}",
        )
    if source_tier is SourceTier.OFFICIAL:
        return Check("tier_discipline", True, "official source may confirm")
    promoted = [
        claim.player_name for claim in result.player_statuses if claim.status in CONFIRMING_STATUSES
    ]
    if promoted or result.confirmed_starting_goalie is not None:
        return Check(
            "tier_discipline",
            False,
            f"unofficial source promoted to a confirmation: {promoted or ['starting goalie']}",
        )
    return Check("tier_discipline", True, "unofficial source stayed unconfirmed")


def check_grounded_starter(
    result: ExtractionResult, source_text: str, source_tier: SourceTier
) -> Check:
    starter = result.confirmed_starting_goalie
    if starter is None:
        return Check("grounded_starter", True, "no starter claimed")
    if source_tier is not SourceTier.OFFICIAL:
        return Check("grounded_starter", False, "starter confirmed from an unofficial source")
    if surname(starter) not in fold(source_text):
        return Check("grounded_starter", False, f"starter {starter!r} is absent from the source")
    return Check("grounded_starter", True, f"starter {starter!r} named by an official source")


def check_stated_as_of(result: ExtractionResult, page_as_of_utc: str | None) -> Check:
    if page_as_of_utc is None:
        return Check("stated_as_of", True, "source states no last-updated time")
    try:
        declared = _parse_timestamp(result.as_of_utc, "as_of_utc")
        stated = _parse_timestamp(page_as_of_utc, "page_as_of_utc")
    except ValueError as error:
        return Check("stated_as_of", False, str(error))
    if declared != stated:
        return Check(
            "stated_as_of",
            False,
            f"as_of_utc {result.as_of_utc} ignores the page's stated {page_as_of_utc}",
        )
    return Check("stated_as_of", True, "as_of_utc matches the page's stated update time")


def check_no_future_dating(result: ExtractionResult, cutoff_utc: str) -> Check:
    try:
        declared = _parse_timestamp(result.as_of_utc, "as_of_utc")
        cutoff = _parse_timestamp(cutoff_utc, "cutoff_utc")
    except ValueError as error:
        return Check("no_future_dating", False, str(error))
    if declared > cutoff:
        return Check("no_future_dating", False, f"as_of_utc {result.as_of_utc} is after {cutoff_utc}")
    return Check("no_future_dating", True, "as_of_utc is at or before the cutoff")


def check_abstains_on_silence(result: ExtractionResult, source_text: str) -> Check:
    if _AVAILABILITY_SIGNAL.search(source_text):
        return Check("abstains_on_silence", True, "source raises availability")
    if result.player_statuses or result.confirmed_starting_goalie is not None:
        return Check(
            "abstains_on_silence",
            False,
            "claims produced from a source with no availability signal",
        )
    return Check("abstains_on_silence", True, "abstained on a silent source")


def run_hard_rules(
    output: str | dict | ExtractionResult,
    *,
    source_text: str,
    source_tier: SourceTier,
    cutoff_utc: str,
    page_as_of_utc: str | None = None,
) -> list[Check]:
    """Run every non-negotiable rule from SPEC.md against one output."""

    if isinstance(output, ExtractionResult):
        result = output
        contract = Check("valid_contract", True, "already parsed")
    else:
        try:
            result = parse_extraction(output)
        except ExtractionFormatError as error:
            return [Check("valid_contract", False, str(error))]
        contract = Check("valid_contract", True, "output satisfies the wire contract")

    return [
        contract,
        check_grounded_players(result, source_text),
        check_verbatim_evidence(result, source_text),
        check_evidence_names_player(result),
        check_no_duplicate_players(result),
        check_tier_discipline(result, source_tier),
        check_grounded_starter(result, source_text, source_tier),
        check_stated_as_of(result, page_as_of_utc),
        check_no_future_dating(result, cutoff_utc),
        check_abstains_on_silence(result, source_text),
    ]


def passed(checks: list[Check]) -> bool:
    return all(check.passed for check in checks)


def failures(checks: list[Check]) -> list[Check]:
    return [check for check in checks if not check.passed]


def _claim_pairs(result: ExtractionResult) -> set[tuple[str, str]]:
    return {(surname(claim.player_name), str(claim.status)) for claim in result.player_statuses}


def score_against_expected(
    predicted: ExtractionResult, expected: ExtractionResult
) -> dict[str, float | int | bool]:
    """Per-claim precision/recall/F1 plus exact starter and tier agreement."""

    predicted_pairs = _claim_pairs(predicted)
    expected_pairs = _claim_pairs(expected)
    true_positives = len(predicted_pairs & expected_pairs)
    precision = true_positives / len(predicted_pairs) if predicted_pairs else 1.0
    recall = true_positives / len(expected_pairs) if expected_pairs else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    predicted_starter = predicted.confirmed_starting_goalie
    expected_starter = expected.confirmed_starting_goalie
    return {
        "true_positives": true_positives,
        "predicted_claims": len(predicted_pairs),
        "expected_claims": len(expected_pairs),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "starter_exact_match": (
            (predicted_starter is None and expected_starter is None)
            or (
                predicted_starter is not None
                and expected_starter is not None
                and surname(predicted_starter) == surname(expected_starter)
            )
        ),
        "tier_match": predicted.source_tier == expected.source_tier,
    }
