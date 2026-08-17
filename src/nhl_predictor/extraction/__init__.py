"""Pregame status extraction: strict contract, checks, and scoring."""

from nhl_predictor.extraction.assertions import Check, run_hard_rules, score_against_expected
from nhl_predictor.extraction.contract import (
    ExtractionFormatError,
    ExtractionResult,
    PlayerStatusClaim,
    parse_extraction,
)

__all__ = [
    "Check",
    "ExtractionFormatError",
    "ExtractionResult",
    "PlayerStatusClaim",
    "parse_extraction",
    "run_hard_rules",
    "score_against_expected",
]
