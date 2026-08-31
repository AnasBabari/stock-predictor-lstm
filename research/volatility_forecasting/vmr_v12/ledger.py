"""Append-only, immutable evaluation ledger primitives for VMR-V12.

The ledger stores canonical record text rather than caller-owned dictionaries.
Appending returns a new ledger and there are deliberately no update or delete
operations.  Release eligibility still belongs to :mod:`gates`, which enforces
the one-official-record rule for a candidate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .canonical import CanonicalizationError, canonical_json
from .schemas import ProtocolValidationError, validate_evaluation_record


class EvaluationLedgerError(ValueError):
    """Raised when an evaluation record cannot be appended to the ledger."""


@dataclass(frozen=True)
class EvaluationLedger:
    """An immutable append-only sequence of validated evaluation records."""

    _canonical_records: tuple[str, ...] = ()

    @classmethod
    def from_records(cls, records: object) -> EvaluationLedger:
        if not isinstance(records, (list, tuple)):
            raise EvaluationLedgerError("evaluation ledger records must be a sequence")
        ledger = cls()
        for record in records:
            ledger = ledger.append(record)
        return ledger

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        """Return detached record copies in append order."""

        return tuple(json.loads(record) for record in self._canonical_records)

    def append(self, record: object) -> EvaluationLedger:
        try:
            validated = validate_evaluation_record(record)
        except ProtocolValidationError as exc:
            raise EvaluationLedgerError(str(exc)) from exc
        digest = validated["evaluation_record_sha256"]
        if any(
            json.loads(item)["evaluation_record_sha256"] == digest
            for item in self._canonical_records
        ):
            raise EvaluationLedgerError(
                "evaluation record already exists in the append-only ledger"
            )
        try:
            encoded = canonical_json(validated)
        except CanonicalizationError as exc:
            raise EvaluationLedgerError(
                "evaluation record is not canonically serializable"
            ) from exc
        return EvaluationLedger(self._canonical_records + (encoded,))

    def to_list(self) -> list[dict[str, Any]]:
        """Return detached records for gate evaluation or JSON export."""

        return [json.loads(record) for record in self._canonical_records]


__all__ = ["EvaluationLedger", "EvaluationLedgerError"]
