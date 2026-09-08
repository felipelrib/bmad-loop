"""Fail-closed frozen-input contract for named deferred-work sweeps."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import deferredwork
from .sweep import Bundle, Decision, TriagePlan, validate_triage

SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
CAPABILITIES = {
    "schema_version": 1,
    "capabilities": {
        "filtered_sweep": {
            "version": 1,
            "triage_workflow": "deferred-sweep-triage",
            "decisions_format": "sweep-decisions-v1",
        }
    },
}


class FanoutError(RuntimeError):
    """A frozen fanout contract is incomplete, incompatible, or unsafe."""


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FanoutError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _loads(raw: bytes, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
        return json.loads(text, object_pairs_hook=_reject_duplicates)
    except FanoutError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise FanoutError(f"{label} is not valid UTF-8 JSON") from exc


def strict_json_file(path: Path, label: str) -> tuple[Any, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FanoutError(f"cannot read {label} file: {path}") from exc
    return _loads(raw, label), hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def entry_sha256(entry: deferredwork.DWEntry) -> str:
    """Digest exact entry content, excluding inter-entry trailing whitespace."""
    return hashlib.sha256(entry.body.rstrip().encode("utf-8")).hexdigest()


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FanoutError(f"{label} must be a JSON object")
    return value


def _validate_triage_document(value: Any) -> tuple[dict[str, Any], TriagePlan, dict[str, str]]:
    document = _require_object(value, "triage")
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise FanoutError("triage schema_version is incompatible; expected 1")
    open_ids = document.get("open_ids")
    if not isinstance(open_ids, list) or not all(isinstance(item, str) for item in open_ids):
        raise FanoutError("triage open_ids must be a list of strings")
    if len(open_ids) != len(set(open_ids)):
        raise FanoutError("triage open_ids contains a duplicate member")
    digests = document.get("entry_sha256")
    if not isinstance(digests, dict):
        raise FanoutError("triage entry_sha256 must be an object")
    missing = sorted(set(open_ids) - set(digests))
    extra = sorted(set(digests) - set(open_ids))
    if missing:
        raise FanoutError(f"triage entry_sha256 is missing {missing[0]}")
    if extra:
        raise FanoutError(f"triage entry_sha256 has unknown member {extra[0]}")
    for dw_id, digest in digests.items():
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise FanoutError(f"triage entry_sha256 for {dw_id} is not lowercase SHA-256")
    try:
        plan, errors = validate_triage(document, None)
    except (AttributeError, TypeError, ValueError) as exc:
        raise FanoutError(f"triage structure is invalid: {exc}") from exc
    if plan is None:
        raise FanoutError("triage is invalid: " + "; ".join(errors))
    return document, plan, {str(key): str(value) for key, value in digests.items()}


def _decision_bundle(
    decision: Decision, answer: Any, bundle_key: str
) -> tuple[Bundle, dict[str, str]]:
    if not isinstance(answer, dict):
        raise FanoutError(f"bundle {bundle_key} decision {decision.id} has no frozen answer")
    key = answer.get("key")
    if not isinstance(key, str) or not key:
        raise FanoutError(f"bundle {bundle_key} decision {decision.id} has no answer key")
    option = decision.option(key)
    if option is None:
        raise FanoutError(f"bundle {bundle_key} decision {decision.id} has unknown answer {key!r}")
    if option.effect != "build":
        raise FanoutError(
            f"bundle {bundle_key} decision {decision.id} answer {key!r} is not a build"
        )
    name = option.bundle_name or "decision-" + decision.id.lower()
    return (
        Bundle(
            name=name,
            dw_ids=(decision.id,),
            intent=option.intent,
            decision_note=(
                f"The human chose option {option.key} ({option.label}) for the "
                f"question: {decision.question}"
            ),
        ),
        {"key": option.key, "effect": "build"},
    )


def _reconcile_bundle(
    bundle_key: str,
    bundle: Bundle,
    entries: dict[str, list[deferredwork.DWEntry]],
    digests: dict[str, str],
) -> Bundle | None:
    remaining: list[str] = []
    for dw_id in bundle.dw_ids:
        matches = entries.get(dw_id, [])
        if not matches:
            raise FanoutError(f"bundle {bundle_key}: missing ledger entry {dw_id}")
        if len(matches) != 1:
            raise FanoutError(f"bundle {bundle_key}: duplicate ledger entry {dw_id}")
        entry = matches[0]
        if entry.done:
            continue
        if not entry.open:
            raise FanoutError(f"bundle {bundle_key}: malformed ledger entry {dw_id}")
        actual = entry_sha256(entry)
        if actual != digests[dw_id]:
            raise FanoutError(f"bundle {bundle_key}: {dw_id} ledger digest changed")
        remaining.append(dw_id)
    if not remaining:
        return None
    return Bundle(
        name=bundle.name,
        dw_ids=tuple(remaining),
        intent=bundle.intent,
        decision_note=bundle.decision_note,
    )


@dataclass(frozen=True)
class FrozenSweep:
    bundle_keys: tuple[str, ...]
    triage_document: dict[str, Any]
    decisions_document: dict[str, Any]
    triage_sha256: str
    decisions_sha256: str
    plan: TriagePlan
    answers: dict[str, dict[str, str]]

    def persisted(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "bundle_keys": list(self.bundle_keys),
            "triage": self.triage_document,
            "decisions": self.decisions_document,
            "triage_sha256": self.triage_sha256,
            "decisions_sha256": self.decisions_sha256,
        }
        body["contract_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        return body


@dataclass(frozen=True)
class FrozenTriage:
    """One validated triage document for a provider-free decisions sitting."""

    triage_document: dict[str, Any]
    triage_sha256: str
    plan: TriagePlan

    def persisted(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "triage": self.triage_document,
            "triage_sha256": self.triage_sha256,
        }
        body["contract_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        return body


def _build(
    *,
    schema_version: int,
    bundle_keys: tuple[str, ...],
    triage_value: Any,
    decisions_value: Any,
    triage_sha256: str,
    decisions_sha256: str,
    ledger_path: Path,
) -> FrozenSweep:
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise FanoutError("fanout schema is incompatible; expected 1")
    for label, digest in (
        ("triage_sha256", triage_sha256),
        ("decisions_sha256", decisions_sha256),
    ):
        if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
            raise FanoutError(f"persisted fanout {label} is not lowercase SHA-256")
    if not bundle_keys:
        raise FanoutError("fanout bundles must not be empty")
    if len(bundle_keys) != len(set(bundle_keys)):
        raise FanoutError("fanout bundles contains a duplicate key")
    triage, plan, digests = _validate_triage_document(triage_value)
    decisions = _require_object(decisions_value, "decisions")

    direct: dict[str, Bundle] = {f"dw-{bundle.name}": bundle for bundle in plan.bundles}
    decision_candidates: dict[str, tuple[Decision, Bundle, dict[str, str]]] = {}
    ambiguous_decision_keys: set[str] = set()
    for decision in plan.decisions:
        answer = decisions.get(decision.id)
        if answer is None:
            continue
        # Derive the key and all executable semantics from the triage option.
        if not isinstance(answer, dict) or not isinstance(answer.get("key"), str):
            continue
        option = decision.option(answer["key"])
        if option is None or option.effect != "build":
            continue
        name = option.bundle_name or "decision-" + decision.id.lower()
        built, canonical_answer = _decision_bundle(decision, answer, f"dw-{name}")
        candidate_key = f"dw-{name}"
        if candidate_key in decision_candidates:
            ambiguous_decision_keys.add(candidate_key)
        decision_candidates[candidate_key] = (decision, built, canonical_answer)

    try:
        ledger_text = ledger_path.read_text(encoding="utf-8") if ledger_path.is_file() else ""
    except (OSError, UnicodeError) as exc:
        raise FanoutError(f"cannot read deferred-work ledger: {ledger_path}") from exc
    entries: dict[str, list[deferredwork.DWEntry]] = {}
    for entry in deferredwork.parse_ledger(ledger_text):
        entries.setdefault(entry.id, []).append(entry)

    selected_bundles: list[Bundle] = []
    selected_decisions: list[Decision] = []
    answers: dict[str, dict[str, str]] = {}
    for bundle_key in bundle_keys:
        if not isinstance(bundle_key, str) or not bundle_key.startswith("dw-"):
            raise FanoutError(f"invalid bundle key {bundle_key!r}")
        if (
            bundle_key in ambiguous_decision_keys
            or bundle_key in direct
            and bundle_key in decision_candidates
        ):
            raise FanoutError(f"ambiguous bundle {bundle_key}")
        if bundle_key in direct:
            candidate = direct[bundle_key]
        elif bundle_key in decision_candidates:
            decision, candidate, answer = decision_candidates[bundle_key]
            selected_decisions.append(decision)
            answers[decision.id] = answer
        else:
            # Diagnose decision-derived names precisely even when the answer is absent/bad.
            matching = []
            for decision in plan.decisions:
                for option in decision.options:
                    name = option.bundle_name or "decision-" + decision.id.lower()
                    if f"dw-{name}" == bundle_key:
                        matching.append(decision)
            if matching:
                decision = matching[0]
                _decision_bundle(decision, decisions.get(decision.id), bundle_key)
            raise FanoutError(f"unknown bundle {bundle_key}")
        reconciled = _reconcile_bundle(bundle_key, candidate, entries, digests)
        if reconciled is not None:
            selected_bundles.append(reconciled)

    selected_plan = TriagePlan(
        open_ids=frozenset(dw_id for bundle in selected_bundles for dw_id in bundle.dw_ids),
        bundles=tuple(bundle for bundle in selected_bundles if not bundle.decision_note),
        decisions=tuple(
            decision
            for decision in selected_decisions
            if any(decision.id in bundle.dw_ids for bundle in selected_bundles)
        ),
    )
    return FrozenSweep(
        bundle_keys=bundle_keys,
        triage_document=triage,
        decisions_document=decisions,
        triage_sha256=triage_sha256,
        decisions_sha256=decisions_sha256,
        plan=selected_plan,
        answers=answers,
    )


def load_frozen_sweep(
    *,
    schema_version: int,
    bundle_keys: tuple[str, ...],
    triage_path: Path,
    decisions_path: Path,
    ledger_path: Path,
) -> FrozenSweep:
    triage, triage_sha256 = strict_json_file(triage_path, "triage")
    decisions, decisions_sha256 = strict_json_file(decisions_path, "decisions")
    return _build(
        schema_version=schema_version,
        bundle_keys=bundle_keys,
        triage_value=triage,
        decisions_value=decisions,
        triage_sha256=triage_sha256,
        decisions_sha256=decisions_sha256,
        ledger_path=ledger_path,
    )


def load_frozen_triage(*, triage_path: Path, ledger_path: Path) -> FrozenTriage:
    """Validate a triage-only fixture against the current open ledger."""
    triage_value, triage_sha256 = strict_json_file(triage_path, "triage")
    triage = _require_object(triage_value, "triage")
    try:
        ledger_text = ledger_path.read_text(encoding="utf-8") if ledger_path.is_file() else ""
    except (OSError, UnicodeError) as exc:
        raise FanoutError(f"cannot read deferred-work ledger: {ledger_path}") from exc
    plan, errors = validate_triage(triage, deferredwork.open_ids(ledger_text))
    if plan is None:
        raise FanoutError("triage is invalid: " + "; ".join(errors))
    return FrozenTriage(triage, triage_sha256, plan)


def build_frozen_triage(value: Any) -> FrozenTriage:
    """Rebuild a persisted triage-only contract without mutable-ledger rereads."""
    persisted = _require_object(value, "persisted triage input")
    required = {"schema_version", "triage", "triage_sha256", "contract_sha256"}
    if set(persisted) != required:
        raise FanoutError("persisted triage input is missing or has unknown fields")
    if (
        type(persisted["schema_version"]) is not int
        or persisted["schema_version"] != SCHEMA_VERSION
    ):
        raise FanoutError("persisted triage input schema is incompatible; expected 1")
    digest = persisted["triage_sha256"]
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise FanoutError("persisted triage input digest is not lowercase SHA-256")
    claimed = persisted["contract_sha256"]
    body = {key: item for key, item in persisted.items() if key != "contract_sha256"}
    actual = hashlib.sha256(_canonical(body)).hexdigest()
    if not isinstance(claimed, str) or claimed != actual:
        raise FanoutError("persisted triage input was tampered with")
    triage = _require_object(persisted["triage"], "triage")
    plan, errors = validate_triage(triage, None)
    if plan is None:
        raise FanoutError("triage is invalid: " + "; ".join(errors))
    return FrozenTriage(triage, digest, plan)


def build_frozen_sweep(value: Any, ledger_path: Path) -> FrozenSweep:
    persisted = _require_object(value, "persisted fanout contract")
    required = {
        "schema_version",
        "bundle_keys",
        "triage",
        "decisions",
        "triage_sha256",
        "decisions_sha256",
        "contract_sha256",
    }
    if set(persisted) != required:
        raise FanoutError("persisted fanout contract is missing or has unknown fields")
    claimed = persisted["contract_sha256"]
    body = {key: value for key, value in persisted.items() if key != "contract_sha256"}
    actual = hashlib.sha256(_canonical(body)).hexdigest()
    if not isinstance(claimed, str) or claimed != actual:
        raise FanoutError("persisted fanout contract was tampered with")
    keys = persisted["bundle_keys"]
    if not isinstance(keys, list) or not all(isinstance(key, str) for key in keys):
        raise FanoutError("persisted fanout bundle_keys must be a list of strings")
    return _build(
        schema_version=persisted["schema_version"],
        bundle_keys=tuple(keys),
        triage_value=persisted["triage"],
        decisions_value=persisted["decisions"],
        triage_sha256=persisted["triage_sha256"],
        decisions_sha256=persisted["decisions_sha256"],
        ledger_path=ledger_path,
    )
