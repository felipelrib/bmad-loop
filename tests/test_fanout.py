from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import pytest

from bmad_loop import deferredwork
from bmad_loop import runsetup
from bmad_loop.model import RunState
from bmad_loop.fanout import (
    CAPABILITIES,
    FanoutError,
    build_frozen_sweep,
    entry_sha256,
    load_frozen_sweep,
    strict_json_file,
)


def _entry(dw_id: str, status: str = "open", title: str | None = None) -> str:
    return f"### {dw_id}: {title or dw_id}\nstatus: {status}\nreason: test\n"


def _triage(ledger: str) -> dict[str, object]:
    entries = deferredwork.parse_ledger(ledger)
    return {
        "schema_version": 1,
        "workflow": "deferred-sweep-triage",
        "open_ids": ["DW-1", "DW-2", "DW-3"],
        "entry_sha256": {entry.id: entry_sha256(entry) for entry in entries},
        "already_resolved": [],
        "bundles": [
            {"name": "alpha", "dw_ids": ["DW-1"], "intent": "build alpha"},
            {"name": "gamma", "dw_ids": ["DW-3"], "intent": "build gamma"},
        ],
        "blocked": [],
        "skip": [],
        "decisions": [
            {
                "id": "DW-2",
                "question": "Build beta?",
                "context": "",
                "options": [
                    {
                        "key": "yes",
                        "label": "Yes",
                        "effect": "build",
                        "intent": "build beta",
                        "bundle_name": "beta",
                    },
                    {"key": "no", "label": "No", "effect": "keep-open"},
                ],
                "recommendation": "yes",
            }
        ],
    }


@pytest.fixture
def frozen_inputs(tmp_path: Path):
    ledger_text = _entry("DW-1") + "\n" + _entry("DW-2") + "\n" + _entry("DW-3")
    ledger = tmp_path / "deferred-work.md"
    ledger.write_text(ledger_text, encoding="utf-8")
    triage = tmp_path / "triage.json"
    triage.write_text(json.dumps(_triage(ledger_text)), encoding="utf-8")
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps({"DW-2": {"key": "yes", "effect": "close"}}), encoding="utf-8")
    return ledger, triage, decisions


def test_capability_document_is_the_atomic_v1_handshake():
    assert CAPABILITIES == {
        "schema_version": 1,
        "capabilities": {
            "filtered_sweep": {
                "version": 1,
                "triage_workflow": "deferred-sweep-triage",
                "decisions_format": "sweep-decisions-v1",
            }
        },
    }


def test_selects_exact_noncontiguous_and_decision_bundles_from_frozen_semantics(
    frozen_inputs,
):
    ledger, triage, decisions = frozen_inputs
    frozen = load_frozen_sweep(
        schema_version=1,
        bundle_keys=("dw-gamma", "dw-beta"),
        triage_path=triage,
        decisions_path=decisions,
        ledger_path=ledger,
    )

    assert [bundle.name for bundle in frozen.plan.bundles] == ["gamma"]
    assert [decision.id for decision in frozen.plan.decisions] == ["DW-2"]
    assert frozen.answers == {"DW-2": {"key": "yes", "effect": "build"}}
    assert frozen.plan.already_resolved == ()


def test_done_selected_member_is_omitted_before_digest_comparison(frozen_inputs):
    ledger, triage, decisions = frozen_inputs
    ledger.write_text(
        _entry("DW-1", "done 2026-09-02") + "\n" + _entry("DW-2") + "\n" + _entry("DW-3"),
        encoding="utf-8",
    )
    frozen = load_frozen_sweep(
        schema_version=1,
        bundle_keys=("dw-alpha",),
        triage_path=triage,
        decisions_path=decisions,
        ledger_path=ledger,
    )
    assert frozen.plan.bundles == ()


def test_new_unassigned_open_entry_is_ignored(frozen_inputs):
    ledger, triage, decisions = frozen_inputs
    ledger.write_text(ledger.read_text(encoding="utf-8") + "\n" + _entry("DW-99"), encoding="utf-8")
    frozen = load_frozen_sweep(
        schema_version=1,
        bundle_keys=("dw-gamma",),
        triage_path=triage,
        decisions_path=decisions,
        ledger_path=ledger,
    )
    assert frozen.plan.open_ids == frozenset({"DW-3"})


@pytest.mark.parametrize(
    ("member", "message"),
    [
        ("duplicate_key", "duplicate JSON key"),
        ("unknown_bundle", "unknown bundle"),
        ("missing_digest", "entry_sha256.*DW-1"),
        ("changed_entry", "dw-alpha.*DW-1.*digest"),
        ("missing_answer", "dw-beta.*DW-2.*answer"),
        ("unknown_answer", "dw-beta.*DW-2.*unknown answer"),
        ("non_build_answer", "dw-beta.*DW-2.*not a build"),
    ],
)
def test_hostile_frozen_inputs_fail_specifically(frozen_inputs, member, message):
    ledger, triage, decisions = frozen_inputs
    if member == "duplicate_key":
        triage.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    elif member == "unknown_bundle":
        pass
    elif member == "missing_digest":
        data = json.loads(triage.read_text(encoding="utf-8"))
        del data["entry_sha256"]["DW-1"]
        triage.write_text(json.dumps(data), encoding="utf-8")
    elif member == "changed_entry":
        ledger.write_text(
            ledger.read_text(encoding="utf-8").replace("reason: test", "reason: changed", 1),
            encoding="utf-8",
        )
    elif member == "missing_answer":
        decisions.write_text("{}", encoding="utf-8")
    elif member == "unknown_answer":
        decisions.write_text(json.dumps({"DW-2": {"key": "maybe"}}), encoding="utf-8")
    elif member == "non_build_answer":
        decisions.write_text(json.dumps({"DW-2": {"key": "no"}}), encoding="utf-8")

    key = (
        "dw-unknown"
        if member == "unknown_bundle"
        else "dw-beta"
        if member in {"missing_answer", "unknown_answer", "non_build_answer"}
        else "dw-alpha"
    )
    with pytest.raises(FanoutError, match=message):
        load_frozen_sweep(
            schema_version=1,
            bundle_keys=(key,),
            triage_path=triage,
            decisions_path=decisions,
            ledger_path=ledger,
        )


def test_duplicate_malformed_and_missing_selected_ledger_entries_fail(frozen_inputs):
    ledger, triage, decisions = frozen_inputs
    original = ledger.read_text(encoding="utf-8")
    for text, message in [
        (original + "\n" + _entry("DW-1"), "duplicate ledger entry"),
        (original.replace("status: open", "status: maybe", 1), "malformed ledger entry"),
        (original.replace(_entry("DW-1") + "\n", ""), "missing ledger entry"),
    ]:
        ledger.write_text(text, encoding="utf-8")
        with pytest.raises(FanoutError, match=message):
            load_frozen_sweep(
                schema_version=1,
                bundle_keys=("dw-alpha",),
                triage_path=triage,
                decisions_path=decisions,
                ledger_path=ledger,
            )


def test_persisted_contract_round_trips_and_detects_tampering(frozen_inputs, tmp_path):
    ledger, triage, decisions = frozen_inputs
    frozen = load_frozen_sweep(
        schema_version=1,
        bundle_keys=("dw-alpha",),
        triage_path=triage,
        decisions_path=decisions,
        ledger_path=ledger,
    )
    persisted = frozen.persisted()
    assert build_frozen_sweep(persisted, ledger).bundle_keys == ("dw-alpha",)
    persisted["bundle_keys"] = ["dw-gamma"]
    with pytest.raises(FanoutError, match="tampered"):
        build_frozen_sweep(persisted, ledger)


def test_strict_json_rejects_unreadable_invalid_and_duplicate_files(tmp_path):
    with pytest.raises(FanoutError, match="cannot read"):
        strict_json_file(tmp_path / "missing", "triage")
    invalid = tmp_path / "invalid"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(FanoutError, match="valid UTF-8 JSON"):
        strict_json_file(invalid, "triage")
    duplicate = tmp_path / "duplicate"
    duplicate.write_text('{"x":1,"x":2}', encoding="utf-8")
    with pytest.raises(FanoutError, match="duplicate JSON key.*x"):
        strict_json_file(duplicate, "triage")


@pytest.mark.parametrize(
    ("schema", "keys", "triage_change", "message"),
    [
        (2, ("dw-alpha",), None, "fanout schema"),
        (1, (), None, "must not be empty"),
        (1, ("dw-alpha", "dw-alpha"), None, "duplicate key"),
        (1, ("dw-alpha",), ("schema_version", 2), "triage schema_version"),
        (1, ("dw-alpha",), ("workflow", "other"), "workflow must be"),
    ],
)
def test_incompatible_envelope_and_triage_are_rejected(
    frozen_inputs, schema, keys, triage_change, message
):
    ledger, triage, decisions = frozen_inputs
    if triage_change is not None:
        data = json.loads(triage.read_text(encoding="utf-8"))
        data[triage_change[0]] = triage_change[1]
        triage.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(FanoutError, match=message):
        load_frozen_sweep(
            schema_version=schema,
            bundle_keys=keys,
            triage_path=triage,
            decisions_path=decisions,
            ledger_path=ledger,
        )


def test_filtered_resume_requires_and_revalidates_tamper_evident_state(frozen_inputs, tmp_path):
    ledger, triage, decisions = frozen_inputs
    frozen = load_frozen_sweep(
        schema_version=1,
        bundle_keys=("dw-alpha",),
        triage_path=triage,
        decisions_path=decisions,
        ledger_path=ledger,
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    state = RunState(run_id="run", project=str(tmp_path), started_at="now", filtered_sweep=True)
    paths = SimpleNamespace(deferred_work=ledger)

    with pytest.raises(FanoutError, match="cannot read filtered sweep state"):
        runsetup.preflight_filtered_resume(paths=paths, run_dir=run_dir, state=state)

    (run_dir / "sweep.json").write_text(
        json.dumps({"fanout": frozen.persisted()}), encoding="utf-8"
    )
    restored = runsetup.preflight_filtered_resume(paths=paths, run_dir=run_dir, state=state)
    assert restored is not None and restored.bundle_keys == ("dw-alpha",)

    value = json.loads((run_dir / "sweep.json").read_text(encoding="utf-8"))
    value["fanout"]["bundle_keys"] = ["dw-gamma"]
    (run_dir / "sweep.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(FanoutError, match="tampered"):
        runsetup.preflight_filtered_resume(paths=paths, run_dir=run_dir, state=state)
