"""Loading and validation of eval suites defined as YAML or JSON.

A suite document looks like::

    name: "quickstart"
    description: "Sanity checks for the harness"
    version: "1"
    cases:
      - id: "capital-fr"
        prompt: "What is the capital of France?"
        expected: "Paris"
        rubric:
          - type: "contains"
            expected: "Paris"
            weight: 1.0

The ``expected`` field on a case is optional metadata used by some
scorers when a rule does not carry its own ``expected``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


class SuiteError(ValueError):
    """Raised when a suite document is malformed."""


@dataclass
class ScoringRule:
    """One scoring rule inside a case's rubric.

    ``type`` is one of ``exact_match``, ``regex``, ``contains``,
    ``max_length`` or ``judge``. ``params`` holds the type-specific
    settings (e.g. ``expected``, ``pattern``, ``criteria``); see the
    usage guide for the full reference. ``weight`` scales the rule's
    contribution to the case's aggregated score.
    """

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "ScoringRule") -> "ScoringRule":
        if isinstance(data, ScoringRule):
            return data
        if not isinstance(data, Mapping):
            raise SuiteError(f"rubric entry must be a mapping, got {type(data).__name__}")
        rule_type = data.get("type")
        if not rule_type:
            raise SuiteError("rubric entry is missing required field 'type'")
        weight = data.get("weight", 1.0)
        try:
            weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise SuiteError(f"rule weight must be numeric, got {weight!r}") from exc
        params = {k: v for k, v in data.items() if k not in ("type", "weight")}
        return cls(type=str(rule_type), params=params, weight=weight)


@dataclass
class EvalCase:
    """A single prompt to run against the model under test."""

    id: str
    prompt: str
    system_prompt: str | None = None
    expected: str | None = None
    rubric: list[ScoringRule] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | "EvalCase") -> "EvalCase":
        if isinstance(data, EvalCase):
            return data
        if not isinstance(data, Mapping):
            raise SuiteError(f"case must be a mapping, got {type(data).__name__}")
        missing = {"id", "prompt"} - set(data)
        if missing:
            raise SuiteError(f"case is missing required field(s): {sorted(missing)}")
        rubric = [ScoringRule.from_dict(r) for r in data.get("rubric", [])]
        return cls(
            id=str(data["id"]),
            prompt=str(data["prompt"]),
            system_prompt=data.get("system_prompt"),
            expected=data.get("expected"),
            rubric=rubric,
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class EvalSuite:
    """A named collection of evaluation cases."""

    name: str
    cases: list[EvalCase]
    description: str = ""
    version: str = "1"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalSuite":
        if not isinstance(data, Mapping):
            raise SuiteError("suite document must be a mapping at the top level")
        if "name" not in data:
            raise SuiteError("suite document is missing required field 'name'")
        raw_cases = data.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise SuiteError("suite document needs a non-empty 'cases' list")
        cases = [EvalCase.from_dict(c) for c in raw_cases]
        ids = [c.id for c in cases]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise SuiteError(f"duplicate case id(s): {sorted(dupes)}")
        return cls(
            name=str(data["name"]),
            cases=cases,
            description=str(data.get("description", "")),
            version=str(data.get("version", "1")),
        )


def load_suite(path: str | Path) -> EvalSuite:
    """Load an :class:`EvalSuite` from a YAML, JSON, or JSONL file.

    The format is chosen from the file extension: ``.json`` -> JSON,
    ``.jsonl`` -> one case per line (see :func:`suite_from_jsonl`),
    anything else -> YAML.
    """
    path = Path(path)
    if not path.is_file():
        raise SuiteError(f"suite file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        return suite_from_jsonl(path)
    text = path.read_text(encoding="utf-8")
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            data = yaml.safe_load(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise SuiteError(f"could not parse {path}: {exc}") from exc
    return EvalSuite.from_dict(data)


def load_cases_jsonl(path: str | Path) -> list[EvalCase]:
    """Load eval cases from a JSONL file: one JSON object per line.

    Each line needs ``id`` and ``prompt``; ``expected``,
    ``system_prompt``, ``rubric`` (a list of rule mappings), and
    ``metadata`` are optional. Blank lines are skipped. A malformed line
    raises :class:`SuiteError` naming the line number.
    """
    path = Path(path)
    if not path.is_file():
        raise SuiteError(f"suite file not found: {path}")
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SuiteError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            try:
                cases.append(EvalCase.from_dict(data))
            except SuiteError as exc:
                raise SuiteError(f"{path}:{lineno}: {exc}") from exc
    if not cases:
        raise SuiteError(f"{path}: no cases found")
    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SuiteError(f"{path}: duplicate case id(s): {sorted(dupes)}")
    return cases


def suite_from_jsonl(
    path: str | Path,
    name: str | None = None,
    default_rubric: list[Mapping[str, Any]] | None = None,
    description: str = "",
) -> EvalSuite:
    """Build an :class:`EvalSuite` from a JSONL case file.

    Lines that already carry a ``rubric`` keep it; lines without one get
    ``default_rubric`` (a list of rule mappings, e.g.
    ``[{"type": "exact_match"}]``), or an empty rubric when no default is
    given (an empty rubric scores a neutral 1.0).
    """
    path = Path(path)
    cases = load_cases_jsonl(path)
    rules = (
        [ScoringRule.from_dict(r) for r in default_rubric]
        if default_rubric is not None
        else []
    )
    for case in cases:
        if not case.rubric:
            case.rubric = list(rules)
    return EvalSuite(
        name=name or path.stem,
        cases=cases,
        description=description or f"Cases loaded from {path.name}",
        version="1",
    )
