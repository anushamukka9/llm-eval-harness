"""Tests for the llm-eval CLI."""

import pathlib

from llm_eval_harness.cli import main

ROOT = pathlib.Path(__file__).resolve().parents[1]
QUICKSTART = ROOT / "examples" / "quickstart"


def test_cli_init_scaffolds_suite(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    suite_file = tmp_path / "suite.yaml"
    assert suite_file.is_file()
    assert "starter" in suite_file.read_text()
    # second init without --force refuses to overwrite
    assert main(["init"]) == 1
    assert "already exists" in capsys.readouterr().err
    assert main(["init", "--force"]) == 0


def test_cli_run_quickstart(tmp_path, capsys):
    out = tmp_path / "report"
    rc = main([
        "run", str(QUICKSTART / "suite.yaml"),
        "--backend", "mock",
        "--mock-file", str(QUICKSTART / "mock_map.json"),
        "--out", str(out),
    ])
    # intentional-failure case fails -> nonzero exit
    assert rc == 1
    printed = capsys.readouterr().out
    assert "passed=3/4" in printed
    assert (out / "report.json").is_file()
    assert (out / "report.md").is_file()


def test_cli_run_stub_all_pass(tmp_path):
    out = tmp_path / "report"
    suite = tmp_path / "suite.yaml"
    suite.write_text('name: s\ncases:\n  - {id: a, prompt: "hello", expected: "hello", rubric: [{type: exact_match}]}\n')
    assert main(["run", str(suite), "--backend", "stub", "--out", str(out)]) == 0
