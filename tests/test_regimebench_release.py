from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_synthetic_evaluator(task: Path) -> None:
    (task / "tests" / "evaluate.py").write_text(
        "import importlib.util\n"
        "import math\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "import numpy as np\n"
        "import polars as pl\n\n"
        "from verifier import pnl_engine as engine\n\n"
        "EPS = engine.EPS\n"
        "TSTAT_THRESHOLD = engine.TSTAT_THRESHOLD\n"
        "SHARPE_THRESHOLD = engine.SHARPE_THRESHOLD\n"
        "SignalMetrics = engine.SignalMetrics\n"
        "_source_policy_violations = engine.source_policy_violations\n"
        "_sort_columns_for = engine.sort_columns_for\n"
        "_return_columns_for = engine.return_columns_for\n"
        "_with_safe_return_lags = engine.with_safe_return_lags\n"
        "_final_eval_feature_frame = engine.final_eval_feature_frame\n"
        "_filter_signal_period = engine.filter_signal_period\n"
        "_preprocess_signal = engine.preprocess_signal\n"
        "compute_ic_per_timestamp = engine.compute_ic_per_timestamp\n"
        "compute_dk_tstat = engine.compute_dk_tstat\n"
        "compute_daily_sharpe = engine.compute_daily_sharpe\n"
        "compute_autocorrelation = engine.compute_autocorrelation\n"
        "compute_quantile_metrics = engine.compute_quantile_metrics\n"
        "compute_ic_decay = engine.compute_ic_decay\n"
        "compute_return_correlation = engine.compute_return_correlation\n"
        "run_backtest_with_alphalib = engine.run_backtest_with_public_engine\n\n"
        "def compute_combined_score(metrics, judge_result):\n"
        "    return engine.combined_score(metrics)\n\n"
        "def _all_finite(values):\n"
        "    return all(value is not None and math.isfinite(float(value)) for value in values)\n\n"
        "def _build_signal_from_code_snapshot(code_path: Path, data_path: Path, module_name: str) -> pl.DataFrame:\n"
        "    if module_name in sys.modules:\n"
        "        del sys.modules[module_name]\n"
        "    spec = importlib.util.spec_from_file_location(module_name, code_path)\n"
        "    module = importlib.util.module_from_spec(spec)\n"
        "    sys.modules[module_name] = module\n"
        "    spec.loader.exec_module(module)\n"
        "    return module.build_signal(str(data_path))\n"
    )


def test_exports_exact_signal_tasks() -> None:
    tasks = sorted(path.name for path in (REPO_ROOT / "tasks").iterdir() if path.is_dir())
    data_bundle = _load_module(REPO_ROOT / "scripts" / "data_bundle.py", "regimebench_data_bundle")
    assert tasks == sorted(data_bundle.SIGNAL_TASK_NAMES)
    assert len(tasks) == 18
    assert not any(name.startswith("strategy-") for name in tasks)


def test_tree_excludes_private_release_artifacts() -> None:
    forbidden = [
        "GH_ALPHA" + "_PR_KEY",
        "tincans-" + "ai/" + "alpha",
        "HETZNER" + "_API_TOKEN",
        "s3://" + "alpha" + "bench",
        "/" + "Users/",
    ]
    for path in REPO_ROOT.rglob("*"):
        ignored_dirs = {".git", ".venv", ".pytest_cache", ".release-data", "task_splits", "__pycache__"}
        if (
            ignored_dirs.intersection(path.parts)
            or path.suffix == ".parquet"
            or not path.is_file()
            or path.stat().st_size > 2_000_000
        ):
            continue
        assert path.name != ".DS_Store"
        assert "__pycache__" not in path.parts
        text = path.read_text(errors="ignore")
        for needle in forbidden:
            assert needle not in text, f"{needle!r} leaked in {path}"
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    assert not any(path.endswith(".parquet") for path in tracked)


def test_public_release_excludes_reference_solutions() -> None:
    assert not list((REPO_ROOT / "tasks").glob("*/solution"))
    assert not list((REPO_ROOT / "task_configs").glob("*/solution.py"))
    assert not list((REPO_ROOT / "task_configs").glob("*/prep_data.py"))
    assert not list((REPO_ROOT / "tasks").glob("*/scripts/prep_data.py"))
    assert not list((REPO_ROOT / "task_configs").glob("*/notebook.ipynb"))
    assert not (REPO_ROOT / "task_templates" / "signal" / "solution").exists()
    for run_local in (REPO_ROOT / "tasks").glob("*/run_local.py"):
        assert "--solution" not in run_local.read_text()


def test_data_bundle_stages_and_verifies_source(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    script = REPO_ROOT / "scripts" / "data_bundle.py"
    env = {**dict(os.environ), "PYTHONDONTWRITEBYTECODE": "1"}
    subprocess.run(
        [
            sys.executable,
            str(script),
            "stage",
            "--repo-root",
            str(REPO_ROOT),
            "--output-dir",
            str(bundle),
            "--include",
            "benchmark-source",
            "--force",
        ],
        check=True,
        env=env,
    )
    subprocess.run(
        [sys.executable, str(script), "verify", "--bundle-dir", str(bundle)],
        check=True,
        env=env,
    )
    assert (bundle / "payload" / "README.md").exists()
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    assert manifest["bundle_name"] == "regime-bench-base-data"
    readme = (bundle / "README.md").read_text()
    assert readme.startswith("---\nlicense: apache-2.0")
    assert "# RegimeBench Base Data Bundle" in readme


def test_reference_smoke_matches_expected(tmp_path: Path) -> None:
    script = REPO_ROOT / "scripts" / "reference_smoke.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(tmp_path / "reference-smoke"),
        ],
        check=True,
    )
    assert (tmp_path / "reference-smoke" / "reference_summary.json").exists()


def test_public_verifier_scores_synthetic_task(tmp_path: Path) -> None:
    verifier = _load_module(REPO_ROOT / "verifier" / "repair_verifier.py", "regimebench_repair_verifier")
    task = tmp_path / "signal-test"
    (task / "environment" / "data").mkdir(parents=True)
    (task / "tests").mkdir(parents=True)
    _write_synthetic_evaluator(task)
    rows = []
    start = datetime(2024, 1, 1)
    for hour in range(120):
        for idx, ticker in enumerate(["BTC", "ETH", "SOL", "XRP"]):
            ret = (idx - 1.5) * 0.0001 + (hour % 7) * 0.00001
            rows.append(
                {
                    "timestamp_agg": start + timedelta(hours=hour),
                    "ticker": ticker,
                    "close": 100.0 + hour + idx,
                    "volume": 1000.0 + idx,
                    "return": ret,
                    "residual_return": ret,
                    "funding_rate": 0.0,
                }
            )
    frame = pl.DataFrame(rows)
    frame.slice(0, 60 * 4).write_parquet(task / "environment" / "data" / "train.parquet")
    frame.slice(60 * 4).write_parquet(task / "tests" / "test.parquet")
    candidate = tmp_path / "candidate.py"
    candidate.write_text(
        "import polars as pl\n\n"
        "def build_signal(data_path):\n"
        "    df = pl.read_parquet(data_path).sort(['ticker', 'timestamp_agg'])\n"
        "    return df.select(['timestamp_agg', 'ticker', pl.col('close').alias('signal')])\n"
    )
    result = verifier.score_code(
        task_dir=task,
        code_path=candidate,
        output_dir=tmp_path / "out",
        causality_gate=True,
    )
    assert result["status"] == "completed"
    assert (tmp_path / "out" / "daily_pnl.parquet").exists()
    assert (tmp_path / "out" / "results.json").exists()


def test_materialize_report_splits_from_base_data(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"
    task = tasks_root / "signal-test"
    (task / "environment" / "data").mkdir(parents=True)
    (task / "tests").mkdir(parents=True)
    (task / "environment" / "starter_code").mkdir(parents=True)
    (task / "environment" / "starter_code" / "code.py").write_text("def build_signal(data_path):\n    return None\n")
    (task / "task.toml").write_text("name = 'signal-test'\n")
    (task / "instruction.md").write_text("# Synthetic task\n")

    rows = []
    start = datetime(2020, 1, 1)
    for day in range(72 * 31):
        ts = start + timedelta(days=day)
        for idx, ticker in enumerate(["BTC", "ETH"]):
            rows.append(
                {
                    "timestamp_agg": ts,
                    "ticker": ticker,
                    "close": 100.0 + day + idx,
                    "return": 0.001 * idx,
                    "residual_return": 0.001 * idx,
                }
            )
    frame = pl.DataFrame(rows)
    frame.filter(pl.col("timestamp_agg") < datetime(2024, 1, 1)).write_parquet(
        task / "environment" / "data" / "train.parquet"
    )
    frame.filter(pl.col("timestamp_agg") >= datetime(2024, 1, 1)).write_parquet(
        task / "tests" / "test.parquet"
    )

    output = tmp_path / "splits"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "materialize_splits.py"),
            "--profile",
            str(REPO_ROOT / "split_profiles" / "report.toml"),
            "--tasks-root",
            str(tasks_root),
            "--task-names",
            "signal-test",
            "--output",
            str(output),
            "--force",
        ],
        check=True,
    )

    manifest = json.loads((output / "split_manifest.json").read_text())
    assert len(manifest) == 6
    assert (output / "split_manifest.parquet").exists()
    assert (output / "metadata.json").exists()
    generated_tasks = sorted((output / "tasks").glob("signal-test__*"))
    assert len(generated_tasks) == 6
    assert all((path / "environment" / "data" / "train.parquet").exists() for path in generated_tasks)
    assert all((path / "tests" / "test.parquet").exists() for path in generated_tasks)
