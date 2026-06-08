#!/usr/bin/env python3
"""Stage, verify, upload, and download RegimeBench data bundles.

This script is intentionally self-contained. It copies only curated paths into
a bundle payload and computes checksums for every staged file.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable


DEFAULT_DATASET_REPO = "amitvpatel06/regime-bench"

SIGNAL_TASK_NAMES = [
    "signal-beta",
    "signal-cnn-volume",
    "signal-defillama",
    "signal-funding",
    "signal-intraday-seasonality",
    "signal-lottery",
    "signal-mean-reversion",
    "signal-ml",
    "signal-momentum-conditioned",
    "signal-pair-cluster",
    "signal-regime-conditioning",
    "signal-top-trader",
    "signal-upbit",
    "signal-volatility",
    "signal-volume",
    "signal-volume-price-corr",
    "signal-volume-time",
    "signal-weekly-seasonality",
]


@dataclass(frozen=True)
class BundleSource:
    source: str
    target: str | None = None


PROFILE_SOURCES = {
    "benchmark-source": [
        BundleSource(".gitignore"),
        BundleSource("DATA.md"),
        BundleSource("LICENSE"),
        BundleSource("README.md"),
        BundleSource("docs"),
        BundleSource("examples"),
        BundleSource("pyproject.toml"),
        BundleSource("runner"),
        BundleSource("scripts"),
        BundleSource("split_profiles"),
        BundleSource("task_configs"),
        BundleSource("task_templates"),
        BundleSource("tasks"),
        BundleSource("tests"),
        BundleSource("uv.lock"),
        BundleSource("regimebench"),
        BundleSource("verifier"),
    ],
    "base-task-data": [
        BundleSource(
            f"tasks/{task_name}/environment/data/train.parquet",
            f"tasks/{task_name}/environment/data/train.parquet",
        )
        for task_name in SIGNAL_TASK_NAMES
    ]
    + [
        BundleSource(
            f"tasks/{task_name}/tests/test.parquet",
            f"tasks/{task_name}/tests/test.parquet",
        )
        for task_name in SIGNAL_TASK_NAMES
    ],
    "report-split-tasks": [
        BundleSource(
            "task_splits/report-profile/tasks",
            "task_splits/report-profile/tasks",
        ),
        BundleSource(
            "task_splits/report-profile/split_manifest.json",
            "task_splits/report-profile/split_manifest.json",
        ),
        BundleSource(
            "task_splits/report-profile/split_manifest.parquet",
            "task_splits/report-profile/split_manifest.parquet",
        ),
        BundleSource(
            "task_splits/report-profile/metadata.json",
            "task_splits/report-profile/metadata.json",
        ),
    ],
}

EXCLUDE_PATTERNS = [
    ".git/*",
    ".release-data/*",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa",
    "id_ed25519",
    "auth.json",
    "worker.env",
    "bootstrap.env",
    "agent/pi.jsonl",
    "*/_workers/*/jobs/*/agent/pi.jsonl",
    "*/.ssh/*",
    "*/.venv/*",
    "*/__pycache__/*",
    "*/.DS_Store",
    "*/worker.log",
    "*/raw_events/*",
    "verifier_runs/*",
    "tasks/*/output/*",
    "tasks/*/environment/data/*.parquet",
    "tasks/*/tests/test.parquet",
]

AUDIT_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile("/" + r"Users/[A-Za-z0-9._-]+/"),
    re.compile("GH_ALPHA" + "_PR_KEY"),
    re.compile("HETZNER" + "_API_TOKEN"),
    re.compile("s3://" + "alpha" + "bench"),
]


@dataclass(frozen=True)
class StagedFile:
    source: str
    relative_path: str
    size_bytes: int
    sha256: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def matches_any(path: Path, patterns: Iterable[str]) -> bool:
    text = path.as_posix()
    name = path.name
    return any(fnmatch.fnmatch(text, pattern) or fnmatch.fnmatch(name, pattern) for pattern in patterns)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file():
            yield candidate


def target_path_for(
    payload_dir: Path,
    source_file: Path,
    source_root: Path,
    repo_rel: Path,
    target_rel: str | None,
) -> Path:
    if target_rel is None:
        return payload_dir / repo_rel
    target = Path(target_rel)
    if source_root.is_file():
        return payload_dir / target
    return payload_dir / target / source_file.relative_to(source_root)


def audit_file(path: Path) -> list[str]:
    if path.stat().st_size > 10 * 1024 * 1024:
        return []
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return []
    failures = []
    for pattern in AUDIT_PATTERNS:
        if pattern.search(text):
            failures.append(f"{path}: matched audit pattern {pattern.pattern}")
    return failures


def scrub_staged_text(path: Path, repo_root: Path) -> None:
    if path.stat().st_size > 10 * 1024 * 1024:
        return
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return
    original = text
    root_text = repo_root.as_posix().rstrip("/")
    text = text.replace(root_text + "/", "")
    text = text.replace(root_text, ".")
    users_prefix = "/" + "Users/"
    internal_workspace = r"[A-Za-z0-9._-]+/Documents/" + "quant-" + "gym/" + "alpha/?"
    text = re.sub(users_prefix + internal_workspace, "", text)
    text = re.sub(users_prefix + r"[A-Za-z0-9._-]+/[^\"'\s,}]+", "<local-path>", text)
    if text != original:
        path.write_text(text)


def copy_one(repo_root: Path, payload_dir: Path, source_spec: BundleSource) -> list[StagedFile]:
    source_rel = source_spec.source
    source = repo_root / source_rel
    if not source.exists():
        print(f"warning: missing source path: {source_rel}", file=sys.stderr)
        return []

    staged: list[StagedFile] = []
    audit_failures: list[str] = []
    for source_file in iter_files(source):
        repo_rel = source_file.relative_to(repo_root)
        if matches_any(repo_rel, EXCLUDE_PATTERNS) and not source.is_file():
            continue
        target = target_path_for(payload_dir, source_file, source, repo_rel, source_spec.target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
        scrub_staged_text(target, repo_root)
        audit_failures.extend(audit_file(target))
        staged.append(
            StagedFile(
                source=source_rel,
                relative_path=target.relative_to(payload_dir).as_posix(),
                size_bytes=target.stat().st_size,
                sha256=sha256_file(target),
            )
        )
    if audit_failures:
        raise SystemExit("Bundle audit failed:\n" + "\n".join(f"  - {item}" for item in audit_failures))
    return staged


def write_bundle_metadata(bundle_dir: Path, includes: list[str], files: list[StagedFile]) -> None:
    payload_dir = bundle_dir / "payload"
    manifest = {
        "bundle_name": "regime-bench-base-data",
        "created_at": utc_now(),
        "includes": includes,
        "file_count": len(files),
        "total_size_bytes": sum(item.size_bytes for item in files),
        "files": [item.__dict__ for item in files],
    }
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    with (bundle_dir / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for item in sorted(files, key=lambda value: value.relative_path):
            handle.write(f"{item.sha256}  payload/{item.relative_path}\n")
    readme = (
        "---\n"
        "license: apache-2.0\n"
        "pretty_name: RegimeBench Base Data\n"
        "task_categories:\n"
        "- tabular-regression\n"
        "tags:\n"
        "- benchmark\n"
        "- quantitative-finance\n"
        "- trading-signals\n"
        "- distribution-shift\n"
        "- pnl-verification\n"
        "---\n\n"
        "# RegimeBench Base Data Bundle\n\n"
        "This dataset contains the base parquet inputs for the 18 public "
        "RegimeBench signal tasks. Split variants are generated locally from "
        "these base files by the benchmark repository rather than stored as a "
        "separate hosted payload.\n\n"
        "The data bundle is released under the Apache License 2.0, matching "
        "the benchmark source release. It contains processed benchmark data "
        "for research and reproducibility, not investment advice. Hidden/OOS "
        "labels are included, so this release supports local reproducibility "
        "rather than blind leaderboard evaluation.\n\n"
        "Restore this bundle with:\n\n"
        "```bash\n"
        "uv run python scripts/data_bundle.py hf-download --output-dir .release-data\n"
        "uv run python scripts/data_bundle.py verify --bundle-dir .release-data\n"
        "uv run python scripts/data_bundle.py restore --bundle-dir .release-data --force\n"
        "```\n\n"
        "Files are staged under `payload/`, with checksums in `SHA256SUMS` and "
        "bundle metadata in `bundle_manifest.json`.\n\n"
        "Benchmark repository: https://github.com/tincans-ai/regime-bench\n"
    )
    (bundle_dir / "README.md").write_text(readme)
    payload_dir.mkdir(exist_ok=True)


def stage(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    bundle_dir = args.output_dir.resolve()
    payload_dir = bundle_dir / "payload"
    if payload_dir.exists() and not args.force:
        raise SystemExit(f"Refusing to overwrite existing payload without --force: {payload_dir}")
    if payload_dir.exists():
        shutil.rmtree(payload_dir)
    payload_dir.mkdir(parents=True)

    includes = args.include or ["benchmark-source", "base-task-data"]
    files: list[StagedFile] = []
    for include in includes:
        if include not in PROFILE_SOURCES:
            raise SystemExit(f"Unknown include profile {include!r}; expected {sorted(PROFILE_SOURCES)}")
        for source_spec in PROFILE_SOURCES[include]:
            files.extend(copy_one(repo_root, payload_dir, source_spec))

    write_bundle_metadata(bundle_dir, includes, files)
    print(f"staged {len(files)} files into {bundle_dir}")
    return 0


def verify(args: argparse.Namespace) -> int:
    bundle_dir = args.bundle_dir.resolve()
    checksum_path = bundle_dir / "SHA256SUMS"
    if not checksum_path.exists():
        raise SystemExit(f"Missing checksum file: {checksum_path}")
    failures = []
    for line in checksum_path.read_text().splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(None, 1)
        rel = rel.strip()
        path = bundle_dir / rel
        if not path.exists():
            failures.append(f"missing: {rel}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(f"checksum mismatch: {rel}")
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print(f"verified bundle: {bundle_dir}")
    return 0


def restore(args: argparse.Namespace) -> int:
    bundle_dir = args.bundle_dir.resolve()
    repo_root = args.repo_root.resolve()
    verify_args = argparse.Namespace(bundle_dir=bundle_dir)
    verify_status = verify(verify_args)
    if verify_status != 0:
        return verify_status
    payload_dir = bundle_dir / "payload"
    for source_file in iter_files(payload_dir):
        rel = source_file.relative_to(payload_dir)
        target = repo_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not args.force:
            raise SystemExit(f"Refusing to overwrite existing file without --force: {target}")
        shutil.copy2(source_file, target)
    print(f"restored payload from {bundle_dir} into {repo_root}")
    return 0


def hf_upload(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub to upload: uv pip install huggingface_hub") from exc

    bundle_dir = args.bundle_dir.resolve()
    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(bundle_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        path_in_repo=args.path_in_repo,
    )
    print(f"uploaded {bundle_dir} to dataset repo {args.repo_id}")
    return 0


def hf_download(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub to download: uv pip install huggingface_hub") from exc

    local_dir = args.output_dir.resolve()
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(local_dir),
    )
    print(f"downloaded dataset repo {args.repo_id} to {local_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--repo-root", type=Path, default=Path("."))
    stage_parser.add_argument("--output-dir", type=Path, required=True)
    stage_parser.add_argument(
        "--include",
        action="append",
        choices=sorted(PROFILE_SOURCES),
        help="Bundle profile to include. Repeat for multiple profiles.",
    )
    stage_parser.add_argument("--force", action="store_true")
    stage_parser.set_defaults(func=stage)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--bundle-dir", type=Path, required=True)
    verify_parser.set_defaults(func=verify)

    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("--bundle-dir", type=Path, required=True)
    restore_parser.add_argument("--repo-root", type=Path, default=Path("."))
    restore_parser.add_argument("--force", action="store_true")
    restore_parser.set_defaults(func=restore)

    upload_parser = subparsers.add_parser("hf-upload")
    upload_parser.add_argument("--bundle-dir", type=Path, required=True)
    upload_parser.add_argument("--repo-id", default=DEFAULT_DATASET_REPO)
    upload_parser.add_argument("--path-in-repo", default=".")
    upload_parser.add_argument("--private", action="store_true")
    upload_parser.set_defaults(func=hf_upload)

    download_parser = subparsers.add_parser("hf-download")
    download_parser.add_argument("--repo-id", default=DEFAULT_DATASET_REPO)
    download_parser.add_argument("--revision")
    download_parser.add_argument("--output-dir", type=Path, required=True)
    download_parser.set_defaults(func=hf_download)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
