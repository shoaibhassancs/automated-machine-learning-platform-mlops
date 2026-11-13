"""Prepare the demo datasets (BUILD_PLAN Step 2).

Downloads UCI Adult and UCI Abalone (only from archive.ics.uci.edu, only if not already in
data/raw/), then writes, for each dataset, a training pool, a disjoint monitoring stream and a
synthetic shifted monitoring copy into data/demo/, plus a manifest.json.

Usage: uv run python scripts/prepare_demo_data.py [--verify-only] [--force-download]
This is demo data: no fairness or real-world drift claims are made.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from automl_ops import demo_data as dd


def _print_report(title: str, report: dict[str, Any]) -> None:
    print(f"\n--- {title}")
    for key, value in report.items():
        print(f"  {key}: {value}")


def _print_checks(results: list[dd.CheckResult]) -> bool:
    print("\nVerification")
    for result in results:
        print(f"  {'PASS' if result.passed else 'FAIL'}  {result.name}  ({result.detail})")
    ok = all(r.passed for r in results)
    print(f"  => {'ALL PASS' if ok else 'SOME CHECKS FAILED'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <data-dir>/demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stream-fraction", type=float, default=0.30)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir if args.out_dir is not None else data_dir / "demo"

    if args.verify_only:
        return 0 if _print_checks(dd.verify_demo_data(data_dir, out_dir)) else 1

    print(dd.DEMO_NOTICE)
    raw_dir = data_dir / "raw"
    raw_files: dict[str, dict[str, Any]] = {}
    for spec in dd.SPECS.values():
        path = raw_dir / spec.zip_name
        if args.force_download or not path.exists():
            print(f"Downloading {spec.zip_url}")
            dd.download(spec.zip_url, path)
        else:
            print(f"Using existing {path} (not downloading)")
        raw_files[spec.zip_name] = {
            "url": spec.zip_url,
            "bytes": path.stat().st_size,
            "sha256": dd.sha256_file(path),
            "note": "SHA-256 computed locally; UCI publishes no checksum to compare against.",
        }

    adult_members = dd.read_zip_members(raw_dir / dd.ADULT.zip_name, ["adult.data", "adult.test"])
    adult, adult_report = dd.prepare_adult(adult_members["adult.data"], adult_members["adult.test"])
    abalone_raw = dd.read_zip_members(raw_dir / dd.ABALONE.zip_name, ["abalone.data"])
    abalone, abalone_report = dd.prepare_abalone(abalone_raw["abalone.data"])
    _print_report("Adult preparation", adult_report)
    _print_report("Abalone preparation", abalone_report)

    entries: dict[str, dict[str, Any]] = {}
    for spec, source in ((dd.ADULT, adult), (dd.ABALONE, abalone)):
        prepared = dd.build_dataset(
            spec, source, seed=args.seed, stream_fraction=args.stream_fraction
        )
        _print_report(
            f"{spec.key}: exact duplicate rows (counted, none removed)", prepared.duplicates
        )
        entries[spec.key] = dd.write_dataset(prepared, out_dir)
        entries[spec.key]["preparation_report"] = (
            adult_report if spec.key == "adult" else abalone_report
        )
        print(
            f"\n{spec.key}: source {len(source)} rows -> pool {len(prepared.pool)}, "
            f"stream {len(prepared.stream)}, shifted {len(prepared.shifted)}"
        )

    manifest = dd.build_manifest(
        entries, raw_files, seed=args.seed, stream_fraction=args.stream_fraction
    )
    manifest_hash = dd.write_manifest(manifest, out_dir / "manifest.json")
    print(f"\nWrote {out_dir}/ (manifest sha256 {manifest_hash})")
    return 0 if _print_checks(dd.verify_demo_data(data_dir, out_dir)) else 1


if __name__ == "__main__":
    sys.exit(main())
