"""Write the control-plane OpenAPI document (deterministic) to docs/openapi-control.json.

    uv run python scripts/export_openapi.py [--check]

`--check` writes nothing and exits 1 if the checked-in file differs from the generated one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from automl_ops.api_control.openapi import OPENAPI_PATH, render_openapi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render_openapi()
    path = Path(OPENAPI_PATH)
    if args.check:
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            print(f"{path} is stale: run scripts/export_openapi.py", file=sys.stderr)
            return 1
        print("openapi-control.json is current")
        return 0
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path} ({len(text):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
