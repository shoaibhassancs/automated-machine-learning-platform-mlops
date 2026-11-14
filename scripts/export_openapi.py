"""Write the OpenAPI documents (deterministic): docs/openapi-control.json (control plane) and
docs/openapi-serving.json (serving app, P4-7).

    uv run python scripts/export_openapi.py [--check]

`--check` writes nothing and exits 1 if a checked-in file differs from the generated one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from automl_ops.api_control import openapi as control
from automl_ops.api_serving import openapi as serving


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = False
    for module in (control, serving):
        text = module.render_openapi()
        path = Path(module.OPENAPI_PATH)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                print(f"{path} is stale: run scripts/export_openapi.py", file=sys.stderr)
                stale = True
            else:
                print(f"{path.name} is current")
        else:
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path} ({len(text):,} bytes)")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
