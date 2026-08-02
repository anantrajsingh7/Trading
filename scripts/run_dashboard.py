#!/usr/bin/env python3
"""Launch the Streamlit dashboard.

    python scripts/run_dashboard.py [--port 8501]

Equivalent to ``streamlit run dashboard/app.py``; this wrapper exists so the
dashboard starts the same way as every other entry point in the project.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--address", default="localhost")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    app = ROOT / "dashboard" / "app.py"
    if not app.exists():
        print(f"Dashboard not found at {app}", file=sys.stderr)
        return 1

    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "Streamlit is not installed. Install the dashboard extra:\n"
            "    pip install -e '.[dashboard]'",
            file=sys.stderr,
        )
        return 1

    command = [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.port", str(args.port),
        "--server.address", args.address,
    ]
    if args.headless:
        command += ["--server.headless", "true"]
    print("Starting dashboard:", " ".join(command))
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
