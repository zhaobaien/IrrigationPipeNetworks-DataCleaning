#!/usr/bin/env python3
"""Run the optional historical plotting script when the separate illustrations are available."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "illustrations/Figure 11/plot_figure11_identifiability.py"


def main() -> None:
    subprocess.run([sys.executable, str(SCRIPT)], check=True)


if __name__ == "__main__":
    main()
