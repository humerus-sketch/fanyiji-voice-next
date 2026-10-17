"""Source-tree CLI for the package's offline file verification implementation."""
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from fanyiji_next.verification import main


if __name__ == "__main__":
    raise SystemExit(main())
