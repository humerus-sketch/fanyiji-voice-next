"""Launch the preserved UI prototype without importing the speech application."""
import sys
from pathlib import Path

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from fanyiji_next.prototype import main
    raise SystemExit(main())
