"""Run the standalone source tree or its frozen Windows executable."""
import sys
import json
import traceback
from pathlib import Path

if not getattr(sys, "frozen", False):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

def launch():
    try:
        if "--prototype" in sys.argv:
            sys.argv.remove("--prototype")
            from fanyiji_next.prototype import main
        else:
            from fanyiji_next.app import main
        return main()
    except Exception:
        detail = traceback.format_exc()
        if "--report" in sys.argv and sys.argv.index("--report") + 1 < len(sys.argv):
            destination = Path(sys.argv[sys.argv.index("--report") + 1])
        else:
            destination = Path(sys.executable).parent / "startup-error.json" if getattr(sys, "frozen", False) else Path(__file__).parent / "artifacts/startup-error.json"
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps({"startup_error": detail}, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
        if sys.stderr is not None:
            print(detail, file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(launch())
