"""PyInstaller onefile entry point; the helper imports no GUI or model code."""
from fanyiji_next.app_update import helper_main


if __name__ == "__main__":
    raise SystemExit(helper_main())
