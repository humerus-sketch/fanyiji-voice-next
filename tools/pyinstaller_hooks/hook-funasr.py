"""FunASR's registration decorators inspect source files at import time."""
from PyInstaller.utils.hooks import collect_data_files

# Explicit implementation/version data only; never collect pretrained weights,
# checkpoints, examples, or an arbitrary repository's files.
datas = collect_data_files(
    "funasr", include_py_files=True, includes=["*.py", "**/*.py", "version.txt"],
)
