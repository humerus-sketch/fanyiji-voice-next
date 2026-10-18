"""Keep segmentation dictionaries without Jieba's optional Paddle checkpoint."""
from PyInstaller.utils.hooks import collect_data_files

datas = collect_data_files("jieba", excludes=["lac_small/model_baseline", "lac_small/model_baseline/**", "**/__pycache__/**"])
