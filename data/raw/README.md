# Raw Vega Datasets

This folder stores the 27 tabular CSV datasets selected from `vega/vega-datasets` for the DashBot reproduction.

Training source manifest:

- `vega_27_manifest.txt`

Notes:

- Only CSV files needed by the project are kept here.
- Original JSON downloads were removed after conversion to keep the repository clean.
- Files not used by the 27-dataset training manifest were removed.
- Run `python scripts/prepare_data.py` from the project root to regenerate `data/processed`.
