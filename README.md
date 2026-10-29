# Recommender systems on Amazon Reviews 2023 (Video Games)

IE7275 Machine Learning and Data Analytics, Fall 2026, Northeastern University. Group Project 1: Building and Evaluating Recommendation Systems.
Devansh Nayak, Rishi Hirpara and Venkatashivasai Muppidi.

Eight recommenders (popularity, item-KNN, ALS, BPR, SVD, NeuMF, SASRec and a content-based hybrid) are compared on one dataset under one evaluation protocol: per-user leave-one-out by time, every model ranks the full catalog of 25,612 items (no sampled negatives), and all hyperparameters are chosen on a validation split only. The project asks whether more complex models pay off.

## Results at a glance (test split, Recall@10 / NDCG@10)
- **ALS scored from each user's history: 0.0785 / 0.0432.** This variant was added after seeing validation and test disagree; it is reported next to plain ALS and labelled post hoc.
- SASRec 0.0632 / 0.0335 and plain ALS 0.0598 / 0.0324 are statistically tied. Item-KNN is 0.0562 / 0.0312, BPR 0.0520 / 0.0271.
- NeuMF 0.0377 / 0.0197 is significantly below ALS and BPR, so on this data the extra complexity did not buy accuracy.
- Popularity baseline: 0.0249 / 0.0126. Random guessing: Recall@10 of 0.0004.

## What to open
- `IE7275_Project1_Final.ipynb` is the main notebook: data preparation, exploratory analysis, all eight models with their hyperparameter sweeps, the evaluation, the cold-start analysis and the top-10 demo. It is already executed, so every table and chart is visible without running anything. It is self-contained: its first cells write the whole `recsys` package to disk.
- `notebooks/` holds the data and EDA notebook (01) and the model and analysis notebooks (02 to 04), which import the package.
- `docs/experiment_log.md` lists every hyperparameter run of every model. `figures/` holds the charts. `results/` holds one JSON per run (`<model>_<tag>.json`); runs tagged `final` are the reported ones.

## Data (not included in this repository)
The processed data is derived from the Amazon Reviews 2023 files, whose redistribution terms we have not confirmed, so it is not stored here. To rebuild it:

1. Download the two raw files for the Video Games category from the dataset page, `McAuley-Lab/Amazon-Reviews-2023` on Hugging Face (about 3 GB in total): `raw/review_categories/Video_Games.jsonl` and `raw/meta_categories/meta_Video_Games.jsonl`. Put them in `data/raw/`.
2. Build the modelling data and the cold-start variant. The pipeline is deterministic.

```bash
python -m recsys.preprocess --reviews data/raw/Video_Games.jsonl --meta data/raw/meta_Video_Games.jsonl --out data/processed
python -m recsys.coldstart --src data/processed --out data/processed_coldstart
```

3. Check that you have the same data: `data/processed/meta.json` should show data version `4fe78b90381a` (94,762 users, 25,612 items, 814,586 interactions) and `data/processed_coldstart/meta.json` version `6943ffc82a19`. Every file in `results/` records the data version it was produced from.

## Reproduce
```bash
pip install -r requirements.txt
python -m pytest tests -q                         # the automated tests build their own small data
python -m recsys.run knn --set k=3200 --tag final  # one model on the full data
RECSYS_DATA_DIR=data/processed_coldstart python -m recsys.run knn --set k=3200 --tag final   # and on the cold-start data
python docs/run_finals.py                          # every reported run (about 80 minutes on an Apple M5 laptop)
python -m recsys.compare --final                   # the comparison table
```
In the notebook, `RETRAIN = False` (the default) reads the saved results in `results/`, including the per-user ranks of the final runs, so once the data has been rebuilt (see above) it runs as it is. The demo cell refits ALS (about 13 minutes) because the saved model file is not included. Set `RETRAIN = True` to refit every final model.

## Layout
- `recsys/`: the package. `preprocess` (raw data to splits), `coldstart` (new-user dataset), `data`, `metrics`, `harness` (the evaluation), `models/` (one file per model), `analysis` (confidence intervals and tests), `plots`, `demo`.
- `tests/`: 82 tests covering the preprocessing rules, the metric definitions, the harness and each algorithm's core maths.
- `docs/`: hyperparameter grids (`sweeps.py`), the final settings (`final_settings.json`) and the scripts that rerun them, plus the generators for the notebooks, the findings text and the experiment log.

## Credits
Dataset: Hou, Y., Li, J., He, Z., Yan, A., Chen, X. and McAuley, J. (2024). Bridging Language and Items for Retrieval and Recommendation. arXiv:2403.03952.
