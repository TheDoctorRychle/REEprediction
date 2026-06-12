# REEprediction

Repo made for prediction of the REE (rare earth elements) market.

Predicts the next-day price change (Close[t+1] - Close[t]) for three
instruments (REMX, AMG_AS, KGH_WA) using four models: a from-scratch MLP,
RandomForest, SVM (SVR) and HistGradientBoosting.

Features: Open, High, Low, Close, Volume + engineered Close_mean_5
(5-day rolling mean) and Close_q75_5 (5-day rolling 75th percentile).

Quality metrics (saved in every results CSV): MAE, MSE, RMSE and
direction accuracy (% of days with correctly predicted change direction).

## Running

```
pip install -r requirements.txt
python main.py --ticker REMX --model mlp        # single model run
```

## Experiments (results land in results/ and results/plots/)

| Script | What it does | Output |
|---|---|---|
| `experiments/run_experiments.py` | MLP grid (hidden x lr x epochs) | `results.csv` |
| `experiments/compare_models.py` | MLP vs RF vs SVM vs HGB | `comparison_results.csv`, `model_comparison.png` |
| `experiments/param_study.py` | full hyperparameter grids for all 4 models | `param_study.csv`, `best_configurations.csv`, `param_study_<model>.png` |
| `experiments/learning_curves.py` | MLP epoch curves + train-size curves for all models | `mlp_epoch_curves.csv`, `learning_curves.csv`, `learning_curve_*.png` |
| `experiments/feature_ablation.py` | impact of Close_mean_5 / Close_q75_5 | `feature_ablation.csv`, `feature_ablation.png` |
| `experiments/prediction_plots.py` | predictions vs actual for all models | `predictions_all_models_<ticker>.png` |
| `experiments/make_summary.py` | report-ready markdown tables from the CSVs | `summary_tables.md` |
