# 📱 Mobile Addiction Prediction (DVC + MLflow + Docker)

Predicts a person's **phone addiction level** from daily habits, sleep and mood data.
The project is a reproducible **DVC pipeline**: five regression models are tuned with **Optuna**,
every run is tracked in **MLflow**, and the best model is served by a **Streamlit** app packaged in **Docker**.

## Pipeline

```
data_ingestion -> data_preprocessing -> feature_engineering -> model_building -> model_evaluation
```

| Stage | File | What it does |
|---|---|---|
| data_ingestion | `src/data_ingestion.py` | loads the raw CSV, removes duplicates, splits train/test |
| data_preprocessing | `src/data_preprocessing.py` | drops identifier columns, fixes the column typo, trims text |
| feature_engineering | `src/feature_engineering.py` | adds social media / gaming / education ratios |
| model_building | `src/model_building.py` | tunes 5 models with Optuna, saves them, picks the best by CV R² |
| model_evaluation | `src/model_evaluation.py` | scores every model on the test set, logs to MLflow, writes `reports/` |

All settings (split size, trials, columns, MLflow experiment) live in `params.yaml`.

## Project structure

```
├── .dvc/                  (created by dvc init)
├── .github/workflows/     CI checks
├── .streamlit/config.toml app theme
├── data/
│   ├── raw/               put the raw CSV here
│   ├── ingested/          (created by the pipeline)
│   └── processed/         (created by the pipeline)
├── models/                (created by the pipeline) best_model.pkl, feature_schema.json, candidates/
├── reports/               (created by the pipeline) metrics.json, comparison table and chart
├── src/                   the five pipeline stages
├── .dockerignore  .dvcignore  .gitignore
├── app.py                 Streamlit app
├── Dockerfile
├── dvc.yaml  dvc.lock     pipeline definition (dvc.lock is created by dvc repro)
├── params.yaml
└── requirements.txt
```

## 1. Set up

Open the folder in VS Code, activate your environment, and install the packages:

```powershell
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy your dataset into the project under this exact name:

```powershell
Copy-Item "C:\Users\tsrav\Downloads\phone_addiction_dataset (1).csv" "data\raw\phone_addiction_dataset.csv"
```

Initialise Git and DVC (skip the ones you already did) and let DVC track the raw data:

```powershell
git init
dvc init
dvc add data/raw/phone_addiction_dataset.csv
git add .
git commit -m "Initial project"
```

## 2. Run the pipeline

```powershell
dvc repro
```

This runs all five stages (a few minutes, Random Forest is the slowest) and creates `dvc.lock`,
`models/`, `reports/`, `mlflow.db` and `mlruns/`. Run it again and DVC skips every stage that did not change.
Change a value in `params.yaml` (for example `n_trials`) and only the affected stages rerun.

```powershell
dvc dag          # show the pipeline graph
dvc metrics show # show the best model's metrics
```

## 3. Compare models in MLflow

In a separate terminal:

```powershell
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open http://127.0.0.1:5001, choose **Mobile_Addiction_Prediction**, filter with
`tags.run_type = "final_model"` and sort by `test_r2`. Each model has its own run with its
parameters, `test_r2`, `rmse`, `overfit_gap`, `cv_r2`, `model_size_kb`, `model_id` and Optuna history.
Close that terminal (trash-can icon) to stop the UI.

## 4. Run the app locally

```powershell
streamlit run app.py
```

Open http://localhost:8501.

## 5. Run it in Docker

Install **Docker Desktop**, start it, and make sure `dvc repro` has been run (the image copies
`models/best_model.pkl` and `models/feature_schema.json`).

**Pin your package versions first.** The model only loads with the same scikit-learn version it was trained with.
This writes your installed versions into `requirements.txt`:

```powershell
pip freeze | Select-String -Pattern "^(pandas|numpy|scikit-learn|joblib|optuna|mlflow|dvc|pyyaml|matplotlib|streamlit)==" | ForEach-Object { $_.Line } | Out-File -Encoding ascii requirements.txt
```

Check your Python version with `python --version`, then build and run:

```powershell
docker build -t mobile-addiction-app .
docker run -d --name addiction-app -p 8501:8501 mobile-addiction-app
```

If your Python is not 3.12, build with the same version, for example
`docker build --build-arg PYTHON_VERSION=3.13 -t mobile-addiction-app .`

Open http://localhost:8501. Useful commands:

```powershell
docker ps                       # is it running?
docker logs addiction-app       # see the app's log
docker stop addiction-app       # stop it
docker rm addiction-app         # remove the stopped container
```

**Share the image (optional)**, with your Docker Hub username:

```powershell
docker login
docker tag mobile-addiction-app YOUR_USERNAME/mobile-addiction-app:1.0
docker push YOUR_USERNAME/mobile-addiction-app:1.0
```

Anyone can then run `docker run -p 8501:8501 YOUR_USERNAME/mobile-addiction-app:1.0`.

## Notes

- The best model is chosen by cross-validated R² during training, so the test set is only used for the final scores.
- `data/ingested/`, `data/processed/` and `models/` are produced by DVC and are not stored in Git.
- The app gives a model estimate, not a medical diagnosis.
