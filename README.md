# Assignment 3 — MNIST deep learning with MLflow and DagsHub

**Deliverables:** `train_mnist_mlflow.py` (submit as Python code on DC Connect), `Assignment3_Final_Report.docx` (generated after you run the real experiments), and your real GitHub/DagsHub project URLs and screenshots.

> **Important:** No training, DagsHub authentication, or remote GitHub push was performed to make this starter package. The DOCX template contains complete Parts A and B, and marked evidence slots for Part C. Run the code in your own account to produce authentic metrics and screenshots.

## 1. Create or connect the repositories

1. Sign in to https://github.com and create a repository named `mnist-mlflow-assignment3` (or another name you choose).
2. Sign in to https://dagshub.com. Choose **Create** → **New Repository** → **Import Repository** → **GitHub Connect**, then select your GitHub repository. The connected DagsHub repo is what hosts MLflow.
3. Check that the DagsHub username and repository name match the `.env` file you will create below.

Official DagsHub guide: https://dagshub.com/docs/quick_start/connect_existing_project/

## 2. Set up your Mac (Terminal in VS Code)

Open the *unzipped* `Assignment3_MNIST_MLflow` project folder in VS Code. In the Terminal, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` to replace `YOUR_DAGSHUB_USERNAME` and `YOUR_GITHUB_USERNAME` with your real account names, and adjust the repository name if necessary. Save it. Never put a password, access token, or OpenAI key in a file committed to GitHub.

Run the interactive DagsHub login once:

```bash
dagshub login
```

A browser sign-in may appear. Use **DagsHub account login**, not the OpenAI API key. Then check dependencies:

```bash
python -c "import tensorflow, mlflow, dagshub; print(tensorflow.__version__, mlflow.__version__)"
```

TensorFlow 2.21 has an Apple Silicon Python 3.13 wheel, although a separate virtual environment is recommended to avoid package conflicts. If package resolution fails, try Python 3.11 or 3.12 in a *new* virtual environment.

## 3. Run and track the experiments

Optional one-epoch **connectivity smoke test** (not acceptable as one of the four required experiments):

```bash
python train_mnist_mlflow.py --smoke-test
```

Run the required four experiments:

```bash
python train_mnist_mlflow.py --only all
```

The baseline has precisely **1 Dense hidden layer, 128 ReLU units, Adam optimizer, batch size 64, and 50 epochs**. It includes Flatten and a 10-unit Softmax output. Three further runs compare: a 256-unit wider model, a 2-hidden-layer model (128 + 64 units), and an SGD optimizer model. Each additional run trains 30 epochs. Every run is logged with a meaningful name and explicit parameters; `mlflow.tensorflow.autolog()` records TensorFlow/Keras training metrics and artifacts.

If a run fails, rerun just that configuration without losing the completed output rows:

```bash
python train_mnist_mlflow.py --only baseline
python train_mnist_mlflow.py --only wider
python train_mnist_mlflow.py --only deeper
python train_mnist_mlflow.py --only sgd
```

Results are saved under `outputs/experiment_results.csv` with real final-epoch train/validation accuracy and loss for each config. With all four successful runs, `outputs/best_run.json` records the highest final-epoch validation accuracy (final val loss breaks ties). **Do not use `smoke_results.csv` for submission.** Run names from partial reruns can have different creation times in MLflow; use the recorded run IDs to identify matching experiment rows.

## 4. Obtain DagsHub evidence

Go to your DagsHub repository → **Experiments** → **MLflow** (or open the MLflow interface offered in the project). Capture these *real* screenshots and save them as:

| Required evidence | Screenshot filename |
|---|---|
| List showing all 4 successful runs | `screenshots/01_run_list.png` |
| One run's logged hyperparameters (layers, nodes, optimizer, batch size) | `screenshots/02_logged_parameters.png` |
| Logged training and validation accuracy and loss | `screenshots/03_logged_metrics.png` |
| Comparison view of at least 2 runs | `screenshots/04_run_comparison.png` |

**MLflow Tracking URI format:** `https://dagshub.com/<DAGSHUB_OWNER>/<DAGSHUB_REPO>.mlflow` (the Python script prints the exact actual configured URI when it starts). This is a remote URL, not `http://localhost:5000`.

## 5. Create the final Word report

After genuine results and screenshots exist:

```bash
python build_report.py
```

This copies the template's complete Parts A/B and fills in Part C with your actual links, hyperparameters, CSV metrics and best-model selection. Images in `screenshots/` are embedded in the report automatically. Check the DOCX before submitting: all 4 runs must be present, all 4 screenshots visible, and every URL should open.

## 6. Push code to GitHub

Create the empty GitHub repository through the website first. From the project folder, with your own username:

```bash
git init
git add train_mnist_mlflow.py build_report.py requirements.txt .env.example .gitignore README.md Assignment3_Report_Template.docx
git commit -m "MNIST DagsHub MLflow assignment experiments"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/mnist-mlflow-assignment3.git
git push -u origin main
```

GitHub might request browser authorization or a personal access token through a credential helper. Do not paste tokens into chat or commit secrets.

## 7. DC Connect submission checklist

- [ ] Post `train_mnist_mlflow.py` as the Python code submission (and optionally `build_report.py` if the instructor allows multiple files).
- [ ] Upload the **generated** `Assignment3_Final_Report.docx` with Parts A/B, Part C comparison and genuine screenshots.
- [ ] In the report include verified GitHub URL, DagsHub URL and MLflow Tracking URI.
- [ ] Verify 4 real final runs (baseline + 3), not just smoke tests.
- [ ] Ensure metrics and images match the real MLflow run IDs.
- [ ] Submit before **October 23, 2026 at 11:59 PM**.

## Documentation used

- DagsHub client setup: https://dagshub.com/docs/client/reference/setup
- DagsHub MLflow tracking: https://dagshub.com/docs/use_cases/track_ml_experiments/
- MLflow TensorFlow autologging: https://mlflow.org/docs/latest/api_reference/python_api/mlflow.tensorflow.html
- TensorFlow install: https://www.tensorflow.org/install/pip
