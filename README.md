Objective

The goal of this assignment is to gain hands-on experience with essential MLOps practices.
This includes setting up Python environments, managing dependencies, using Git and GitHub branching workflows, training machine learning models with scikit-learn, and automating experiments through Continuous Integration (CI) using GitHub Actions.

Tasks

Environment Setup

Create and activate a virtual environment.

Install dependencies (numpy, pandas, scikit-learn, joblib).

Save them in a requirements.txt file.

GitHub Repository

Initialize a new Git repository locally.

Create a public GitHub repository.

Push all changes using Git CLI (not via GitHub web upload).

Branch dtree

Create a new branch named dtree.

Implement misc.py containing helper functions (data loading, preprocessing, splitting, training, evaluation).

Implement train.py to train a DecisionTreeRegressor.

Commit and push changes, then merge dtree into main.

Branch kernelridge

Create a new branch named kernelridge.

Implement train2.py to train a Kernel Ridge Regressor.

Add GitHub Actions workflow (.github/workflows/ci.yml) to:

Install dependencies

Run train.py

Run train2.py

Push changes to kernelridge.

Continuous Integration

On push to kernelridge, the GitHub Actions workflow should automatically run.

Logs should show model training output including per-run and average MSE values.
