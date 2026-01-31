# Heart Disease Risk Prediction: Logistic Regression

## Project Title
Heart Disease Risk Prediction with Logistic Regression (From Scratch)

## Project Description
This project implements logistic regression from scratch (NumPy) to predict heart disease risk using the Kaggle Heart Disease dataset. It covers EDA, preprocessing, training, evaluation, decision-boundary visualization, L2 regularization, and a guided exploration of deployment on Amazon SageMaker. The notebook is fully executable and documents findings, plots, and metrics.

## Exercise Summary
Implements logistic regression for heart disease prediction: EDA, training/viz, regularization, and SageMaker deployment exploration.

## Dataset Description
Kaggle Heart Disease dataset (303 patients; 14 features). Example feature ranges: Age 29–77, Cholesterol 112–564 mg/dL, Resting BP 94–200 mmHg. Target is binary (1 = disease presence, 0 = absence), with ~55% positive class rate. Downloaded from https://www.kaggle.com/datasets/neurocipher/heartdisease.

## Repository Structure
- Heart_Disease_Prediction.csv — dataset
- heart_disease_risk_prediction.ipynb — main notebook (EDA, model, results)
- README.md — project overview and instructions
- images/ — screenshots for SageMaker evidence

## Getting Started
Follow these steps to run the notebook locally.

### Prerequisites
- Python 3.9+
- Jupyter Notebook or JupyterLab
- Python packages: NumPy, Pandas, Matplotlib

### Installing
1. Create a virtual environment (optional but recommended).
2. Install dependencies:
	- NumPy
	- Pandas
	- Matplotlib
3. Launch Jupyter and open the notebook.

### Running the Notebook
Open heart_disease_risk_prediction.ipynb and run cells in order to:
- Load and clean data
- Train logistic regression (from scratch)
- Plot costs and decision boundaries
- Evaluate metrics on train/test splits
- Apply L2 regularization and compare results

## Results (High-Level)
- Metrics reported for train/test (accuracy, precision, recall, F1).
- Cost vs. iterations plots for convergence.
- Decision boundary plots for multiple feature pairs.
- Regularization comparison (unregularized vs. L2).

## Deployment Evidence (SageMaker)
Include screenshots in images/ and summarize below after completing the SageMaker exploration:
- Training job status screenshot
- Endpoint configuration screenshot
- Inference response screenshot

Example evidence statement:
Model deployed at [endpoint ARN]. Tested input [Age=60, Chol=300, RestingBP=140, MaxHR=150, Oldpeak=2.0, Vessels=1] → Output: Prob=0.68 (high risk).

## Deployment Notes
High-level steps followed in SageMaker:
1. Create notebook instance or Studio workspace.
2. Upload data and notebook.
3. Train model and serialize weights (w, b).
4. Create inference script to load model and return probability.
5. Deploy endpoint and invoke with sample input.

## Built With
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook

## Authors
- Tulio Riaño Sánchez

## Acknowledgments
- Kaggle and the UCI Heart Disease dataset contributors
- World Health Organization statistics for context
- Notebooks explaining logistic regression
