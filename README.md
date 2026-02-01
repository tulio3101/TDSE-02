# Heart Disease Risk Prediction: Logistic Regression

## Project Title
Heart Disease Risk Prediction with Logistic Regression (From Scratch)

## Project Description
This project implements logistic regression from scratch (NumPy) to predict heart disease risk using the Kaggle Heart Disease dataset. It covers EDA, preprocessing, training, evaluation, decision-boundary visualization, L2 regularization, and a guided exploration of deployment on Amazon SageMaker. The notebook is fully executable and documents findings, plots, and metrics.

## Exercise Summary
Implements logistic regression for heart disease prediction: EDA, training/viz, regularization, and SageMaker deployment exploration.

## Dataset Description

In this dataset provided by **Kaggle**, we can find features that can influence wether a person has heart disease; these are:

| Column Name  | Second Header |
| ------------- | ------------- | 
| 🧓 Age  | Age of the patient in years  | 
| 🚹 Sex  | Gender of the patient (**1 = Male, 0 = Female**)  | 
| 💔 Chest pain type | **1** = Typical Angina - **2** = Atypical Angina - **3** = Non-Anginal Pain - **4** = Asymptomatic  | 
| 💉 BP | Resting blood pressure (mm Hg)  | 
| 🧈 Cholesterol | Serum cholesterol level (mg/dL)  | 
| 🍬 FBS over 120 | Fasting blood sugar > 120 mg/dL (**1 = True, 0 = False**)  | 
| 📈 EKG results  | ❤️ **0** = Normal - ⚠️ **1** = ST-T wave abnormality - 💥 **2** = Left ventricular hypertrophy | 
| ❤️ Max HR  | Maximum heart rate achieved  | 
| 🏃 Exercise angina | Exercise-induced angina (**1 = Yes, 0 = No**)  | 
| 📉 ST depression | ST depression induced by exercise relative to rest  | 
| ⛰️ Slope of ST | Slope of the peak exercise ST segment  | 
| 🩸 Number of vessels fluro | Number of major vessels (0–3) colored by fluoroscopy  | 
| 🧬 Thallium | Thallium stress test result (categorical medical indicator)  | 
| 🎯 Heart Disease | ❤️ **Presence** = Heart disease detected - 💚 **Absence** = No heart disease  | 

Taking the above into account, the v alues associated with each one will not necessarily be between 0 and 1, so it was neccesary to normalize these data using the **average** and the **standard deviation**

## Repository Structure
- Heart_Disease_Prediction.csv — dataset
- heart_disease_lr_analysis.ipynb — main notebook (EDA, model, results)
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
Open heart_disease_lr_analysis.ipynb and run cells in order to:
- Load and clean data
- Train logistic regression (from scratch)
- Plot costs and decision boundaries
- Evaluate metrics on train/test splits
- Apply L2 regularization and compare results

### Exploratory Data Analysis (EDA)

To perform this procedure, we use function provided by pandas such as **info()** and **describe()**. From these, we can determine if there are missing values in the dataset or what the average, minimum, or maximum values of a given characteristic is.

Similarly, we can use a **whisker box** with **Matplot** to identify outliers and consider normalize them.

We can also determine which characteristics are most influential in predicting whether a person has **heart disease**.

For Example, **ST Depression**, according to some studies, can be highly significan in indicating myocardial damage if it is the only unusual finding or symptom. **(Flores, G. (2025))**

## Results (High-Level)
- Metrics reported for train/test (accuracy, precision, recall, F1).


- Cost vs. iterations plots for convergence.

By graphing this type of function for the **complete model**, i.e., > 6 features, we can see how it converges depending on the appropriate values we give to **alpha** and **number of iterations** respectively.

![alt text](docs/images/costVsIterations.png)

Here we can see how it tends towards convergence when it exceeds **1000** iterations with the **training dataset**.

- Decision boundary plots for multiple feature pairs.

The objective of retraining, but choosing only pairs of features, is to see how separable the class is. In some cases, we saw that good separation can be obtained through the decision limit, and in other cases, it cannot.

- **Age vs Cholesterol**: Hard to separate linearly.

- **BP vs Max HR**: Shows clearer separation but with some overlap.

![alt text](docs/images/ageCholesterol.png)

---

![alt text](docs/images/bpMax.png)

- Regularization comparison (unregularized vs. L2).

When we add the regularization term, which aims to penalyze very large weights, we can obtain the best result for the model. This can be seen by graphing **cost vs iterations** again with a respective **lambda**.

![alt text](docs/images/costVsIterationsReg.png)

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
- Notebooks explaining logistic regression
- Flores, G. (2025). **La depresión del ST en aVL como clave diagnóstica: un caso que lo cambia todo**. *ECC Trainings*. [https://ecctrainings.com/...](https://ecctrainings.com/la-depresion-del-st-en-avl-como-clave-diagnostica-un-caso-que-lo-cambia-todo/)
