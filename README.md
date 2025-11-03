# Diabetes risk prediction

A simple, visual, and interactive web app that explores how everyday health factors influence diabetes risk — powered by machine learning.

## Problem statement
Diabetes is a condition that happens when your blood sugar (glucose) is too high. It develops when your pancreas doesn’t make enough insulin or any at all, or when your body isn’t responding to the effects of insulin properly.

Diabetes affects people of all ages.

<img src="https://github.com/user-attachments/assets/7c741f49-17f6-4999-8380-74215dc52b92" alt="Understanding Blood Sugar Ranges_ What Your Glucose Level Really Means" height="400" width="700"/>

This project focuses on building a highly sensitive classification model to predict the onset of diabetes based on diagnostic measurements from the Pima Indians Diabetes Dataset. Given the medical nature of the problem, the primary goal was to maximize Recall (Sensitivity) to minimize the risk of a false-negative diagnosis.

## 🧬 Dataset
* Source: Pima Indians Diabetes Database (Kaggle)
* Samples: 768
* Features: 8 independent variables (Glucose, BMI, Age, Insulin, etc.)
* Target: Outcome — 1 indicates diabetes, 0 indicates no diabetes

## 💻 Dependencies
This project requires Python 3.x and the following libraries:
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn

## 🧠 Process

### 1. Data Loading and Initial Cleaning 
Load Data: The project begins by loading the raw Pima Indians Diabetes dataset into a Pandas DataFrame.

### 2. EDA and Preprocessing
Visualizations like Histograms,Heatmaps and Scatter Plots were used to identify data quality issues (zeros), check distributions, and confirm complex feature interactions (like BMI and Age) before engineering new variables.

This phase transforms the raw data into more predictive features:
* Imputation: Critical zero values in columns like Glucose, BMI, BloodPressure, and Insulin are treated as missing data and are replaced with the median of the respective column. This corrects major data quality issues without discarding valuable rows.

* A key feature, BMI_Age_Interaction (BMI×Age), is created to capture the synergistic effect of both high BMI and advanced age on diabetes risk.

* Ratio Feature: The Insulin_Glucose_Ratio is calculated to model metabolic efficiency.

* Binning and Encoding: Continuous features (Age, BMI) are converted into categorical groups (e.g., 'Senior', 'Obese') using pd.cut(). These new categorical features are then converted into numerical dummy variables using One-Hot Encoding (pd.get_dummies()).

### 3. Data Splitting and Scaling 
This phase prepares the data for modeling while preventing leakage:

* Splitting: The feature set (X) and the target (y) are separated. The data is then split into X train, X test, y train, and y test using train_test_split with stratify=y to ensure an equal proportion of diabetic and non-diabetic cases in both sets.

* Scaling: The StandardScaler is applied to only the continuous, high-range numeric columns (Pregnancies, Glucose, etc.).

* Fit and Transform: scaler.fit_transform() is used on the training data to learn the mean and standard deviation (μ,σ) and scaler.transform() is used on the test data.
  
### 4. Model Training
Multiple models are trained on the prepared X train

* Model Selection: Logistic Regression, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN) are chosen for comparison.

* Imbalance Handling: The class_weight='balanced' parameter is used for Logistic Regression and SVM. This tuning step is crucial, as it forces the models to prioritize the minority (diabetic) class, directly boosting the Recall metric.

* Training: Each model is fitted to the scaled training data.

### 5. Evaluation and Conclusion ✅
* Prediction: Each trained model makes predictions (y_pred) on the completely unseen X_test data.

* Metric Calculation: Key classification metrics are calculated by comparing y_pred against the true y_test values, with a focus on Recall (Sensitivity).

* Final Selection: The Logistic Regression model, which achieved the highest Recall (nearly 80%), is selected as the optimal predictor because it provides the best diagnostic sensitivity (lowest rate of missed diagnoses). The process concludes with a presentation of the results in a clear format (a Pandas DataFrame) and a concluding summary.

##  📊 Key Results
The model evaluation focused on Recall to ensure high diagnostic sensitivity.

<img width="453" height="170" alt="image" src="https://github.com/user-attachments/assets/d7436d22-73a9-4a34-a108-32fd0cf18be1" />


## Conclusion: 
The Tuned Logistic Regression model was selected as the final predictor due to its superior Recall score (79.6%), making it the most reliable tool for preliminary diabetes screening.


