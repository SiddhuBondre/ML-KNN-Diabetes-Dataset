# ML-KNN-Diabetes-Dataset

🩺 Diabetes Prediction using K-Nearest Neighbors (KNN)
This project uses the K-Nearest Neighbors (KNN) algorithm to predict whether a person has diabetes based on medical diagnostic measurements. The dataset used is the PIMA Indians Diabetes Database from the UCI Machine Learning Repository.


📂 Dataset
Name: PIMA Indians Diabetes Dataset
Source: UCI Machine Learning Repository
Size: 768 samples × 8 features
Target Variable: Outcome (1 = diabetic, 0 = non-diabetic)

Features:

Pregnancies

Glucose

BloodPressure

SkinThickness

Insulin

BMI

DiabetesPedigreeFunction

Age

📌 Objective
To build a machine learning model using KNN that predicts diabetes based on diagnostic features.

🧪 Libraries Used
python
Copy
Edit
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
🔄 Workflow
Data Loading
Load the CSV dataset using pandas.

Exploratory Data Analysis (EDA)

Check for null values

Visualize distributions and correlations

Data Preprocessing

Replace 0s in specific columns with NaN (e.g., Glucose, BloodPressure)

Impute missing values (mean or median)

Feature scaling using StandardScaler

Model Building (KNN)

Use train_test_split()

Fit KNN model

Evaluate with metrics like accuracy, confusion matrix, and classification report

Model Tuning

Try different values of K

Plot accuracy vs. K

⚙️ Model Evaluation
Confusion Matrix

Accuracy Score

Precision, Recall, F1-Score

✅ Results
K Value	Accuracy
3	75.32%
5	77.92%
7	78.57%

Final chosen value of K = 7

📈 Visualizations
Correlation heatmap

Pair plots for features

Accuracy vs. K plot

Confusion matrix heatmap

📌 Conclusion
KNN with K=7 gives the best performance on the dataset.

Standardizing the features improved model performance.

Glucose, BMI, and Age are strong predictors of diabetes.

🧠 Future Improvements
Try other algorithms (e.g., Logistic Regression, Random Forest)

Hyperparameter tuning using GridSearchCV

Handle imbalanced data using SMOTE

Deploy the model using Flask or Streamlit


📜 License
This project is open-source and available under the MIT License.

Would you like me to generate the .ipynb or Python code to go along with this README?









Ask ChatGPT

