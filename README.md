# Diabetes Prediction Using Machine Learning
**Author:** Tirthak Likhar

---

## Overview
This project aims to predict whether a patient is likely to develop diabetes based on diagnostic measurements. Using machine learning techniques, this model analyzes features such as glucose level, BMI, insulin levels, and more, to assist in early identification and preventive care.

The project follows the standard ML workflow — including data loading, preprocessing, visualization, model training, evaluation, and comparison — all implemented in Python using `scikit-learn`, `pandas`, and `matplotlib`.

---

## Table of Contents
1. Objective
2. Dataset Description
3. Methodology
4. Exploratory Data Analysis (EDA)
5. Data Preprocessing
6. Model Training
7. Model Evaluation
8. Results and Discussion
9. Setup & Usage
10. Conclusion

---

## 1. Objective
The objective of this project is to build and evaluate machine learning models that can accurately predict the presence of diabetes based on medical and physiological parameters. The goal is to identify key contributing features and benchmark different classification algorithms.

---

## 2. Dataset Description
- **Dataset Name:** `data_diabetes.csv`
- **Source:** Pima Indians Diabetes Dataset
- **Attributes:**
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (0 = Non-Diabetic, 1 = Diabetic)

**Code Snippet:**
```python
import pandas as pd

# Load dataset
df = pd.read_csv('data_diabetes.csv')
print(df.head())

# Dataset info
print(df.info())
```

---

## 3. Methodology
The following workflow was implemented:  
→ Data Collection  
→ Exploratory Data Analysis (EDA)  
→ Data Preprocessing  
→ Model Selection and Training  
→ Model Evaluation and Comparison

**Flowchart:**
```
Dataset → Preprocessing → Model Training → Evaluation → Insights
```

---

## 4. Exploratory Data Analysis (EDA)
The EDA process identifies trends, correlations, and distributions in the dataset. Visualization techniques were used to understand feature relationships.

**Code Snippet:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
```

Key Insights:
- **Glucose** has the strongest correlation with diabetes outcome.
- Higher **BMI** and **Age** are associated with a greater likelihood of diabetes.
- Several features contain zero values that represent missing data.

---

## 5. Data Preprocessing
Zero or missing values were replaced using mean or median imputation. The dataset was then standardized for consistent model performance.

**Code Snippet:**
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Replace zero values in selected columns
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_replace:
    df[col] = df[col].replace(0, df[col].mean())

# Split features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 6. Model Training
Several classification models were trained to identify the most accurate and generalizable algorithm for predicting diabetes.

**Models Implemented:**
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine (SVM)

**Code Snippet:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC()
}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name} trained successfully.")
```

---

## 7. Model Evaluation
Each model was evaluated using accuracy, precision, recall, F1-score, and confusion matrix.

**Code Snippet:**
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
```

---

## 8. Results and Discussion

| Model                 | Accuracy | Precision | Recall | F1-Score |
|------------------------|----------|------------|--------|-----------|
| Logistic Regression    | 0.78     | 0.76       | 0.73   | 0.74      |
| Decision Tree          | 0.72     | 0.69       | 0.71   | 0.70      |
| Random Forest          | 0.83     | 0.80       | 0.78   | 0.79      |
| Support Vector Machine | 0.81     | 0.79       | 0.76   | 0.77      |

**Observations:**
✓ Random Forest Classifier achieved the highest accuracy (83%) and generalization performance.  
✓ SVM also performed consistently well but required longer training time.  
✓ Logistic Regression provided interpretability and baseline comparison.

---

## 9. Setup & Usage

**Requirements:**
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

**Installation:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**Run Instructions:**
```bash
jupyter notebook Diabetes_Prediction_Code.ipynb
```

---

## 10. Conclusion
The project demonstrates that machine learning algorithms, particularly ensemble methods like Random Forest, can effectively predict diabetes with high accuracy when supported by proper preprocessing and feature engineering.

This study reinforces the importance of data quality, normalization, and model interpretability in healthcare-based prediction systems.

---

## Author
**Tirthak Likhar**  
Department of Computer Science  
(Year 2025)
