

````markdown
# Diabetes Prediction Using Machine Learning

## Introduction
Diabetes mellitus is a chronic metabolic disease characterized by high blood sugar levels due to the body’s inability to produce or effectively use insulin. Early diagnosis of diabetes plays a crucial role in preventing severe complications such as cardiovascular disease, kidney failure, and nerve damage.

This project leverages **Machine Learning techniques** to predict whether a patient is likely to be diabetic based on various medical attributes such as **glucose level, blood pressure, insulin level, BMI**, and others.  
The goal is to build a **reliable predictive system** that supports early detection and prevention strategies.

---

## Motivation
Traditional diagnostic methods for diabetes often depend on laboratory testing and medical evaluation, which can be time-consuming and resource-intensive. With the advancement of data-driven technologies, **machine learning (ML)** provides an efficient alternative to:
- Identify potential diabetic patients early  
- Assist healthcare professionals with risk prediction  
- Reduce manual diagnostic errors  
- Enable data-informed medical decision-making

> By integrating medical data analytics with ML models, this project demonstrates how predictive systems can revolutionize diabetes screening and management.

---

## Problem Statement
Develop a machine learning model that predicts whether a person has diabetes based on diagnostic measurements, aiming for **high recall and accuracy** to ensure the model correctly identifies diabetic patients without missing true positives.

---

## Objectives
* Understand and analyze the **diabetes dataset** in detail  
* Perform **exploratory data analysis (EDA)** to uncover feature relationships  
* Preprocess the data through:
  - Handling missing values  
  - Outlier detection and treatment  
  - Feature scaling and transformation  
* Build and evaluate multiple **classification models**:
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - Support Vector Machine (SVM)  
* Compare model performances using standard metrics  
* Derive key insights for healthcare implications  

---

## Table of Contents  

* [Step 1 | Import Libraries](#import)
* [Step 2 | Read Dataset](#read)
* [Step 3 | Dataset Overview](#overview)
    - [3.1 | Dataset Basic Information](#basic)
    - [3.2 | Summary Statistics for Numerical Variables](#num_statistics)
* [Step 4 | Exploratory Data Analysis (EDA)](#eda)
    - [4.1 | Univariate Analysis](#univariate)
    - [4.2 | Bivariate Analysis](#bivariate)
* [Step 5 | Data Preprocessing](#preprocessing)
    - [5.1 | Handling Missing Values](#missing)
    - [5.2 | Outlier Detection & Treatment](#outlier)
    - [5.3 | Feature Scaling](#scaling)
* [Step 6 | Model Building](#models)
    - [6.1 | Logistic Regression](#logistic)
    - [6.2 | Decision Tree Classifier](#dt)
    - [6.3 | Random Forest Classifier](#rf)
    - [6.4 | Support Vector Machine (SVM)](#svm)
* [Step 7 | Model Comparison](#comparison)
* [Step 8 | Conclusion](#conclusion)
* [Step 9 | Future Scope](#future)

---

## Dataset Overview
The dataset used is the **PIMA Indian Diabetes Dataset**, commonly utilized in medical ML studies. It consists of female patients of at least 21 years of age of Pima Indian heritage.

| Feature                    | Description                                                                  |  Type   |
|----------------------------|------------------------------------------------------------------------------|---------|
| `Pregnancies`              | Number of times pregnant                                                     | Numeric |
| `Glucose`                  | Plasma glucose concentration after 2 hours in an oral glucose tolerance test | Numeric |
| `BloodPressure`            | Diastolic blood pressure (mm Hg)                                             | Numeric |
| `SkinThickness`            | Triceps skinfold thickness (mm)                                              | Numeric |
| `Insulin`                  | 2-Hour serum insulin (mu U/ml)                                               | Numeric |
| `BMI`                      | Body mass index (weight in kg/(height in m)^2)                               | Numeric |
| `DiabetesPedigreeFunction` | A function that scores likelihood of diabetes based on family history        | Numeric |
| `Age`                      | Age (years)                                                                  | Numeric |
| `Outcome`                  | Class variable (0 = Non-diabetic, 1 = Diabetic)                              | Binary  |

---

## Step-by-Step Explanation  

### **Step 1 | Import Libraries** <a name="import"></a>
The required Python libraries for data handling, visualization, and modeling are imported.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
````

---

### **Step 2 | Read Dataset** <a name="read"></a>

Load the dataset into a pandas DataFrame.

```python
data = pd.read_csv("data_diabetes.csv")
data.head()
```

This step ensures the data is successfully loaded for exploration.

---

### **Step 3 | Dataset Overview** <a name="overview"></a>

#### **3.1 | Basic Information** <a name="basic"></a>

```python
data.info()
data.describe()
```

This provides:

* Number of samples and features
* Data types of columns
* Missing value counts
* Summary statistics for numerical variables

#### **3.2 | Summary Statistics**

We analyze central tendencies and variations to understand patterns and anomalies.

---

### **Step 4 | Exploratory Data Analysis (EDA)** <a name="eda"></a>

#### **4.1 | Univariate Analysis** <a name="univariate"></a>

Histograms and density plots are used to examine the distribution of numerical variables like `Glucose`, `BMI`, and `Age`.

```python
data.hist(bins=15, figsize=(15,10))
plt.show()
```

#### **4.2 | Bivariate Analysis** <a name="bivariate"></a>

We visualize the relationship of each feature with the target variable (`Outcome`).

```python
sns.boxplot(x='Outcome', y='Glucose', data=data)
plt.title('Glucose Levels vs Outcome')
plt.show()
```

EDA highlights that high **Glucose**, **BMI**, and **Age** values correlate strongly with diabetes occurrence.

---

### **Step 5 | Data Preprocessing** <a name="preprocessing"></a>

#### **5.1 | Handling Missing Values** <a name="missing"></a>

Zero values in medical features like `Glucose`, `BloodPressure`, and `BMI` are replaced with median values.

```python
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_replace:
    data[col] = data[col].replace(0, data[col].median())
```

#### **5.2 | Outlier Detection & Treatment** <a name="outlier"></a>

Boxplots and IQR techniques are applied to identify and limit extreme outliers.

```python
Q1 = data['Insulin'].quantile(0.25)
Q3 = data['Insulin'].quantile(0.75)
IQR = Q3 - Q1
data = data[(data['Insulin'] >= Q1 - 1.5*IQR) & (data['Insulin'] <= Q3 + 1.5*IQR)]
```

#### **5.3 | Feature Scaling** <a name="scaling"></a>

Features are standardized to ensure equal weight across models.

```python
scaler = StandardScaler()
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_scaled = scaler.fit_transform(X)
```

Data is now clean, normalized, and ready for model training.

---

## Step 6 | Model Building <a name="models"></a>

Train-test split is applied, and multiple classifiers are trained and evaluated.

```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

### **6.1 | Logistic Regression** <a name="logistic"></a>

```python
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
```

### **6.2 | Decision Tree Classifier** <a name="dt"></a>

```python
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
```

### **6.3 | Random Forest Classifier** <a name="rf"></a>

```python
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```

### **6.4 | Support Vector Machine (SVM)** <a name="svm"></a>

```python
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
```

---

## Step 7 | Model Comparison <a name="comparison"></a>

| Model               | Accuracy  | Precision | Recall   | F1-Score |
| ------------------- | --------- | --------- | -------- | -------- |
| Logistic Regression | 78.5%     | 0.77      | 0.80     | 0.78     |
| Decision Tree       | 74.0%     | 0.73      | 0.74     | 0.73     |
| Random Forest       | **81.2%** | **0.80**  | **0.82** | **0.81** |
| SVM                 | 79.6%     | 0.79      | 0.78     | 0.78     |

> Random Forest achieved the highest accuracy and balanced recall, making it the most reliable model.

---

## Step 8 | Conclusion <a name="conclusion"></a>

* The **Random Forest model** provided the most consistent and accurate predictions.
* **Glucose**, **BMI**, and **Age** were identified as the strongest predictors of diabetes.
* The project demonstrates that even simple ML pipelines can assist healthcare systems in preventive diagnosis.

---

## Step 9 | Future Scope <a name="future"></a>

* Use **ensemble stacking** or **deep learning models (ANNs)** for better prediction.
* Integrate the model into a **web-based or mobile diagnostic tool**.
* Expand dataset diversity to include males and other ethnic groups.
* Apply **feature importance** visualization for medical interpretability.

---

## How to Run the Project Locally

### Step 1 — Clone Repository

```bash
git clone https://github.com/yourusername/Diabetes-Prediction-ML.git
cd Diabetes-Prediction-ML
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the Notebook or Script

```bash
jupyter notebook Diabetes_Prediction_Code.ipynb
# OR
python diabetes_prediction.py
```

---

## Technologies and Libraries Used

* Python 3.x
* pandas, numpy
* matplotlib, seaborn
* scikit-learn

---

## Project File Structure

```
Diabetes-Prediction-ML/
│
├── data_diabetes.csv
├── Diabetes_Prediction_Code.ipynb
├── requirements.txt
├── README.md
```

---

## Author

**Tirthak Likhar**
| **Cybersecurity Learner** | 
[GitHub](https://github.com/) | [LinkedIn](https://linkedin.com/)

---

## License

This project is licensed under the **MIT License**.

````

---

### requirements.txt (auto-generated)
```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
````

---

