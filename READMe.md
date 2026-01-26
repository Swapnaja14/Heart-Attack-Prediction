# ❤️ Heart Attack Prediction Using Logistic Regression

## 📌 Project Overview
This project predicts the **possibility of a heart attack** using a **Logistic Regression** classification model.  
It uses patient health data from the **Cleveland Heart Disease dataset** to classify whether a person has a **lower or higher risk of heart attack**.

The project focuses on:
- Data loading and preprocessing
- Training a Logistic Regression model
- Evaluating performance using **Accuracy** and **Precision**

---

## 🎯 Objective
To build a machine learning model that can:
- Analyze medical attributes of patients
- Predict heart attack possibility (`0` or `1`)
- Provide a simple and interpretable healthcare prediction system

---

## 📊 Dataset Information
- **Dataset:** Cleveland Heart Disease Dataset
- **File Used:** `heart.csv`
- **Target Column:** `target`
  - `0` → Less chance of heart attack  
  - `1` → More chance of heart attack

---

## 🧬 Features Used
The model is trained using the following medical attributes:

| Feature | Description |
|------|------------|
| age | Age of the patient |
| sex | Gender (1 = male, 0 = female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels |
| thal | Thalassemia type |

---

## 🛠️ Technologies Used
- **Python**
- **Pandas** – data handling
- **Matplotlib & Seaborn** – data visualization
- **Scikit-learn** – machine learning

---

## 🧠 Machine Learning Algorithm
### Logistic Regression
- Used for **binary classification**
- Outputs class predictions (`0` or `1`)
- Simple, efficient, and interpretable for healthcare data

```python
model = LogisticRegression(max_iter=1300)
model.fit(X_train, y_train)
```
---

## 🔄 Workflow

- Load dataset using Pandas
- Separate features (X) and target (y)
- Split data into training and testing sets (80% / 20%)
- Train Logistic Regression model
- Make predictions on test data
- Evaluate model performance

---
## 📈 Model Evaluation

The model is evaluated using Accuracy and Precision metrics.

```python
print("accuracy: ", accuracy_score(y_test, y_pred) * 100, "%")
print("precision: ", precision_score(y_test, y_pred) * 100, "%")
```

### 🔹 Metrics Explained

Accuracy: Measures overall correctness of predictions

Precision: Measures how many predicted positive cases are actually positive
(important in medical diagnosis)

---
## ✅ Results

The Logistic Regression model successfully predicts heart attack possibility.

The model achieves good accuracy and precision on unseen test data.

Suitable as a baseline healthcare classification model.

---