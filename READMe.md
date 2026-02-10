# ❤️ Heart Attack Prediction Using Logistic Regression and Gaussian Naive Bayes

## 📌 Project Overview
This project predicts the **possibility of a heart attack** using classification models **Logistic Regression** and **Naive Bayes**.  
It uses patient health data from the **Cleveland Heart Disease dataset** to classify whether a person has a **lower or higher risk of heart attack**.

The project focuses on:
- Data loading and preprocessing
- Training multiple classification models
- Evaluating performance using **Accuracy**, **Precision** and **Recall**

---

## 🎯 Objective
To build and compare machine learning models that can:
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

## 🧠 Machine Learning Models Used
### Logistic Regression
- Used for **binary classification**
- Outputs class predictions (`0` or `1`)
- Simple, efficient, and interpretable for healthcare data

```python
model = LogisticRegression(max_iter=1300)
model.fit(X_train, y_train)
```

### Gaussian Naive Bayes
- Based on **Bayes Theorem**
- Assumes features follow a **Gaussian distribution**
- Fast and effective for probabilistic classification

```python
gnb_model = GaussianNB()
gnb_model.fit(X_train, y_train)
```
---

## 🔄 Workflow

- Load dataset using Pandas
- Separate features (X) and target (y)
- Split data into training and testing sets (80% / 20%)
- Train Logistic Regression and Gaussian Naive Bayes models
- Make predictions on test data
- Evaluate and compare model performance

---
## 📈 Model Evaluation

The model is evaluated using Accuracy, Precision and Recall metrics.

```python
print("accuracy: ", accuracy_score(y_test, y_pred) * 100, "%")
print("precision: ", precision_score(y_test, y_pred) * 100, "%")
```

```python
print("recall score: ", recall_score(y_test, y_pred))
print("precision: ", precision_score(y_test, y_pred))
print("accuracy_score: ", accuracy_score(y_test, y_pred))
```

### 🔹 Metrics Explained

Accuracy: Measures overall correctness of predictions

Precision: Measures how many predicted positive cases are actually positive
(important in medical diagnosis)

Recall: Proportion of actual positive cases correctly identified
📌 Recall is especially important in healthcare to minimize false negatives.

---
## ✅ Results and Comparison

Both Logistic Regression and Gaussian Naive Bayes models were evaluated on the test dataset using standard classification metrics.

🔹 Logistic Regression Performance

Accuracy: 0.8525
Precision: 0.8710
Recall: 0.8438

🔹 Gaussian Naive Bayes Performance

Accuracy: 0.8689
Precision: 0.9000
Recall: 0.8438

### 📈 Model Comparison Summary
| Metric    | Logistic Regression | Gaussian Naive Bayes |
| --------- | ------------------- | -------------------- |
| Accuracy  | 0.8525              | **0.8689**           |
| Precision | 0.8710              | **0.9000**           |
| Recall    | **0.8438**          | **0.8438**           |

### Interpretation of Results

- Gaussian Naive Bayes achieves higher accuracy and precision, indicating fewer false positive predictions.

- Both models show identical recall, meaning they are equally effective at identifying actual heart attack cases.

- Logistic Regression offers better interpretability, while Gaussian Naive Bayes provides slightly better predictive performance.

📌 Overall, Gaussian Naive Bayes performs marginally better on this dataset, while Logistic Regression remains a strong and interpretable baseline.

---