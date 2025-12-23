# Prediction of Customer Purchasing the Travel Package

🔗 **Live Demo:** https://prediction-of-customer-purchasing-the-rsab.onrender.com/

Predict whether a customer will purchase a travel package based on historical customer data and purchase trends using machine learning models.

---
# 🏖️ Holiday Package Purchase Prediction

## 📌 Project Overview

This project predicts whether a customer will **purchase a holiday travel package** based on demographic, behavioral, and interaction-based features.

The solution helps travel companies:
- Identify **high-potential customers**
- Optimize **targeted marketing**
- Reduce **customer acquisition cost**
- Increase **conversion rate**

This is a **binary classification problem**:
- `1` → Customer purchased the package  
- `0` → Customer did not purchase the package  

---

## 📊 Dataset Information

- **Source**: Kaggle  
- **Dataset Name**: Holiday Package Purchase Prediction  
- **Total Records**: ~4,800  
- **Target Variable**: `ProdTaken`

### 📁 Feature Description

| Feature | Description |
|------|------------|
| Age | Customer age |
| TypeofContact | How customer was contacted |
| CityTier | City classification (1–3) |
| DurationOfPitch | Sales pitch duration |
| Occupation | Customer occupation |
| Gender | Customer gender |
| ProductPitched | Package offered |
| PreferredPropertyStar | Hotel rating preference |
| MaritalStatus | Marital status |
| NumberOfTrips | Previous trips |
| Passport | Passport availability |
| PitchSatisfactionScore | Pitch satisfaction |
| OwnCar | Car ownership |
| MonthlyIncome | Monthly income |
| TotalVisiting | Total people traveling |

---

## 🧹 Data Cleaning & Preprocessing

### ✔️ Missing Value Treatment

| Feature | Strategy |
|------|---------|
| Age | Median |
| DurationOfPitch | Median |
| MonthlyIncome | Median |
| TypeofContact | Mode |
| NumberOfTrips | Median |
| PreferredPropertyStar | Mode |
| NumberOfChildrenVisiting | Mode |

### ✔️ Data Corrections
- `Fe Male` → `Female`
- `Single` → `Unmarried`
- Dropped `CustomerID` (identifier)

---

## 🧠 Feature Engineering

### ➕ New Feature
```text
TotalVisiting = NumberOfPersonVisiting + NumberOfChildrenVisiting
```

## 🔄 Data Transformation

- **Categorical Features** → One-Hot Encoding  
- **Numerical Features** → Standard Scaling  
- Implemented using **ColumnTransformer** to ensure a clean and reusable ML pipeline.

---

## 🔀 Train-Test Split

- **Training Set**: 80%  
- **Testing Set**: 20%  
- **Random State**: 42  

---

## 🤖 Model Selection

Multiple classification models were initially experimented with.  
Based on performance, generalization ability, and business relevance, **XGBoost** was selected as the **final model**.

---

## 🏆 Final Model: XGBoost Classifier

XGBoost delivered the **best balance between accuracy, recall, and ROC-AUC**, making it suitable for identifying customers most likely to purchase a holiday package.

### 🔥 Performance Before Hyperparameter Tuning

| Metric | Train | Test |
|------|------|------|
| Accuracy | 99.92% | 93.56% |
| F1 Score | 99.92% | 93.18% |
| Recall | 99.59% | 70.68% |
| ROC-AUC | 0.99 | 0.84 |

---

## ⚙️ Hyperparameter Tuning

Hyperparameter tuning was performed to reduce overfitting and improve generalization.

### Best XGBoost Parameters
```json
{
  "n_estimators": 200,
  "max_depth": 8,
  "learning_rate": 0.1,
  "colsample_bytree": 0.8
}
```
## 🥇 Final Model Performance (Tuned XGBoost)

| Metric   | Train | Test |
|---------|-------|------|
| Accuracy | 100%  | 94.99% |
| F1 Score | 100%  | 94.75% |
| Recall   | 100%  | ~75% |
| ROC-AUC  | 1.00  | ~0.85 |

✅ The tuned **XGBoost** model demonstrates **strong predictive performance** while maintaining **good generalization** on unseen data.

---

### 📊 Why This Works Well for This Problem

- Captures **non-linear relationships** between customer attributes  
- Handles **feature interactions** automatically  
- Regularization controls overfitting despite high training accuracy  
- Gradient-based optimization improves **recall for minority class**

---

### ✅ Summary

XGBoost combines:
- Gradient descent optimization  
- Tree-based feature learning  
- Explicit regularization  

This results in a **highly accurate, well-generalized model** for predicting holiday package purchases.


---
