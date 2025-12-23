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

----

## 🧠 Mathematical Intuition (XGBoost)

XGBoost (Extreme Gradient Boosting) is an **ensemble learning algorithm** that builds a strong classifier by **sequentially adding decision trees**, where each new tree corrects the mistakes made by the previous ones.

---

### 📌 Model Objective Function

At each boosting iteration \( t \), XGBoost minimizes the following objective:

\[
\mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t)}) + \sum_{k=1}^{t} \Omega(f_k)
\]

Where:
- \( l(y_i, \hat{y}_i) \) is the **loss function** (log loss for binary classification)
- \( \hat{y}_i^{(t)} \) is the prediction at iteration \( t \)
- \( f_k \) represents the decision tree added at iteration \( k \)
- \( \Omega(f_k) \) is the **regularization term**

---

### 📉 Regularization Term

XGBoost controls model complexity using:

\[
\Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2
\]

Where:
- \( T \) = number of leaves in the tree  
- \( w_j \) = leaf weight  
- \( \gamma \) penalizes adding new leaves  
- \( \lambda \) applies L2 regularization on leaf weights  

This prevents **overfitting**, even when training accuracy is very high.

---

### 🔁 Gradient Boosting Mechanism

Instead of fitting directly to the target, XGBoost fits each tree to the **negative gradient of the loss function**:

\[
g_i = \frac{\partial l(y_i, \hat{y}_i)}{\partial \hat{y}_i}, \quad
h_i = \frac{\partial^2 l(y_i, \hat{y}_i)}{\partial \hat{y}_i^2}
\]

Where:
- \( g_i \) = first-order gradient  
- \( h_i \) = second-order Hessian  

Using both gradients allows **faster convergence** and **more stable optimization**.

---

### 🌲 Optimal Split Criterion

For each split, XGBoost computes the **gain**:

\[
Gain = \frac{1}{2}
\left[
\frac{G_L^2}{H_L + \lambda}
+
\frac{G_R^2}{H_R + \lambda}
-
\frac{(G_L + G_R)^2}{(H_L + H_R + \lambda)}
\right]
- \gamma
\]

Where:
- \( G_L, G_R \) = sum of gradients for left and right nodes  
- \( H_L, H_R \) = sum of Hessians  

Splits are chosen only if they **increase the objective function**, enforcing model simplicity.

---

### 🎯 Probability Estimation

The final model prediction is:

\[
\hat{y} = \sigma\left(\sum_{t=1}^{T} f_t(x)\right)
\]

Where:
- \( \sigma(\cdot) \) is the **sigmoid function**
- Output represents the **probability of a customer purchasing a holiday package**

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
