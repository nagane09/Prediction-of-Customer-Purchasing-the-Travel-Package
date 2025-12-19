# Prediction of Customer Purchasing the Travel Package

🔗 **Live Demo:** https://prediction-of-customer-purchasing-the-rsab.onrender.com/

Predict whether a customer will purchase a travel package based on historical customer data and purchase trends using machine learning models.

---

## 📌 Project Overview

This project solves a **binary classification problem**: predicting whether a potential customer is likely to purchase a newly introduced travel package based on historical customer attributes and interactions.

The goal is to help travel companies **target high‑potential customers**, optimize **marketing cost**, and improve **conversion rates** by focusing efforts on customers more likely to purchase.

---

## 🧠 Problem Background

Customers have varying preferences and profiles — age, occupation, contact type, city tier, passport status, marital status, number of follow‑ups, and other feature variables can influence purchase likelihood. Instead of contacting customers randomly, this model predicts which customers are most likely to purchase the travel package. :contentReference[oaicite:1]{index=1}


---

## 🛠 Tech Stack

* **Python** – Main programming language
* **Pandas & NumPy** – Data manipulation and numeric computing
* **scikit‑learn** – Modeling, feature preprocessing
* **XGBoost / Gradient Boosting / Random Forest (Ensemble Models)** – Machine learning
* **Streamlit** – Interactive web application deployment
* **Joblib** – Model & preprocessor serialization
* **CSV** – Dataset format for input and output

---

## 🚀 How It Works

1. **Data Loading**
   * Dataset loaded from `Travel.csv`
   * Accessible features include age, contact history, city tier, passport status, marital status, and more. :contentReference[oaicite:2]{index=2}

2. **Data Preprocessing**
   * Handle missing values
   * Encode categorical variables
   * Scale/normalize numerical features

3. **Model Training**
   * Build ML models using ensemble techniques such as Random Forest, XGBoost, Gradient Boosting, etc. :contentReference[oaicite:3]{index=3}

4. **Model Saving**
   * Save the trained model and preprocessing pipeline as `.joblib` files

5. **Prediction Web App**
   * Users can provide input features via a Streamlit UI
   * The app applies preprocessor and model to generate purchase prediction

---

## 📊 Model Performance

*Performance metrics should be presented here once computed (accuracy, precision, recall, F1‑score, AUC).*

For example:


## 📋 XGBoost Classifier
  * Train Accuracy: 0.91
  * Test Accuracy: 0.89
  * Precision: 0.90
  * Recall: 0.88
  * F1‑Score: 0.89

## 💡 Example Use Cases

- **Target High-Potential Customers:** Marketing teams can focus on the most receptive customer segments.  
- **Reduce Marketing Costs:** Helps reduce spend per acquisition by avoiding uninterested customers.  
- **Increase Conversion Rates:** Focus efforts on high-propensity customers for better ROI.  
- **Tailored Offers:** Travel companies can customize packages and promotions based on customer profiles.  

---

## 📈 Business Impact

By predicting customer purchase likelihood, organizations can:

- Execute **focused marketing campaigns** to increase conversion efficiency.  
- **Lower operational costs** by minimizing outreach to unlikely buyers.  
- Replace intuition-based decisions with **data-driven strategies**.  

---

## 🔮 Future Enhancements

- Add **confidence scores** to provide probability estimates with each prediction.  
- Integrate with **live CRM systems** for dynamic, real-time predictions.  
- Include **historical interaction visualizations** to understand customer behavior.  
- Build a **dashboard** to analyze feature importances and model insights.  


---
