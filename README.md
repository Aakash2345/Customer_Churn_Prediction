# 🧠 Customer Churn Prediction App

This Streamlit app predicts whether a customer is likely to **churn (leave)** or **stay** based on their behavior and demographics.  
It helps businesses identify at-risk customers and take proactive retention actions.

👉 **Try it here:** [Open Streamlit App](https://customer-churn-prevention-project.streamlit.app)

---

# 🧩 Customer Churn Prediction using Machine Learning

This project predicts customer churn — whether a customer is likely to discontinue a service — based on behavioral, demographic, and billing data. The aim is to help businesses identify at-risk customers early

---

## 🎯 Objective

Build and evaluate machine learning models that can accurately predict customer churn using telecom service data.

---

## 📊 Dataset

* **Source:** Telco Customer Churn dataset
* **Size:** ~7,000 customer records
* **Features:**

  * Customer demographics (gender, senior citizen, partner, dependents)
  * Service details (Internet, phone, contract, payment method)
  * Billing information (tenure, monthly charges, total charges)

---

## 🧹 Data Preprocessing

* Handled **missing values** in the `TotalCharges` column by replacing blank spaces and converting to float
* Converted categorical features using **LabelEncoder**, **OrdinalEncoder**, and **OneHotEncoder**
* Applied **StandardScaler** for numerical feature scaling
* Split the dataset into **training** and **testing** sets

---

## 🔍 Exploratory Data Analysis (EDA)

* Visualized distributions of categorical features such as gender, contract type, and payment method
* Explored relationships between **tenure**, **monthly charges**, and **churn**
* Found that customers with **shorter tenure** and **higher monthly charges** are more likely to churn

---

## ⚙️ Modeling Approach

Trained and compared multiple machine learning models:

1. **Logistic Regression**
2. **Random Forest Classifier**
3. **XGBoost Classifier**

### Model Optimization

* Applied **Hyperparameter Tuning** using `RandomizedSearchCV` to improve model performance
* Compared models using Accuracy, Precision, Recall, and ROC-AUC Score

The **XGBoost model** achieved the highest ROC-AUC score and was selected as the final model.

---

## 📈 Results & Insights

* **ROC-AUC Score:** ~0.88
* **Top Churn Drivers:**

  * High monthly charges
  * Short tenure period
  * Electronic check payment method
  * Fiber optic internet service

**Recommendations:**

* Offer loyalty incentives for new customers to increase retention
* Introduce discounts for high-bill customers
* Target electronic check users with alternative payment methods

---

## 🧠 Tech Stack

Python • Pandas • NumPy • Scikit-learn • XGBoost • Matplotlib • Seaborn

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

# Install dependencies
pip install -r requirements.txt

# Run the notebook or script
jupyter notebook Customer_Churn_pred.ipynb
# or
python customer_churn_prediction.py
```


