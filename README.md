# 🧠 Customer Churn Prediction Using Artificial Neural Networks (ANN)

## 📌 Project Overview

Customer churn is a major challenge in the banking sector. This project uses an **Artificial Neural Network (ANN)** to predict whether a customer is likely to leave the bank based on their **behavioral and financial data**.

The model helps banks **identify at-risk customers** early and take proactive steps to retain them.

---

## 🎯 Objective

To build a **binary classification model** using ANN that predicts:

* **1** → Customer exits the bank
* **0** → Customer stays with the bank

---

## 📂 Dataset Information

* **Dataset Name:** `Churn_Modelling (AI).csv`
* **Total Records:** 10,000 customers
* **Total Features:** 14
* **Target Variable:** `Exited`

### Key Features:

* Geography
* Gender
* Age
* Balance
* Credit Score
* Estimated Salary
* Number of Products
* Is Active Member

---

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Libraries:**

  * NumPy
  * Pandas
  * Matplotlib / Seaborn
  * Scikit-learn
  * TensorFlow / Keras

---

## 🔄 Project Workflow

### 1️⃣ Data Cleaning

* Removed null values (if any)
* Removed duplicate records

### 2️⃣ Feature Engineering

* Dropped irrelevant columns:

  * `RowNumber`
  * `CustomerId`
  * `Surname`
* Applied **Label Encoding** on `Gender`
* Applied **One-Hot Encoding** on `Geography`

### 3️⃣ Train-Test Split

* 80% Training data
* 20% Testing data
* Feature scaling using **StandardScaler**

---

## 🧠 ANN Model Architecture

* **Input Layer:** 16 neurons (ReLU activation)
* **Hidden Layer:** 8 neurons (ReLU activation)
* **Output Layer:** 1 neuron (Sigmoid activation)

✔ Sigmoid is used for **binary classification**

---

## 🚀 Model Training

* **Optimizer:** Adam
* **Loss Function:** Binary Crossentropy
* **Epochs:** 50

---

## 📊 Model Evaluation

### 🔹 Accuracy

* **Overall Accuracy:** **85%**

### 🔹 Classification Report

| Metric    | Class 0 (Stayed) | Class 1 (Exited) |
| --------- | ---------------- | ---------------- |
| Precision | 0.88             | 0.62             |
| Recall    | 0.92             | 0.50             |
| F1-Score  | 0.90             | 0.55             |

---

## 📈 Results & Insights

* The model performs very well in identifying **customers who stay**
* Reasonable performance in predicting **customers likely to exit**
* Suitable for **real-world churn prediction use cases**

---

## ✅ Conclusion

This project demonstrates how **Artificial Neural Networks** can be effectively used for **customer churn prediction** in the banking domain.
With an accuracy of **85%**, the model can help organizations:

* Reduce customer loss
* Improve retention strategies
* Increase long-term profitability

---

## 🔮 Future Improvements

* Handle class imbalance using SMOTE
* Try deeper neural networks
* Hyperparameter tuning
* Deploy model using Flask or FastAPI

---

## 👨‍💻 Author

**Armaan S**
Computer Science Engineer
📌 AI | ML | Data Analysis

