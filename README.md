# 🇱🇰 Sri Lanka Telecom & Broadband Customer Churn Predictor

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-EB6424.svg)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-green.svg)](https://shap.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Preprocessing-Scikit--Learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning web application designed to predict customer churn for Sri Lankan telecommunications and broadband service providers (e.g., SLT-Mobitel, Dialog, Airtel). 

Powered by **XGBoost** for classification and **SHAP (SHapley Additive exPlanations)** for Explainable AI (XAI), the platform enables customer success and retention teams to identify high-risk subscribers in real time and understand the exact drivers behind each prediction.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Dataset & Sri Lankan Telecom Context](#-dataset--sri-lankan-telecom-context)
- [Project Architecture](#-project-architecture)
- [Directory Structure](#-directory-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage Workflow](#-usage-workflow)
  - [1. Generate Synthetic Data](#1-generate-synthetic-data)
  - [2. Train the XGBoost Model](#2-train-the-xgboost-model)
  - [3. Launch the Streamlit App](#3-launch-the-streamlit-app)
- [Machine Learning & XAI Pipeline](#-machine-learning--xai-pipeline)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Overview

In the competitive Sri Lankan telecommunication landscape, retaining subscribers is significantly more cost-effective than customer acquisition. This tool bridges the gap between machine learning insights and actionable business operations:

1. **Calculates Churn Probability**: Delivers real-time percentage risk for any subscriber configuration.
2. **Generates Prescriptive Retention Strategies**: Recommends targeted interventions (discounts, speed upgrades, loyalty perks) based on risk tier.
3. **Transparent Explainable AI (XAI)**: Visualizes positive and negative churn triggers using SHAP Waterfall plots to eliminate black-box machine learning.

---

## ✨ Key Features

- **Domain-Specific Modeling**: Formulated around Sri Lankan telecom dynamics—including local payment ecosystems (**FriMi**, **eZ Cash**, over-the-counter payments) and localized billing standards (**LKR**).
- **Explainable Predictions**: Powered by `shap.TreeExplainer` to visualize how each individual feature pushes churn risk up or down.
- **Dynamic Decision System**: Categorizes subscribers into Low Risk and High Risk, prescribing immediate actionable retention countermeasures.
- **Interactive Web Interface**: Clean, wide-layout dashboard built with Streamlit featuring responsive sidebar sliders and dropdown controls.
- **Reproducible Pipeline**: Includes end-to-end synthetic data synthesis, model training, artifact serialization (`joblib`), and serving scripts.

---

## 📊 Dataset & Sri Lankan Telecom Context

The model is trained on synthetic subscriber records (`churn_data.csv`) modeled after Sri Lankan broadband and telecommunication patterns:

| Feature Name | Type | Description / Range | Contextual Relevance |
| :--- | :--- | :--- | :--- |
| `Tenure_Months` | Numeric | 1 to 72 months | Loyalty lifecycle (newer customers have higher churn propensity). |
| `Monthly_Bill_LKR` | Numeric | 1,500 – 15,000 LKR | Average monthly spend in Sri Lankan Rupees. |
| `Support_Tickets_30d` | Numeric | 0 to 10 complaints | Volume of network/customer service issues in the last 30 days. |
| `Connection_Type` | Categorical | `Fibre`, `4G Home Broadband`, `Postpaid Mobile` | Differentiates fixed high-speed fiber from volatile 4G wireless. |
| `Payment_Method` | Categorical | `eZ Cash / FriMi / e-Wallet`, `Bank Direct Debit`, `Over the Counter / Cash` | Captures digital vs. manual recurring payment behaviors. |
| `Churn` | Binary Target | `0` (Retained) / `1` (Churned) | Ground truth status. |

---

## 🏗 Project Architecture

```mermaid
flowchart LR
    A[Synthetic Data Generator<br/>generate_data.py] -->|churn_data.csv| B[ML Training Pipeline<br/>train_model.py]
    B -->|Preprocesses with Scikit-Learn| C[ColumnTransformer / OHE]
    C -->|Trains Classifier| D[XGBoost Classifier]
    D -->|Exports Artifacts| E[model.pkl<br/>preprocessor.pkl<br/>feature_names.pkl]
    E --> F[Streamlit Web App<br/>app.py]
    F -->|Real-Time Input| G[Prediction & Retention Actions]
    F -->|TreeExplainer| H[SHAP Waterfall Plot]
