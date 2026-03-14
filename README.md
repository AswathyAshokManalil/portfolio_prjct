# <div align="center">📈 An End-to-End Deep Learning System for Stock Portfolio Optimization</div>

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/TensorFlow-2.10+-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-v1.20+-ff4b4b.svg" alt="Streamlit">
</div>

---

## 📑 Table of Contents
* [1. Project Overview](#1-project-overview)
* [2. System Architecture](#2-system-architecture)
* [3. Data Collection](#3-data-collection)
* [4. Feature Engineering](#4-feature-engineering)
* [5. Data Preparation](#5-data-preparation)
* [6. Model Architecture](#6-model-architecture)
* [7. Model Training & Evaluation](#7-model-training--evaluation)
* [8. Model Analysis & Validation](#8-model-analysis--validation)
* [9. Live Recommender System](#9-live-recommender-system)
* [10. User Interface](#10-user-interface)
* [11. Performance Metrics](#11-performance-metrics)
* [12. Installation & Setup](#12-installation--setup)

---

## 1. 📋 Project Overview
The **AI-Powered Portfolio Recommender** is a complete end-to-end machine learning system that:
* **Collects** historical stock data from NIFTY 50 companies (47+ stocks).
* **Processes** data by creating 15+ technical indicators and a target variable.
* **Trains** deep learning models (**LSTM** and **GRU**) to predict price direction.
* **Analyzes** performance using walk-forward validation and confidence thresholds.
* **Recommends** a personalized portfolio based on investment amount and risk tolerance.

---

## 2. 🏗️ System Architecture
```text
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ Data Collection │───▶  │ Preprocessing   │───▶  │ Model Training  │
│ (yfinance)      │      │ & Feature Eng   │      │ (LSTM/GRU)      │
└─────────────────┘      └─────────────────┘      └────────┬────────┘
                                                           │
                                                           ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Streamlit UI   │◀───  │  Recommender    │◀───  │ Model Analysis  │
│  (portfolio_ui) │      │  Engine         │      │ & Backtesting   │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```
## 3. 📥 Data Collection

### 3.1 Data Sources

- **data_collection.py**  
  Downloads daily **OHLCV** data for **47+ NIFTY 50 tickers**.

- **datacollection_nifty50.py**  
  Downloads the **NIFTY 50 index (^NSEI)** as a market benchmark.

### 3.2 Why NIFTY Index Data?

By adding the **NIFTY_Return** feature, the model learns to separate **market-driven movements** from **company-specific performance (Alpha)**.

---

## 4. 🔬 Feature Engineering

### 4.1 Technical Indicators
| Category      | Features                          | Purpose                          |
|---------------|----------------------------------|----------------------------------|
| Price Action  | Open, High, Low, Close, Volume  | Base market inputs               |
| Trend         | MA7, MA30                       | Short & medium-term direction    |
| Volatility    | Volatility_7, Volatility_14     | Risk measurement                 |
| Momentum      | RSI, MACD, MACD_Signal          | Entry/exit timing signals        |
### 4.2 Target Variable

- **1**: Next day's close > today's close  
- **0**: Next day's close ≤ today's close  

---

## 5. 📊 Data Preparation

- **Scaling:** `MinMaxScaler` is fitted per stock to normalize price ranges.
- **Sequencing:** Inputs are transformed into **3D arrays** using a **30-day lookback window**.
- **Train-Test Split:** Chronological **80/20 split** to respect temporal order.

---

## 6. 🤖 Model Architecture

### 6.1 Why LSTM?

Designed for **time-series data**.  
Its **Forget Gate** allows the model to ignore random daily noise while remembering **long-term market trends**.

---

### 6.2 Why GRU?

A simplified recurrent unit with **fewer parameters**, which results in:

- Faster training
- Lower overfitting risk
- Comparable performance to LSTM on financial time-series data

---

### 6.3 LSTM vs GRU Comparison

| Metric | LSTM | GRU |
|------|------|------|
| Accuracy | 67.45% | 68.10% |
| F1 Score | 71.25% | 72.17% |
| Parameters | 20,545 | 15,617 |
## 7. 🧠 Model Training & Evaluation

- **Loss Function:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Regularization:** 0.2 Dropout  
- **Early Stopping:** Patience = 5 epochs  

---

## 8. 📈 Model Analysis & Validation

### 8.1 Confidence Threshold Analysis

Increasing the probability requirement improves accuracy but reduces coverage.

| Threshold | Accuracy | Coverage |
|-----------|----------|----------|
| 0.50 | 82.85% | 21.54% |
| 0.60 | 85.40% | 11.76% |
| 0.70 | 90.07% | 5.46% |

---

### 8.2 Walk-Forward Validation

Simulates real-world trading by **retraining the model on a sliding window of data**.

Average accuracy achieved:

**47.62%**

---

## 9. 🎯 Live Recommender System

The **PortfolioRecommender** class performs the following tasks:

- **Ensemble Logic:** Averages probabilities from LSTM and GRU models  
- **Volatility Analysis:** Categorizes stocks into **Low, Medium, and High risk** groups  
- **Allocation:** Dynamically calculates stock quantity based on **user investment capital**

---

## 10. 🌐 User Interface

The **Streamlit dashboard** provides:

- **Interactive Inputs:** Investment amount and risk appetite  
- **Dynamic Recommendations:** High-confidence buy signals  
- **Portfolio Charts:**  
  - Pie charts for diversification  
  - Plotly line graphs for stock history  

---

## 11. 📊 Performance Metrics

| Model | Accuracy | Precision | Recall |
|------|----------|-----------|--------|
| LSTM | 67.45% | 70.44% | 72.08% |
| GRU | 68.10% | 70.52% | 73.90% |
---

## 🏁 Conclusion

The **Stock Portfolio Optimization System** successfully demonstrates the integration of **deep learning with financial theory** to create a data-driven investment tool. By shifting away from traditional static analysis and embracing **gated recurrent architectures**, the system achieves a more nuanced understanding of market dynamics.

---

## 🔑 Key Takeaways

### Model Synergy
The project highlights that while **GRU models** offer superior computational efficiency and slightly better raw accuracy, an **Ensemble approach** (combining **LSTM and GRU**) provides more stable predictions by mitigating the individual biases of each architecture.

### The Power of Selectivity
Through **Confidence Threshold Analysis**, we demonstrated that prediction accuracy can increase from **68% to 90%** by applying stricter confidence thresholds. This emphasizes a **“quality over quantity”** strategy in automated trading systems.

### Realism over Optimism
The implementation of **Walk-Forward Validation** provided a realistic evaluation of model performance. It revealed that **real-world trading accuracy differs from backtested results**, ensuring the system remains a **decision-support tool** rather than a speculative black-box trading system.

### User-Centric Design
By integrating the deep learning models into a **Streamlit-based dashboard**, the system makes advanced financial modeling accessible to everyday investors. Users can input **investment capital and risk preferences** to receive **personalized portfolio recommendations and diversification insights**.
