# 📈 Machine Learning Framework for Gold Price Trend Forecasting (VNĐ/Lượng)

## 📌 Problem Overview
Forecasting domestic gold price movements in Vietnam requires capturing a **dual-risk mechanism**:
1. International commodity price shocks (**XAU/USD**).
2. Domestic currency depreciation and foreign exchange fluctuations (**USD/VND**).

Unlike traditional regression models that suffer from price inertia, this project frames the challenge as a **5-day forward binary classification problem ($t+5$)**, delivering actionable trading signals for short-term risk management and portfolio allocation.

---

## 🛠 Tech Stack & Workflow
- **Language & Core:** Python, Pandas, NumPy, Scikit-learn, Statsmodels, Matplotlib
- **Algorithms:** Logistic Regression (L2-Regularized Baseline), Random Forest, Bootstrap Aggregating Decision Trees (Tree Bagging)
- **Data Engineering:** Winsorization (1%–99%), Z-score Standardization (strictly fitted on Train set), Multicollinearity Testing (VIF)

---

## 🔬 Mathematical Logic & Methodology

### 1. Target Definition
$$\text{Return}_{t,5} = \frac{P_{t+5} - P_t}{P_t} \quad \longrightarrow \quad Y_t = \begin{cases} 1 & \text{if } \text{Return}_{t,5} > 0 \text{ (Upward Trend)} \\ 0 & \text{if } \text{Return}_{t,5} \le 0 \text{ (Downward/Neutral)} \end{cases}$$
Where $P_t$ is the converted price (VNĐ/lượng):
$$P_t = \text{XAU/USD}_t \times \text{USD/VND}_t \times \frac{37.5}{31.1035}$$

### 2. Time-Series Preserving Split (Preventing Data Leakage)
To prevent forward-looking bias (data leakage), the dataset (1,496 daily records from 2020 to 2025) is split sequentially without shuffling:
- **Training Set (2020–2023):** 988 records for model parameter fitting and VIF diagnostics.
- **Validation Set (2024):** 254 records for Hyperparameter Tuning (AUC-ROC) and Optimal Classification Threshold Selection (Balanced Accuracy).
- **Out-of-Sample Test Set (2025):** 254 records strictly reserved for final empirical evaluation.

### 3. Feature Space (30 Variables)
- **Price Dynamics & Returns:** Daily log-returns and lag structures ($t-1$ to $t-20$).
- **Trend & Momentum Indicators:** Distance to moving averages (`p_over_ma5`, `p_over_ma20`, `ma5_over_ma20`), multi-horizon momentum (`mom5`, `mom20`, `mom60`).
- **Volatility & Oscillators:** Rolling standard deviations (`vol5`, `vol20`, `vol60`), RSI (14-day), MACD & MACD Histogram.
- **Macro & Currency Linkages:** USD/VND log-returns, rolling FX volatility (`fx_vol20`), S&P 500, VIX Index, US 10Y Treasury Yields (`DGS10`), and WTI Crude Oil prices.

---

## 📊 Empirical Evaluation (Out-of-Sample Test 2025)

| Metric | Logistic Regression (Baseline) | Random Forest | Tree Bagging (Champion) |
| :--- | :---: | :---: | :---: |
| **Optimal Threshold** | 0.435 | 0.505 | **0.425** |
| **Accuracy** | 59.06% | 60.63% | **64.57%** |
| **Precision** | **86.41%** | 82.64% | **81.56%** |
| **Recall (Sensitivity)** | 49.72% | 55.87% | **64.25%** |
| **F1-Score** | 0.6312 | 0.6667 | **0.7188** |
| **AUC-ROC** | 0.6696 | 0.6664 | **0.6743** |
| **Specificity** | 81.33% | 72.00% | **65.33%** |
| **Balanced Accuracy** | 65.53% | 63.93% | **64.79%** |

---

## 💡 Key Findings & Practical Insights

1. **Non-linear Ensemble Superiority:** Tree Bagging effectively mitigates high multicollinearity among technical indicators (where VIF exceeded critical thresholds in the linear baseline), yielding the highest predictive stability (AUC = 0.674, F1 = 0.7188).
2. **The USD/VND Volatility Anchor:** Feature Importance analysis revealed that **20-day rolling exchange rate volatility (`fx_vol20`) is the single most decisive predictor (8.2% weight)**, surpassing traditional gold momentum indicators. This confirms that currency instability directly amplifies domestic safe-haven demand in Vietnam.
3. **High Precision for Capital Preservation:** Both Tree Bagging and Random Forest achieved Precision above **81%**, meaning when the system signals an "Upward Trend", false positives are strictly contained, offering significant downside risk protection for market participants.
