# 📈 Quantitative Portfolio Risk & Directional ML Alpha Engine

An end-to-end, production-grade quantitative asset analytics engine and predictive machine learning dashboard. This software architecture utilizes decoupled pipeline automation to clean live time-series variables, extract mathematical volatility arrays, map cross-asset diversification risks, and evaluate next-day price movement directions.

🚀 **[Click Here to View the Live Web Application Interface](YOUR_STREAMLIT_CLOUD_LINK_HERE)**

---

## 🛠️ System Architecture & Workflow Blueprint

1. **Ingestion Pipeline (`src/finance_pipeline.py`)**: Connects to financial market endpoints to collect 5 years of daily closing bars for blue-chip assets (`AAPL`, `MSFT`, `GOOGL`, `AMZN`).
2. **Feature Engineering**: Vectorizes raw financial price arrays into technical micro-features (Moving Averages, Trailing Historical Volatility, Volume metrics).
3. **Machine Learning Inference**: Trains decoupled `Scikit-Learn` Gradient Boosting Decision Trees (GBDT) on past chronological sequences to isolate technical market confluences.
4. **Interactive Dashboarding (`finance_app.py`)**: Exposes an executive BI control interface using `Seaborn` multi-asset covariance heatmaps and live input scoring vectors.

---

## 🔬 Core Technologies & Mathematical Stack

* **Environment Infrastructure**: Anaconda Virtual Environments Engine (`python=3.10`)
* **Data Processing & Analytics**: Pandas Vector Dataframes, NumPy Vectorization Elements
* **Statistical Graphics**: Seaborn Core, Matplotlib Graphics Framework
* **Machine Learning Engineering**: Scikit-Learn Ensemble Gradient Boosting Classification Tree Framework
* **Production UI Deployment**: Streamlit Micro-Web Application Hosting Server

---

## 📊 Core Application Features

### 1. Multi-Asset Risk Analytics (Business Intelligence Track)
* Automated calculations of vectorized asset annual realized volatility coefficients.
* Dynamic **Seaborn Cross-Asset Correlation Matrices** illustrating directional beta shifts to evaluate systemic portfolio concentration risks and portfolio diversification limits.

### 2. Directional Movement Alpha Engine (Machine Learning Track)
* Trailing chronological training splits preserving strict financial time-series integrity (preventing future-lookahead data leakage).
* Live scoring blocks feeding current indicator matrices directly into trained tree models.
* Real-time extraction of classification certainty probabilities via `.predict_proba()` to compute high-conviction market signals (e.g., 85% confidence vectors).

---

## 📦 Project Directory Layout

```text
HealthcareFinanceProject/
│
├── .gitignore               # Excludes large binaries, datasets, and local artifacts
├── requirements.txt         # Pinned enterprise dependency mapping
├── finance_app.py           # Production Streamlit UI Engine
├── generate_data.py         # Mock medical infrastructure foundation (Healthcare track)
│
├── src/
│   ├── __init__.py          # Marks folder as importable package module
│   └── finance_pipeline.py  # Production automated data extraction and ML optimization script
│
└── data/
    └── processed/           # Local storage cache for engineered variables and trained model binaries
```

---

## ⚙️ Local Installation & Reproducibility Matrix

Follow these terminal steps to execute this platform inside an isolated local environment:

```bash
# 1. Clone this repository structure
git clone https://github.com
cd Quant-Trading-ML-Dashboard

# 2. Spin up the dedicated virtual micro-environment via Anaconda
conda create -n internship_env python=3.10 -y
conda activate internship_env

# 3. Secure dependency lock files
pip install -r requirements.txt

# 4. Trigger backend computation network to extract data assets and compile models
python src/finance_pipeline.py

# 5. Launch the live dashboard interface locally
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python -m streamlit run finance_app.py
```

---
## 💡 Analytical Insight: Model Target Accuracy Reflection
*The optimization layer yields ~50-51% accuracy on chronological testing frameworks. In efficient financial markets, predicting raw asset delta vector directions presents a near-random walk distribution. This project serves as a robust engineering framework. Production-grade scaling can be expanded by injecting alternative data metrics, order book market depths, and sentiment extraction components.*
