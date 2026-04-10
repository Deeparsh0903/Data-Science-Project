🏏 IPL Delivery Run Prediction — ML Regression Project

Predicting total runs scored per delivery in IPL cricket using an end-to-end machine learning pipeline with data leakage detection, feature engineering, and multi-model comparison.


📌 Project Overview
This project applies supervised machine learning regression techniques on IPL ball-by-ball delivery data to predict the total_runs scored on each delivery. The project covers the complete data science workflow — from raw data exploration to hyperparameter-tuned model deployment — with a strong emphasis on data integrity and honest evaluation.
A key highlight of this project is the identification and removal of data leakage: the dataset contains batsman_runs and extra_runs which mathematically sum to total_runs, causing any model using them to achieve a false R² of 1.0. Removing these and building a clean, leak-free pipeline is the core engineering challenge addressed here.

ipl-delivery-run-prediction/
│
├── deliveries_last12000.csv       # Dataset (12,000 ball-by-ball records)
├── ipl_regression_project.ipynb   # Main Jupyter Notebook (full pipeline)
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Source | IPL ball-by-ball delivery data |
| Rows | 12,000 deliveries |
| Columns | 17 features |
| Target Variable | `total_runs` (0–7 runs per ball) |



## ⚙️ Pipeline

### 1. Data Cleaning
- Filled structural nulls (`extras_type`, `dismissal_kind`, `fielder`) with `'none'`
- Removed duplicate rows
- Dropped **leakage columns**: `batsman_runs`, `extra_runs`
- Dropped post-delivery columns: `is_wicket`, `player_dismissed`, `dismissal_kind`, `fielder`
- Dropped irrelevant columns: `match_id`, `non_striker`

### 3. Encoding
- Label Encoding applied to: `batting_team`, `bowling_team`, `batter`, `bowler`, `extras_type`

### 4. Train-Test Split
- 80% training / 20% testing
- `random_state=42` for reproducibility

---

## 🤖 Models Trained

 Model | Notes 
 Linear Regression | Baseline linear model |
 Random Forest Regressor | Ensemble of decision trees |
 XGBoost Regressor | Gradient boosted trees |
 Gradient Boosting Regressor | Sklearn's native boosting implementation |

---

## 📈 Results

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| Linear Regression | +0.0652 | 1.7031 | 1.3142 |
| Random Forest | +0.0460 | 1.7205 | 1.3274 |
| Tuned XGBoost | +0.0536 | 1.7136 | 1.3152 |
| Gradient Boosting | −0.0110 | 1.7712 | 1.3550 |
| Baseline (mean) | 0.0000 | 1.7868 | 1.3681 |

> **Best Model: Linear Regression (R² = 0.065)**
> Tuned XGBoost is a close second and more robust across different data splits.

### Why is R² low?

This is **expected and correct** for this problem:

- `total_runs` depends on split-second batsman decisions — inherently unpredictable
- No in-play context available (field placement, pitch conditions, weather)
- Only 8 possible values (0–7) — a discrete distribution that regression naturally struggles with
- Any model claiming R² ≈ 1.0 on this data is using leakage features

A well-engineered, honest R² of **0.05–0.15 is realistic and meaningful** for ball-level run prediction.

---

## 🔍 Top Predictive Features

Based on XGBoost feature importance:

1. `bowler_avg_conceded` — how expensive is this bowler historically?
2. `batter_avg_runs` — how productive is this batsman historically?
3. `over_avg_runs` — average run rate in this over number
4. `is_wide` / `is_noball` — extras directly inflate runs
5. `is_death_over` — run rate spikes in overs 16–19

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | ML models, preprocessing, evaluation |
| XGBoost | Gradient boosted regression |
| Jupyter Notebook | Interactive development |

---

## 🚀 How to Run

**1. Clone the repository**
git clone https://github.com/your-username/ipl-delivery-run-prediction.git
cd ipl-delivery-run-prediction

**2. Install dependencies**
pip install -r requirements.txt

**3. Launch Jupyter Notebook**
jupyter notebook ipl_regression_project.ipynb

---

## 📦 Requirements

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter

Save as `requirements.txt` or install directly:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter

---

## 💡 Future Improvements

- Use **rolling averages** (last 5–10 balls) instead of global historical averages
- Add **venue/stadium features** — some grounds are significantly higher scoring
- Try **Ordinal Regression or Poisson Regression** — better suited for discrete count targets
- Include **match-state features** (current score, required run rate, wickets fallen)
- Experiment with **SHAP values** for more interpretable feature importance

---

## 📚 Key Learnings

- **Data leakage is the #1 silent killer** of ML model validity — always verify features are available at prediction time
- **Domain knowledge matters** — understanding cricket helped identify post-event vs pre-event columns
- **High R² is not always the goal** — an honest low R² on a stochastic problem is far more valuable than an inflated score from leaky features
- **Feature engineering** using historical player/team performance added more signal than any model tuning

---

## 👨‍💻 Author

Deeparsh Singh
B.Tech — [Lovely Professional Univers]
[Lovely Professional University]
[LinkedIn Profile] | [GitHub Profile]

---

## 📄 License

This project is open source and available under the MIT License.
