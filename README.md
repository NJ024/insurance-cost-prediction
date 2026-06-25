# 🏥 Medical Insurance Cost Prediction

Predicting individual medical insurance charges from demographic and lifestyle attributes using exploratory data analysis, feature engineering, statistical feature selection, and linear regression.

**Final model: R² = 0.804 | Adjusted R² = 0.799 (on held-out test set)**

---

## 📌 Project Overview

Health insurers price premiums based on a mix of demographic, physiological, and behavioral risk factors. This project explores how well those factors — age, BMI, smoking status, number of dependents, sex, and region — explain variation in actual medical charges, and builds a regression model to predict charges for new individuals.

The goal wasn't just to fit a model, but to understand *which* factors actually drive cost, using statistical tests (Pearson correlation, Chi-square independence tests) rather than just throwing every feature into a model.

## 📊 Dataset

- **Source:** [Medical Cost Personal Dataset (Kaggle)](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Size:** 1,338 records, 7 columns
- **Features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`
- **Target:** `charges` (medical cost billed by health insurance, USD)

## 🔍 Workflow

1. **Exploratory Data Analysis** — distribution plots, boxplots for outlier checks, correlation heatmap
2. **Data Cleaning** — duplicate removal, binary encoding (`sex`, `smoker`), one-hot encoding (`region`)
3. **Feature Engineering** — derived a `bmi_category` feature (underweight / healthy / overweight / obese) from clinical BMI thresholds
4. **Feature Scaling** — standardized continuous features (`age`, `bmi`, `children`)
5. **Feature Selection** —
   - Pearson correlation for numeric/binary features against `charges`
   - Chi-square test of independence (against binned `charges`) to validate which categorical features are statistically significant
6. **Modeling** — Linear Regression on the selected feature subset, 80/20 train-test split
7. **Evaluation** — R² and Adjusted R² on the test set

## 💡 Key Findings

| Feature | Pearson Correlation with Charges |
|---|---|
| Smoker status | **0.79** (by far the strongest driver) |
| Age | 0.30 |
| Obesity (BMI category) | 0.20 |

- **Smoking status is the single strongest predictor of cost** — far ahead of age or BMI individually.
- Chi-square tests confirmed `smoker` and `region_southeast` as statistically significant categorical predictors (p < 0.05), while several other regional/BMI categories were not.
- Final feature set used for modeling: `age`, `is_female`, `bmi`, `children`, `is_smoker`, `region_southeast`, `bmi_category_obese`

## 📈 Results

| Metric | Score |
|---|---|
| R² (test set) | 0.804 |
| Adjusted R² (test set) | 0.799 |

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `SciPy` · `Scikit-learn`

## 📁 Repository Structure

```
insurance-cost-prediction/
├── data/
│   └── insurance.csv
├── notebooks/
│   └── insurance_eda_and_modeling.ipynb
├── images/
│   ├── correlation_heatmap.png
│   └── charges_distribution.png
├── README.md
├── requirements.txt

```

## ▶️ How to Run

```bash
git clone https://github.com/NJ024/insurance-cost-prediction.git
cd insurance-cost-prediction
pip install -r requirements.txt
jupyter notebook notebooks/insurance_eda_and_modeling.ipynb
```

## 🚀 Future Improvements

- Compare Linear Regression against Random Forest / Gradient Boosting / XGBoost with cross-validation
- Add residual diagnostics (heteroscedasticity, normality of residuals) to check linear regression assumptions
- Use SHAP values for model interpretability
- Deploy as a lightweight Streamlit app where a user inputs age/BMI/smoker status and gets a predicted premium estimate
- Hyperparameter tuning with GridSearchCV

