# 🌍 CSSE-415 Team 5 – Predicting the Gini Index 🎯

## 📖 Full Project Summary
In this project, we developed a comprehensive, end-to-end pipeline to forecast next-year Gini coefficients—the most widely used measure of income inequality—for 163 countries using World Bank Indicators (2002–2020). We began by ingesting and merging raw CSVs, then rigorously cleaned and imputed missing values (per-country means with global fallback), producing a complete dataset of socio-economic metrics. We engineered three complementary feature representations:  
- **X1:** single-year lags  
- **X3:** multi-year (1–5-year) lags  
- **X2:** annual value plus 5-year rolling mean & standard deviation  

Next, we benchmarked a diverse suite of models—from a naive bias baseline, through linear regressions (with and without forward feature selection), k-nearest neighbors, and random forests, to gradient boosted trees—using group-aware cross-validation to prevent country-level leakage. Our top performer, **Gradient Boosted Trees on rolling statistics (X2)**, achieved **53.6 % R²**, and surfaced the key drivers of global inequality:  
- 💥 **Homicide rate** (per 100 000 people)  
- 🌲 **Forest land** (% of land area)  
- ⚡ **Access to electricity** (% of population)  
- ⚰️ **Death rate** (per 1 000 people)  

Finally, we serialized our best model (`best_gbr_model.pkl`) and scaler (`scaler.pkl`) and wrapped the workflow into an **interactive Jupyter demo**, allowing users to select any country & year to generate on-demand Gini forecasts. This framework not only delivers strong predictive performance but also empowers policymakers and researchers with actionable insights into the socioeconomic factors driving inequality.

## 🧑‍💻 Authors
- Abdullah Islam  
- Agnay Srivastava  
- Parth Sundaram  
- Steven Johnson  

---
## 📂 Directory Structure

```text
.
├── World Bank Indicators.pdf         📑  (project report & slides)
└── Code/
    ├── 01_Clean_Impute.ipynb         🧼  Data cleaning & imputation
    ├── 02_Simple_Regressors.ipynb    🔢  Benchmark basic regressors
    ├── 02b_LR_with_FS.ipynb          🎯  Linear regression + feature selection
    ├── 03_RandomForest.ipynb         🌳  Random Forest hyperparameter tuning
    ├── 04_Demo_Prep.ipynb            🛠️  Demo data preparation
    ├── 05_Demo.ipynb                 🚀  Interactive prediction demo
    ├── best_gbr_model.pkl            📦  Serialized best GBT model
    ├── scaler.pkl                    ⚖️  Feature scaler for demo
    ├── world_bank.csv                🌐  Raw merged data
    ├── groups_train.csv              🚉  Training group assignments
    ├── groups_test.csv               🚥  Test group assignments
    ├── groups_demo.csv               🎬  Demo group assignments
    ├── X1_train.csv                  ⏳  1-year lag features (train)
    ├── X1_test.csv                   ⏳  1-year lag features (test)
    ├── X1_demo.csv                   ⏳  1-year lag features (demo)
    ├── X2_train.csv                  📈  Year + 5-yr rolling stats (train)
    ├── X2_test.csv                   📈  Year + 5-yr rolling stats (test)
    ├── X2_demo.csv                   📈  Year + 5-yr rolling stats (demo)
    ├── X3_train.csv                  🔄  1–5-yr lag features (train)
    ├── X3_test.csv                   🔄  1–5-yr lag features (test)
    ├── X3_demo.csv                   🔄  1–5-yr lag features (demo)
    ├── y_train.csv                   🎯  Target Gini (train)
    ├── y_test.csv                    🎯  Target Gini (test)
    ├── y_demo.csv                    🎯  Target Gini (demo)
    ├── Data Dump/                    📂  Raw & imputed CSVs by method
    │   ├── world_bank.csv           🌐  Raw data copy
    │   ├── cleaned.csv              🧹  Cleaned & default-imputed dataset
    │   ├── Mean/                    ➖  Mean imputation
    │   │   ├── mean_train.csv       ➖  Train set
    │   │   ├── mean_test.csv        ➖  Test set
    │   │   └── cleaned_mean.csv     ➖  Full dataset
    │   ├── KNN/                     🤝  KNN imputation
    │   │   ├── knn_train.csv        🤝  Train set
    │   │   ├── knn_test.csv         🤝  Test set
    │   │   └── cleaned_knn.csv      🤝  Full dataset
    │   ├── MICE/                    🧩  MICE imputation
    │   │   ├── mice_train.csv       🧩  Train set
    │   │   ├── mice_test.csv        🧩  Test set
    │   │   └── cleaned_mice.csv     🧩  Full dataset
    │   └── Forest/                  🌲  MissForest imputation
    │       ├── forest_train.csv     🌲  Train set
    │       ├── forest_test.csv      🌲  Test set
    │       └── cleaned_forest.csv   🌲  Full dataset
    └── Old Notebooks/               📚  Archived exploratory work
        ├── 00_Preliminary_Analysis.ipynb   📚 Initial exploration
        ├── 01_Clean_Impute-Copy1.ipynb     📚 Early imputation trials
        ├── 02_Simple_Regressors-old.ipynb  📚 Legacy baselines
        └── Prelimary_Work.ipynb            📚 Misc. preliminary analyses

```


## 🔧 Dependencies & Setup
- **Python 3.8+**, **Jupyter Notebook**  
- **pandas**, **numpy**, **scikit-learn**, **joblib**, **matplotlib**, **xgboost** (or `sklearn.ensemble.GradientBoostingRegressor`)  
- Install essentials with:
  ```bash
  pip install pandas numpy scikit-learn matplotlib joblib xgboost
## 🚀 Usage Workflow

1. **Data Cleaning & Imputation**  
   - **Notebook:** `01_Clean_Impute.ipynb`  
   - **Steps:**  
     - Load raw `world_bank.csv` (2002–2020)  
     - Drop countries & indicators with excessive missingness  
     - Impute missing entries (per-country mean → global fallback)  
     - Split into feature sets & target:  
       - **X1:** 1-year lag (`X1_train.csv`, `X1_test.csv`, `X1_demo.csv`)  
       - **X2:** year + 5-yr rolling mean & std (`X2_*.csv`)  
       - **X3:** 1–5-yr lags (`X3_*.csv`)  
       - **y:** next-year Gini (`y_*.csv`)

2. **Baseline Modeling**  
   - **Notebook:** `02_Simple_Regressors.ipynb`  
   - **Models:**  
     - 🎯 Bias-only baseline  
     - 📝 Linear Regression  
     - 📐 Ridge (α grid)  
     - 🔗 Lasso (α grid)  
     - 🤝 K-Nearest Neighbors  
     - 🌳 Random Forest  
     - 🚀 Gradient Boosted Trees  
   - **Evaluation:** Group-aware cross-validation → R² scores for Sets 1, 2, 3.

3. **Feature Selection**  
   - **Notebook:** `02b_LR_with_FS.ipynb`  
   - **Method:** Forward stepwise selection on linear model  
   - **Outcome:** Identify top predictors, compare against full-feature model.

4. **Random Forest Tuning**  
   - **Notebook:** `03_RandomForest.ipynb`  
   - **Hyperparameters:**  
     - `n_estimators` ∈ [50, 260]  
     - `max_depth` ∈ [5, 20]  
   - **Goal:** Optimize RF per feature set for best R².

5. **Demo Preparation**  
   - **Notebook:** `04_Demo_Prep.ipynb`  
   - **Tasks:**  
     - Load `best_gbr_model.pkl` & `scaler.pkl`  
     - Assemble feature vector for chosen country & year  

6. **Interactive Demo**  
   - **Notebook:** `05_Demo.ipynb`  
   - **Flow:**  
     1. Select country from `groups_demo.csv`  
     2. Notebook scales inputs & predicts next-year Gini  
     3. Display result & feature‐importance summary  

---

## 📊 Results Summary

| Model                          | Set 1 (1-yr) | Set 2 (rolling) | Set 3 (lags) | **Avg R² %** |
|:-------------------------------|-------------:|---------------:|------------:|-------------:|
| **Bias-only**                  |        –2.2  |          –2.2  |      –2.2  |       –2.2   |
| **Linear Regression**          |        39.9  |          53.4  |      43.0  |       45.1   |
| **Linear + FS**                |        40.0  |          53.1  |      44.2  |       45.8   |
| **Ridge (α=100–1000)**         |        34.7  |          42.5  |      38.4  |       38.5   |
| **Lasso (α=0.01–1)**           |        35.6  |          51.1  |      33.3  |       40.0   |
| **Random Forest (50–260)**     |        54.3  |          49.3  |      49.7  |       51.1   |
| **Gradient Boosted Trees**     |        50.6  |          53.6  |      50.4  |       51.5   |
| **K-Nearest Neighbors (k≈40)** |        45.4  |          43.9  |      45.4  |       44.6   |

> 🚀 **Top Performer:** Gradient Boosted Trees on Rolling Stats (Set 2) with **53.6 % R²**

---

## 🌟 Key Drivers (GBT on Set 2)

1. 💥 **Intentional Homicides** (per 100 000 people)  
2. 🌲 **Forest Land** (% of land area)  
3. ⚡ **Electricity Access** (% of population)  
4. ⚰️ **Death Rate** (per 1 000 people)  

---

## 💡 Next Steps & Extensions

- 🔬 **Advanced Imputation**  
  - Compare MICE & MissForest in-depth  
- 📏 **Feature Engineering**  
  - Create ratio features (e.g., GDP / capita, health expenditure / population)  
- 🎯 **Dimensionality Reduction**  
  - Apply PCA or autoencoders to denoise input space  
- ⚖️ **Fairness Analysis**  
  - Evaluate bias across income tiers & regions  
- 🤖 **Bayesian & Hierarchical Models**  
  - Quantify uncertainty, incorporate multi-level priors  
- 📈 **Ensemble Blends**  
  - Combine top models from Sets 1–3 for robust forecasts  
