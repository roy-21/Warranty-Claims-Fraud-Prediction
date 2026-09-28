# 🛡️ Warranty Claims Fraud Prediction

An end-to-end Machine Learning project designed to analyze customer, location, product, call center, and service data to detect and predict **fraudulent warranty claims** in consumer electronics (Air Conditioners and Televisions).

---

## 📌 Project Overview

Warranty claim fraud costs manufacturing and consumer electronics companies millions annually. The primary goal of this data science project is to build machine learning models capable of identifying fraudulent warranty claims using historical claim records.

- **Dataset Source**: Kaggle
- **Dataset Size**: 358 Records × 21 Features
- **Problem Type**: Supervised Binary Classification (`Fraud`: 1 = Fraudulent, 0 = Genuine)
- **Primary Algorithms**: Decision Tree, Random Forest, Logistic Regression

---

## 📊 Data Dictionary

| Feature Name | Description | Data Type |
|---|---|---|
| `Region` | Geographic region of the claim (North, South, East, West) | Categorical |
| `State` | State where the claim originated | Categorical |
| `Area` | Location type (`Urban` / `Rural`) | Categorical |
| `City` | City where the claim was filed | Categorical |
| `Consumer_profile` | Customer profile category (`Personal` vs `Business`) | Categorical |
| `Product_category` | Product domain (`Household` / `Entertainment`) | Categorical |
| `Product_type` | Equipment type (`AC` / `TV`) | Categorical |
| `AC_1001_Issue` to `AC_1003_Issue` | AC component failure code (0 = No Issue, 1 = Repair, 2 = Replacement) | Categorical / Ordinal |
| `TV_2001_Issue` to `TV_2003_Issue` | TV component failure code (0 = No Issue, 1 = Repair, 2 = Replacement) | Categorical / Ordinal |
| `Claim_Value` | Monetary value of the warranty claim (in INR) | Numerical |
| `Service_Center` | Code identifier for the servicing center | Categorical / Numerical |
| `Product_Age` | Age of product since purchase (in days) | Numerical |
| `Purchased_from` | Purchase channel (`Manufacturer`, `Dealer`, `Internet`) | Categorical |
| `Call_details` | Customer care call duration (in minutes) | Numerical |
| `Purpose` | Call purpose (`Claim`, `Complaint`, `Information`) | Categorical |
| `Fraud` | **Target Variable** — `1` (Fraudulent), `0` (Genuine) | Binary (0 / 1) |

---

## 🔍 Key Insights from Exploratory Data Analysis (EDA)

1. **Geographic Concentration**:
   - Warranty claims are heavily clustered in **Southern India** (predominantly Andhra Pradesh & Tamil Nadu).
   - **Urban cities** (e.g., Hyderabad and Chennai) account for a significantly higher proportion of fraudulent claims.
2. **Purchase Channels**:
   - Claims on products purchased directly from the **Manufacturer** exhibit a higher fraud rate compared to Dealer or Internet sales.
3. **Claim Value & Product Lifecycle**:
   - Fraudulent claims feature **substantially higher claim values** than genuine ones.
   - Fraud attempts peak shortly after purchase (often within **< 50 days of product age**).
4. **Service Center Patterns**:
   - **Service Center 13** registered an abnormally high ratio of fraudulent claims relative to its total claim volume.
5. **Call Behavior**:
   - Fraudulent claims often correspond to short call durations (typically under 3–4 minutes) filed under `Complaint` or `Claim`.

---

## 🛠️ Data Preprocessing & Workflow

1. **Data Cleaning**: Checked for null values (0 found) and exact duplicates (0 found).
2. **Outlier Mitigation**: Applied the Interquartile Range (IQR) technique to handle extreme outliers in `Claim_Value`.
3. **Categorical Encoding**: Encoded categorical attributes using `LabelEncoder`.
4. **Train-Test Split**: Divided data into 70% Training and 30% Test sets (`random_state=42`).
5. **Hyperparameter Tuning**: Utilized `GridSearchCV` to optimize Decision Tree and Random Forest parameters.

---

## 🤖 Models & Evaluation

| Model | Hyperparameter Optimization | Configuration / Tuning Parameters | Performance Summary |
|---|---|---|---|
| **Decision Tree Classifier** | `GridSearchCV` | `criterion='gini'`, `max_depth=4`, `min_samples_leaf=2`, `min_samples_split=2` | **~91-92% Accuracy** |
| **Random Forest Classifier** | `GridSearchCV` | `criterion='gini'`, `max_depth=2`, `min_samples_leaf=2`, `min_samples_split=2` | **~91-92% Accuracy** |
| **Logistic Regression** | Baseline | Default parameters | **~90-91% Accuracy** |

> ⚠️ **Class Imbalance Note**: Because fraudulent claims represent a minority (~9.8%) of the dataset, accuracy is high due to correct genuine class predictions. Fraud recall can be further enhanced using oversampling.

---

## 🚀 Future Enhancements

- 🔄 **Class Oversampling**: Apply **SMOTE** (Synthetic Minority Over-sampling Technique) to balance minority fraud cases.
- ⚡ **Gradient Boosting**: Integrate state-of-the-art algorithms like **XGBoost**, **LightGBM**, or **CatBoost**.
- 📐 **Encoding Upgrades**: Replace `LabelEncoder` with **One-Hot Encoding** or **Target Encoding** for nominal features.
- 🧪 **Cross Validation**: Implement Stratified $K$-Fold Cross Validation to ensure generalized performance.
- 🌐 **Web Dashboard & Deployment**: Build a **Streamlit** interactive dashboard and deploy via **FastAPI**.

---

## 🛠️ Requirements & How to Run

### Dependencies
- Python 3.8+
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `jupyter`

### Running the Notebook

1. Clone or download this project workspace.
2. Open your terminal or command prompt in the workspace directory.
3. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```
4. Run Jupyter Notebook:
   ```bash
   jupyter notebook "Warranty Claims Fraud Prediction.ipynb"
   ```

---

## 📂 Repository Structure

```
Warranty Claims Fraud Prediction/
├── Warranty Claims Fraud Prediction.ipynb   # Main Jupyter Notebook containing EDA & ML code
├── Warranty Claims Fraud Prediction.pdf     # PDF export of the notebook
├── df_Clean.csv                            # Processed dataset post-cleaning
├── description.md                           # Initial project summary & data dictionary
├── notebook_briefing.md                     # Deep technical analysis report
└── README.md                                # Comprehensive project documentation
```

---

## 📜 License
This project is open-source and intended for analytical, research, and educational purposes.
