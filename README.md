# 🌱 Crop Recommendation System Using Machine Learning

> **A Data Science Based Approach for Crop Selection Using Soil and Environmental Parameters**

---

## 📌 Problem Statement

Selecting a suitable crop is a vital decision for sustainable and profitable agriculture. Traditional crop selection often depends on intuition or historical convention without systematically factoring in soil chemistry and environmental variations. 

Farmers must analyze multiple interrelated parameters:
- Soil nutrients: **Nitrogen (N)**, **Phosphorus (P)**, and **Potassium (K)**
- Soil chemical environment: **pH level**
- Local climatic conditions: **Temperature**, **Relative Humidity**, and **Rainfall precipitation**

Manually interpreting the interaction of these parameters is difficult. The **Crop Recommendation System** addresses this challenge by leveraging **Data Science and Machine Learning classification algorithms** trained on historical agricultural datasets to accurately identify the most suitable crop for a given set of conditions.

---

## 🎯 Objective

- To develop an end-to-end, educational **Data Science and Machine Learning pipeline** for agricultural crop recommendation.
- To compare multiple classification models (**Decision Tree**, **K-Nearest Neighbors**, and **Random Forest**) on multi-class crop datasets using standardized metrics (Accuracy, Precision, Recall, F1-Score).
- To demonstrate **Principal Component Analysis (PCA)** for high-dimensional data visualization and dimensionality reduction.
- To provide an interactive, responsive **Streamlit web application** for real-time user input and prediction.

---

## 💻 Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3** | Core programming language |
| **Pandas** | Data loading, manipulation, and statistical summarization |
| **NumPy** | High-performance numerical computations and array operations |
| **Scikit-learn** | Model training, classification algorithms, metric evaluation, and PCA |
| **Matplotlib & Seaborn** | Exploratory data analysis, heatmaps, and distribution plots |
| **Streamlit** | Modern, interactive Python web application framework |
| **Joblib** | Model serialization, pipeline persistence, and caching |

---

## 📊 Dataset Description

The system uses a balanced agricultural dataset consisting of **2,200 records** across **22 crop categories** (100 instances per crop).

### Input Features (7 Numerical Parameters)

1. **Nitrogen (N):** Ratio of Nitrogen content in soil ($kg/ha$)
2. **Phosphorus (P):** Ratio of Phosphorus content in soil ($kg/ha$)
3. **Potassium (K):** Ratio of Potassium content in soil ($kg/ha$)
4. **Temperature:** Ambient temperature in degrees Celsius ($^\circ C$)
5. **Humidity:** Relative atmospheric humidity percentage ($\%$)
6. **Soil pH:** Soil acidity or alkalinity ($0 - 14$ scale)
7. **Rainfall:** Depth of rainfall precipitation in millimeters ($mm$)

### Target Variable (Label)

- **22 Crop Categories:** `rice`, `maize`, `chickpea`, `kidneybeans`, `pigeonpeas`, `mothbeans`, `mungbean`, `blackgram`, `lentil`, `pomegranate`, `banana`, `mango`, `grapes`, `watermelon`, `muskmelon`, `apple`, `orange`, `papaya`, `coconut`, `cotton`, `jute`, `coffee`.

---

## 🤖 Machine Learning Algorithms

The project trains and compares three multi-class classification models:

1. **Decision Tree Classifier (`DecisionTreeClassifier`):**
   - A non-parametric supervised learning algorithm that creates tree-like decision rules.
   - Transparent, interpretable, and models non-linear relationships.

2. **K-Nearest Neighbors (`KNeighborsClassifier`):**
   - An instance-based (lazy) learning algorithm that classifies samples based on Euclidean distance in 7D feature space.
   - Non-parametric and simple to explain during viva examinations.

3. **Random Forest Classifier (`RandomForestClassifier`):**
   - An ensemble method combining multiple decision trees using bootstrap aggregating (bagging) and random feature selection.
   - High robustness against overfitting and produces reliable feature importance rankings.

### Model Evaluation & Selection

Models are evaluated using an **80/20 train/test split** with stratification:
- **Accuracy**
- **Precision (Weighted)**
- **Recall (Weighted)**
- **F1-Score (Weighted)**
- **Confusion Matrix & Classification Report**

> **Dynamic Selection:** The system automatically identifies and selects the best model based on evaluation results without hardcoding.

---

## 🔄 Data Science Process

```text
1. Data Collection
   └── CSV dataset loaded from dataset/crop_data.csv (2200 rows × 8 cols)
2. Data Cleaning
   └── Column name normalization, missing value verification, duplicate check
3. Exploratory Data Analysis (EDA)
   └── Class distribution, correlation heatmap, feature density & box plots
4. Preprocessing
   └── Separation into X and y, target LabelEncoder fitting, train/test split (80/20)
5. Model Training & Evaluation
   └── Fit Decision Tree, KNN, and Random Forest; compute accuracy, precision, recall, F1
6. Model Persistence
   └── Best model and label encoder serialized using Joblib (best_model.pkl)
7. Dimensionality Reduction (PCA)
   └── Standardize features with StandardScaler, reduce 7D to 2D for visualization
8. Web Deployment
   └── Streamlit interactive web application with user inputs and real-time recommendations
```

---

## 🔬 Principal Component Analysis (PCA)

In **Unit IV**, we demonstrate dimensionality reduction using **PCA**:

1. **Standardization:** All 7 features are scaled to zero mean and unit variance using `StandardScaler` so that features with large numerical magnitudes (e.g., Rainfall) do not disproportionately dominate.
2. **Eigen-Decomposition:** PCA projects the 7-dimensional observations onto 2 orthogonal Principal Components ($PC_1$ and $PC_2$).
3. **2D Visualization:** Visualizes the separation of crop clusters in reduced space.
4. **Educational Note:** Original 7 features are retained for final prediction to preserve practical interpretability and soil test measurement mapping.

---

## 🚀 How to Run Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Machine Learning Models

```bash
python train_model.py
```

*This will evaluate Decision Tree, KNN, and Random Forest, print performance metrics, select the top model, and generate artifacts in `model/` and `outputs/`.*

### 3. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

*The application will open in your default browser at `http://localhost:8501`.*

---

## 📁 Project Structure

```text
Crop-Recommendation-System/
│
├── app.py                      # Interactive Streamlit Web Application
├── train_model.py              # ML Training & Evaluation Pipeline
├── requirements.txt            # Python Dependencies
├── README.md                   # Complete Documentation
│
├── dataset/
│   └── crop_data.csv           # 2,200 record agricultural CSV dataset
│
├── model/
│   ├── best_model.pkl          # Serialized Best Classifier
│   ├── label_encoder.pkl       # Target Class Label Encoder
│   └── model_metadata.pkl      # Precomputed metrics and feature importances
│
├── notebooks/
│   └── crop_analysis.ipynb     # Jupyter Notebook for Step-by-Step EDA & Modeling
│
└── outputs/
    ├── model_comparison.csv    # Evaluated comparison metrics table
    └── metrics.txt             # Detailed classification report & confusion matrix summary
```

---

## ⚠️ Academic Disclaimer

This project is created for **academic and educational purposes** to demonstrate Data Science and Machine Learning concepts. Recommendations are data-driven predictions based on historical patterns and do not serve as a replacement for on-ground agricultural soil tests or regional agronomic expertise.
