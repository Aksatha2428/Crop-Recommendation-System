"""
=============================================================================
CROP RECOMMENDATION SYSTEM USING MACHINE LEARNING
A Data Science Based Approach for Crop Selection Using Soil and Environmental Parameters
Streamlit Web Application: app.py
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for academic aesthetic & responsive cards
st.markdown("""
<style>
    /* Main Theme Overrides */
    .main {
        background-color: #fcfdfc;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e4620;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        font-weight: 400;
        color: #3d6a40;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #d9e8d8;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-num {
        font-size: 1.8rem;
        font-weight: 700;
        color: #23692a;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555555;
        margin-top: 0.2rem;
    }
    .highlight-box {
        background: #f0f8f1;
        border-left: 5px solid #2e7d32;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
    .recommendation-banner {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 100%);
        color: #ffffff !important;
        padding: 2rem;
        border-radius: 14px;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 6px 16px rgba(46, 125, 50, 0.25);
    }
    .crop-name {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #ffffff;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background-color: #2e7d32;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        border: none;
        width: 100%;
        transition: background 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. PATH RESOLUTION & DATA/MODEL LOADERS
# -----------------------------------------------------------------------------
def resolve_paths():
    """Finds relative file locations regardless of working directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check dataset candidates
    candidates = [
        os.path.join(script_dir, "dataset", "crop_data.csv"),
        os.path.join("dataset", "crop_data.csv"),
        os.path.join("Crop-Recommendation-System", "dataset", "crop_data.csv"),
        os.path.join(script_dir, "Crop-Recommendation-System", "dataset", "crop_data.csv")
    ]
    dataset_path = None
    for c in candidates:
        if os.path.exists(c):
            dataset_path = c
            break
            
    # Model candidates
    model_dirs = [
        os.path.join(script_dir, "model"),
        os.path.join("model"),
        os.path.join("Crop-Recommendation-System", "model"),
        os.path.join(script_dir, "Crop-Recommendation-System", "model")
    ]
    model_dir = None
    for md in model_dirs:
        if os.path.exists(os.path.join(md, "best_model.pkl")):
            model_dir = md
            break
    if model_dir is None:
        model_dir = os.path.join(script_dir, "model")
        
    return dataset_path, model_dir


@st.cache_data
def load_dataset(dataset_path):
    """Loads and standardizes the agricultural dataset."""
    if not dataset_path or not os.path.exists(dataset_path):
        return None
    df = pd.read_csv(dataset_path)
    df.columns = [c.strip() for c in df.columns]
    
    expected_cols = {'n': 'N', 'p': 'P', 'k': 'K', 'temperature': 'temperature',
                     'humidity': 'humidity', 'ph': 'ph', 'rainfall': 'rainfall', 'label': 'label'}
    rename_dict = {}
    for col in df.columns:
        if col.lower() in expected_cols:
            rename_dict[col] = expected_cols[col.lower()]
    df = df.rename(columns=rename_dict)
    return df


@st.cache_resource
def load_models(model_dir):
    """Loads saved model, label encoder, and training metadata."""
    best_model_path = os.path.join(model_dir, "best_model.pkl")
    label_encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    metadata_path = os.path.join(model_dir, "model_metadata.pkl")

    if not (os.path.exists(best_model_path) and os.path.exists(label_encoder_path)):
        return None, None, None

    best_model = joblib.load(best_model_path)
    label_encoder = joblib.load(label_encoder_path)
    metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else None

    return best_model, label_encoder, metadata


# -----------------------------------------------------------------------------
# 3. HELPER: FALLBACK TRAINING IF NOT ALREADY RUN
# -----------------------------------------------------------------------------
def trigger_training(dataset_path, model_dir):
    """Runs model training if artifacts are missing."""
    import train_model
    with st.spinner("Training Decision Tree, KNN, and Random Forest models... Please wait."):
        df, features, target_col = train_model.load_and_preprocess_data(dataset_path)
        eval_data = train_model.train_and_evaluate_models(df, features, target_col)
        output_dir = os.path.join(os.path.dirname(model_dir), "outputs")
        train_model.save_artifacts(eval_data, model_dir, output_dir)
    st.success("Models successfully trained and saved!")
    st.cache_resource.clear()
    st.rerun()


# -----------------------------------------------------------------------------
# 4. INITIALIZE APP STATE
# -----------------------------------------------------------------------------
dataset_path, model_dir = resolve_paths()
df = load_dataset(dataset_path)
best_model, label_encoder, metadata = load_models(model_dir)

feature_list = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Crop details dictionary for educational insights
CROP_INFO = {
    "rice": {"desc": "Water-intensive staple crop thriving in hot and humid climates with heavy rainfall.", "season": "Kharif", "soil": "Clayey / Alluvial"},
    "maize": {"desc": "Versatile cereal crop requiring well-drained fertile loamy soils and warm weather.", "season": "Kharif / Rabi", "soil": "Well-drained Loam"},
    "chickpea": {"desc": "Important pulse crop thriving in cool, dry climate and moderate moisture.", "season": "Rabi", "soil": "Deep Sandy Loam"},
    "kidneybeans": {"desc": "Nutritious legume requiring moderate temperatures and balanced soil nutrients.", "season": "Kharif", "soil": "Loamy, Rich Soil"},
    "pigeonpeas": {"desc": "Drought-tolerant pulse crop that performs well in semi-arid tropical regions.", "season": "Kharif", "soil": "Black Cotton Soil"},
    "mothbeans": {"desc": "Extremely drought-hardy legume suitable for dry and arid sandy regions.", "season": "Kharif", "soil": "Sandy Loam"},
    "mungbean": {"desc": "Short-duration pulse crop that enriches soil through atmospheric nitrogen fixation.", "season": "Zaid / Kharif", "soil": "Loamy Sand"},
    "blackgram": {"desc": "Warm weather pulse crop requiring moderate rainfall and well-aerated soils.", "season": "Kharif", "soil": "Loamy to Clayey"},
    "lentil": {"desc": "Cool-season legume widely cultivated for nutritious protein-rich seeds.", "season": "Rabi", "soil": "Alluvial / Clay Loam"},
    "pomegranate": {"desc": "Fruit crop that tolerates dry heat and thrives in well-drained alkaline soils.", "season": "Perennial", "soil": "Deep Sandy Loam"},
    "banana": {"desc": "Heavy feeder requiring rich organic matter, warm climate, and ample moisture.", "season": "Year-round", "soil": "Fertile Clay Loam"},
    "mango": {"desc": "Tropical fruit tree requiring dry weather during flowering and warm summers.", "season": "Summer", "soil": "Well-drained Alluvial"},
    "grapes": {"desc": "High-value fruit vine demanding good drainage, warm dry summers, and mild winters.", "season": "Rabi", "soil": "Sandy / Gravelly Loam"},
    "watermelon": {"desc": "Warm-season fruit that thrives in hot climates with well-drained sandy soil.", "season": "Zaid (Summer)", "soil": "Sandy Loam"},
    "muskmelon": {"desc": "Sweet melon crop requiring warm, sunny weather and low humidity during ripening.", "season": "Zaid (Summer)", "soil": "Sandy Loam"},
    "apple": {"desc": "Temperate fruit crop requiring chilling winter temperatures and loamy mountain soil.", "season": "Temperate", "soil": "Rich Loam"},
    "orange": {"desc": "Citrus fruit requiring subtropical conditions, moderate moisture, and good aeration.", "season": "Winter / Spring", "soil": "Slightly Acidic Loam"},
    "papaya": {"desc": "Fast-growing tropical fruit requiring warm weather and porous rich soils.", "season": "Year-round", "soil": "Porous Loam"},
    "coconut": {"desc": "Coastal perennial palm thriving in humid tropical conditions and saline-tolerant sands.", "season": "Perennial", "soil": "Sandy / Coastal Alluvial"},
    "cotton": {"desc": "Commercial fiber crop requiring high heat, moderate rainfall, and deep black soils.", "season": "Kharif", "soil": "Black Cotton Soil"},
    "jute": {"desc": "Golden fiber crop requiring warm, humid climate and flooded alluvial river basins.", "season": "Kharif", "soil": "Alluvial Silt"},
    "coffee": {"desc": "Highland plantation crop requiring shade, mild humid climate, and fertile volcanic soil.", "season": "Perennial", "soil": "Organic Rich Loam"}
}


# -----------------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.markdown("""
<div style='text-align: center; padding: 1rem 0;'>
    <h2 style='color: #1e4620; margin: 0;'>🌱 Crop Recommender</h2>
    <p style='color: #4b7a4e; font-size: 0.85rem;'>Data Science & ML System</p>
</div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Navigation Menu",
    [
        "🌱 Crop Prediction",
        "🧠 How the Model Works",
        "📈 Model Performance",
        "📊 Dataset & EDA",
        "🔬 PCA Analysis"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 System Status")
if df is not None:
    st.sidebar.success(f"✓ Dataset Loaded ({len(df):,} rows)")
else:
    st.sidebar.error("✗ Dataset Not Found")

if best_model is not None:
    st.sidebar.success(f"✓ Trained Model Ready")
    if metadata and "best_model_name" in metadata:
        st.sidebar.info(f"Active Model: **{metadata['best_model_name']}**")
else:
    st.sidebar.warning("⚠ Model Not Trained")

st.sidebar.markdown("---")
# st.sidebar.markdown(
#     "<small style='color: #777;'>Academic Data Science Project<br>Demonstrating Units I, III, IV, & V</small>",
#     unsafe_allow_html=True
# )


# =============================================================================
# PAGE 1: HOME
# =============================================================================
if menu == "🧠 How the Model Works":
    st.markdown("<div class='main-header'>🧠 HOW THE MODEL WORKS</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Understand the dataset, training workflow, algorithms, and how a crop prediction is generated.</div>", unsafe_allow_html=True)

    # KPI Summary Cards (values are derived from the loaded dataset where possible)
    record_count = f"{len(df):,}" if df is not None else "N/A"
    crop_count = str(df["label"].nunique()) if df is not None and "label" in df.columns else "N/A"
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-num'>{record_count}</div>
            <div class='metric-label'>Agricultural Records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-num'>7</div>
            <div class='metric-label'>Input Parameters (N, P, K, etc.)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-num'>{crop_count}</div>
            <div class='metric-label'>Crop Class Categories</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-num'>3</div>
            <div class='metric-label'>ML Models (DT, KNN, RF)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Problem Statement & Objective
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.markdown("### 🎯 Project Problem Statement")
        st.write(
            "Selecting a suitable crop depends heavily on various soil nutrients and environmental conditions. "
            "Farmers need to analyze complex, interrelated parameters such as **Nitrogen (N)**, **Phosphorus (P)**, "
            "**Potassium (K)**, **Temperature**, **Humidity**, **Soil pH**, and **Rainfall**. "
            "Manually analyzing these parameters and identifying the most suitable crop can be challenging and prone to error."
        )
        st.write(
            "This system applies **Data Science and Machine Learning classification algorithms** to historical agricultural data. "
            "By learning non-linear multidimensional decision boundaries, the system recommends the most suitable crop for any "
            "given combination of soil and climate parameters."
        )
        
        st.markdown("""
        <div class='highlight-box'>
            <b>Academic Objective:</b> To demonstrate end-to-end Data Science workflows including exploratory data analysis,
            data cleaning, multi-class model training, hyperparameter evaluation, dimensionality reduction (PCA),
            and interactive model deployment.
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("### 📋 Features & Parameters")
        params_data = {
            "Parameter": ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)", "Temperature", "Humidity", "Soil pH", "Rainfall"],
            "Type": ["Nutrient", "Nutrient", "Nutrient", "Climate", "Climate", "Soil", "Climate"],
            "Unit / Scale": ["Ratio / kg/ha", "Ratio / kg/ha", "Ratio / kg/ha", "°Celsius", "% Relative", "pH (0 - 14)", "mm (Depth)"]
        }
        st.dataframe(pd.DataFrame(params_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    
    # Workflow Architecture
    st.markdown("### 🔄 Data Science & Machine Learning Workflow")
    st.markdown("""
    ```text
    ┌────────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
    │ Agricultural   │  ──>  │ Data Cleaning   │  ──>  │ Exploratory     │  ──>  │ Train/Test      │
    │ CSV Dataset    │       │ & Preprocessing │       │ Data Analysis   │       │ Split (80/20)   │
    └────────────────┘       └─────────────────┘       └─────────────────┘       └─────────────────┘
                                                                                          │
                                                                                          ▼
    ┌────────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
    │ Streamlit Web  │  <──  │ Best Model      │  <──  │ Metric Eval     │  <──  │ 3 Classifiers:  │
    │ Recommendation │       │ Selection (Auto)│       │ (Acc, Prec, F1) │       │ DT, KNN, RF     │
    └────────────────┘       └─────────────────┘       └─────────────────┘       └─────────────────┘
    ```
    """)

    st.markdown("---")
    
    # Technology Stack
    st.markdown("### 💻 Technologies Used")
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    with tech_col1:
        st.markdown("**Core Python & Data Science:**")
        st.markdown("- **Python 3**: Primary programming language")
        st.markdown("- **Pandas**: Data loading, cleaning, manipulation")
        st.markdown("- **NumPy**: Numerical matrix computations")
    with tech_col2:
        st.markdown("**Machine Learning & Evaluation:**")
        st.markdown("- **Scikit-learn**: Decision Tree, KNN, Random Forest")
        st.markdown("- **Dimensionality Reduction**: StandardScaler & PCA")
        st.markdown("- **Joblib**: Model serialization & persistence")
    with tech_col3:
        st.markdown("**Visualization & Web UI:**")
        st.markdown("- **Matplotlib & Seaborn**: Statistical plots & heatmaps")
        st.markdown("- **Streamlit**: Interactive web application framework")

    # st.markdown("""
    # <div style='background-color: #fff9db; border-left: 4px solid #f59f00; padding: 0.8rem 1rem; border-radius: 4px; margin-top: 1.5rem;'>
    #     <b>⚠️ Educational Disclaimer:</b> This project is built as an educational Data Science demonstration.
    #     Predictions are data-driven statistical recommendations based on historical patterns and do not constitute
    #     a formal agricultural or economic guarantee of crop success.
    # </div>
    # """, unsafe_allow_html=True)


# =============================================================================
# PAGE 2: DATASET & EDA
# =============================================================================
elif menu == "📊 Dataset & EDA":
    st.markdown("<div class='main-header'>📊 Dataset & Exploratory Data Analysis</div>", unsafe_allow_html=True)
    # st.markdown("<div class='sub-header'>Unit I: Data Understanding, Cleaning, and Statistical Summarization</div>", unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not found at `dataset/crop_data.csv`. Please ensure the file exists.")
        st.stop()

    # Dataset Overview Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{df.shape[0]:,}")
    c2.metric("Total Features", f"{df.shape[1] - 1}")
    c3.metric("Crop Classes", f"{df['label'].nunique()}")
    c4.metric("Missing Values", f"{df.isnull().sum().sum()}")

    st.markdown("### 1. Dataset Preview (First 10 Rows)")
    st.dataframe(df.head(10), use_container_width=True)

    # Statistical Summary
    st.markdown("### 2. Statistical Summary (`df.describe()`)")
    st.write("Measures of central tendency, dispersion, and quartiles for all numerical attributes:")
    st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)

    # Attribute Types Table
    with st.expander("🔍 View Attribute Types & Objects"):
        attr_data = [
            {"Attribute": "N", "Type": "Numeric (Continuous)", "Description": "Ratio of Nitrogen content in soil (kg/ha)"},
            {"Attribute": "P", "Type": "Numeric (Continuous)", "Description": "Ratio of Phosphorous content in soil (kg/ha)"},
            {"Attribute": "K", "Type": "Numeric (Continuous)", "Description": "Ratio of Potassium content in soil (kg/ha)"},
            {"Attribute": "temperature", "Type": "Numeric (Continuous)", "Description": "Temperature in degree Celsius (°C)"},
            {"Attribute": "humidity", "Type": "Numeric (Continuous)", "Description": "Relative humidity percentage (%)"},
            {"Attribute": "ph", "Type": "Numeric (Continuous)", "Description": "Soil pH value (acidity/alkalinity scale 0-14)"},
            {"Attribute": "rainfall", "Type": "Numeric (Continuous)", "Description": "Rainfall precipitation in millimeters (mm)"},
            {"Attribute": "label", "Type": "Categorical (Nominal Target)", "Description": "Agricultural crop category (22 unique classes)"}
        ]
        st.dataframe(pd.DataFrame(attr_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 3. Exploratory Data Visualizations (Seaborn & Matplotlib)")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Visualization 1: Crop Distribution",
        "Visualization 2: Correlation Heatmap",
        "Visualization 3: Feature Distributions",
        "Visualization 4: Bivariate Scatter Analysis"
    ])

    # Viz 1: Crop Distribution
    with tab1:
        st.markdown("#### Crop Class Distribution (Record Count per Crop)")
        crop_counts = df['label'].value_counts()
        fig, ax = plt.subplots(figsize=(10, 4.5))
        sns.barplot(x=crop_counts.index, y=crop_counts.values, ax=ax, palette="crest")
        ax.set_title("Number of Historical Records per Crop Class", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Crop Name", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        st.pyplot(fig)
        st.caption("Use the bar chart to inspect whether crop classes are balanced. The actual counts depend on the loaded dataset.")

    # Viz 2: Correlation Heatmap
    with tab2:
        st.markdown("#### Correlation Heatmap of Soil & Climate Features")
        numeric_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, cbar=True, square=True, linewidths=0.5)
        ax.set_title("Pearson Correlation Matrix Between Numerical Features", fontsize=12, fontweight='bold', pad=12)
        st.pyplot(fig)
        st.caption("Use the heatmap to inspect the direction and strength of linear relationships between numerical features. Correlation does not imply causation.")

    # Viz 3: Feature Distributions
    with tab3:
        st.markdown("#### Feature Distribution (Histograms & Density)")
        selected_feature = st.selectbox("Select Feature to Inspect:", numeric_cols, index=0)
        
        fig, (ax_hist, ax_box) = plt.subplots(2, 1, figsize=(8, 5), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        sns.histplot(df[selected_feature], kde=True, ax=ax_hist, color="#2e7d32", bins=25)
        ax_hist.set_title(f"Distribution & Spread of {selected_feature}", fontsize=12, fontweight='bold')
        ax_hist.set_ylabel("Frequency", fontsize=10)
        ax_hist.grid(axis='y', linestyle='--', alpha=0.5)
        
        sns.boxplot(x=df[selected_feature], ax=ax_box, color="#81c784")
        ax_box.set_xlabel(selected_feature, fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)

    # Viz 4: Bivariate Scatter Plot
    with tab4:
        st.markdown("#### Bivariate Feature Relationship")
        col_x, col_y = st.columns(2)
        with col_x:
            feat_x = st.selectbox("X-Axis Feature:", numeric_cols, index=0)
        with col_y:
            feat_y = st.selectbox("Y-Axis Feature:", numeric_cols, index=1)

        fig, ax = plt.subplots(figsize=(9, 5))
        # Select a representative subset of crops to prevent overwhelming colors
        sample_crops = ['rice', 'maize', 'coffee', 'cotton', 'apple', 'banana', 'grapes']
        subset_df = df[df['label'].isin(sample_crops)]
        sns.scatterplot(data=subset_df, x=feat_x, y=feat_y, hue='label', ax=ax, palette='tab10', s=55, alpha=0.85)
        ax.set_title(f"Scatter Plot: {feat_x} vs {feat_y} (Sample Crops)", fontsize=12, fontweight='bold', pad=12)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(title="Crop Class", bbox_to_anchor=(1.02, 1), loc='upper left')
        st.pyplot(fig)
        st.caption("Different crops occupy distinct clusters in feature space, enabling machine learning classifiers to establish clear decision boundaries.")


# =============================================================================
# PAGE 3: MODEL PERFORMANCE
# =============================================================================
elif menu == "📈 Model Performance":
    st.markdown("<div class='main-header'>📈 Model Evaluation & Comparison</div>", unsafe_allow_html=True)
    # st.markdown("<div class='sub-header'>Unit III: Classification using Decision Tree, K-Nearest Neighbors, and Random Forest</div>", unsafe_allow_html=True)

    if best_model is None or metadata is None:
        st.warning("⚠️ Trained model artifacts not found. You can train the models directly using the button below.")
        if st.button("🚀 Train Models Now (Decision Tree, KNN, Random Forest)"):
            trigger_training(dataset_path, model_dir)
        st.stop()

    comp_df = metadata.get("comparison_df")
    best_name = metadata.get("best_model_name", "Random Forest")

    # Best Model Highlight Banner
    best_row = comp_df[comp_df["Model"] == best_name].iloc[0]
    best_acc = best_row["Accuracy"] * 100

    st.markdown(f"""
    <div class='recommendation-banner'>
        <div style='font-size: 1.1rem; opacity: 0.9;'>🏆 SELECTED BEST PERFORMING MODEL</div>
        <div class='crop-name'>{best_name}</div>
        <div style='font-size: 1.3rem; font-weight: 600;'>Test Accuracy: {best_acc:.2f}%</div>
        <div style='font-size: 0.9rem; opacity: 0.85; margin-top: 0.5rem;'>
            Automatically selected based on highest test accuracy and weighted F1-score across all 3 evaluated classifiers.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model Comparison Table
    st.markdown("### 1. Classification Metrics Comparison Table")
    formatted_comp = comp_df.copy()
    formatted_comp["Accuracy"] = formatted_comp["Accuracy"].apply(lambda v: f"{v*100:.2f}%")
    formatted_comp["Precision"] = formatted_comp["Precision"].apply(lambda v: f"{v*100:.2f}%")
    formatted_comp["Recall"] = formatted_comp["Recall"].apply(lambda v: f"{v*100:.2f}%")
    formatted_comp["F1 Score"] = formatted_comp["F1 Score"].apply(lambda v: f"{v*100:.2f}%")
    st.dataframe(formatted_comp, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Visual Comparison Bar Chart
    st.markdown("### 2. Multi-Metric Model Comparison Chart")
    
    # Melt comparison dataframe for grouped bar plotting
    melted = comp_df.melt(id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
                          var_name="Metric", value_name="Score")
    melted["Score (%)"] = melted["Score"] * 100

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=melted, x="Model", y="Score (%)", hue="Metric", ax=ax, palette="Greens_d")
    ax.set_ylim(80, 102)
    ax.set_title("Side-by-Side Model Evaluation (Weighted Multi-Class Metrics)", fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel("Score (%)", fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')
    st.pyplot(fig)

    st.markdown("---")

    # Feature Importance (Random Forest)
    st.markdown("### 3. Feature Importance Analysis (Random Forest)")
    rf_imp = metadata.get("rf_feature_importances", {})
    if rf_imp:
        imp_df = pd.DataFrame(list(rf_imp.items()), columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=False)
        imp_df["Importance (%)"] = imp_df["Importance"] * 100

        col_imp_chart, col_imp_expl = st.columns([1.3, 1])
        with col_imp_chart:
            fig_imp, ax_imp = plt.subplots(figsize=(7, 4))
            sns.barplot(data=imp_df, x="Importance (%)", y="Feature", ax=ax_imp, palette="crest_r")
            ax_imp.set_title("Relative Feature Importance in Crop Selection", fontsize=11, fontweight='bold')
            ax_imp.grid(axis='x', linestyle='--', alpha=0.5)
            st.pyplot(fig_imp)
        with col_imp_expl:
            st.markdown("#### 💡 Understanding Feature Importance")
            st.write(
                "Feature importance in Random Forest measures the total reduction in **Gini Impurity** brought by each feature "
                "across all decision trees in the ensemble."
            )
            st.write(
                "**Key Takeaways:**"
            )
            st.markdown("- **Rainfall and Nutrients (N, P, K)** generally serve as primary split criteria for crops.")
            st.markdown("- **Educational Note:** Feature importance indicates model utility, not scientific causation.")

    st.markdown("---")

    # Confusion Matrix
    st.markdown("### 4. Confusion Matrix (Best Model)")
    matrices = metadata.get("matrices", {})
    classes = metadata.get("classes", [])
    
    if best_name in matrices and classes:
        matrix = matrices[best_name]
        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGn", xticklabels=classes, yticklabels=classes, ax=ax_cm, cbar=False)
        ax_cm.set_title(f"Confusion Matrix for {best_name}", fontsize=12, fontweight='bold', pad=12)
        ax_cm.set_xlabel("Predicted Crop", fontsize=10)
        ax_cm.set_ylabel("True Crop", fontsize=10)
        ax_cm.tick_params(axis='x', rotation=45)
        ax_cm.tick_params(axis='y', rotation=0)
        st.pyplot(fig_cm)

    # Detailed Classification Report
    with st.expander("📄 View Full Classification Report (Per-Class Precision, Recall, F1)"):
        reports = metadata.get("reports", {})
        if best_name in reports:
            st.code(reports[best_name])


# =============================================================================
# PAGE 4: PCA ANALYSIS
# =============================================================================
elif menu == "🔬 PCA Analysis":
    st.markdown("<div class='main-header'>🔬 Principal Component Analysis (PCA)</div>", unsafe_allow_html=True)
    # st.markdown("<div class='sub-header'>Unit IV: Dimensionality Reduction for High-Dimensional Agricultural Data</div>", unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not found at `dataset/crop_data.csv`.")
        st.stop()

    st.markdown("""
    <div class='highlight-box'>
        # <b>Unit IV Concept — Dimensionality Reduction:</b><br>
        <i>"PCA is used here for visualization and dimensionality-reduction analysis.
        The original features are retained for the main crop prediction model so that the prediction system
        remains simple and interpretable."</i>
    </div>
    """, unsafe_allow_html=True)

    # Educational Explanation
    col_pca_text1, col_pca_text2 = st.columns(2)
    with col_pca_text1:
        st.markdown("### 1. Why Standardize First?")
        st.write(
            "Principal Component Analysis calculates eigenvectors of the covariance matrix. Features with larger variance "
            "or larger scales (such as Rainfall in mm vs Soil pH on a scale of 0-14) would artificially dominate the principal components."
        )
        st.write(
            "We apply `StandardScaler` to transform all 7 features to **zero mean and unit variance** ($\mu=0, \sigma=1$):"
        )
        st.latex(r"z = \frac{x - \mu}{\sigma}")

    with col_pca_text2:
        st.markdown("### 2. Dimensionality Reduction Steps")
        st.markdown("1. **Feature Extraction:** Select 7 numerical features ($N, P, K, temp, humidity, ph, rainfall$).")
        st.markdown("2. **Standardization:** Normalize variables with `StandardScaler()`.")
        st.markdown("3. **Eigen-Decomposition:** Compute principal directions capturing maximal variance.")
        st.markdown("4. **Projection:** Project 7-dimensional observations onto 2 orthogonal axes ($PC_1$ and $PC_2$).")

    st.markdown("---")

    # Perform PCA Computation
    X_num = df[feature_list]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_

    # PCA Metrics
    col_var1, col_var2, col_var3 = st.columns(3)
    col_var1.metric("PC1 Explained Variance", f"{var_ratio[0]*100:.2f}%")
    col_var2.metric("PC2 Explained Variance", f"{var_ratio[1]*100:.2f}%")
    col_var3.metric("Cumulative 2D Variance", f"{(var_ratio[0] + var_ratio[1])*100:.2f}%")

    # 2D PCA Scatter Plot
    st.markdown("### 3. 2D Principal Component Projection")
    
    pca_df = pd.DataFrame(data=X_pca, columns=["PC1", "PC2"])
    pca_df["Crop"] = df["label"].values

    # Filter by user selection for clarity
    all_crops = sorted(df["label"].unique().tolist())
    selected_crops = st.multiselect(
        "Filter crops to display in PCA plot (Default displays sample subset for visual clarity):",
        all_crops,
        default=["rice", "maize", "chickpea", "coffee", "cotton", "apple", "banana", "grapes"]
    )

    filtered_pca = pca_df[pca_df["Crop"].isin(selected_crops)] if selected_crops else pca_df

    fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=filtered_pca,
        x="PC1",
        y="PC2",
        hue="Crop",
        palette="tab10",
        s=60,
        alpha=0.85,
        ax=ax_pca
    )
    ax_pca.set_title(
        f"2D PCA Projection of Agricultural Features (Explained Variance: {(var_ratio[0]+var_ratio[1])*100:.2f}%)",
        fontsize=12,
        fontweight='bold',
        pad=12
    )
    ax_pca.set_xlabel(f"Principal Component 1 ({var_ratio[0]*100:.2f}% Variance)", fontsize=10)
    ax_pca.set_ylabel(f"Principal Component 2 ({var_ratio[1]*100:.2f}% Variance)", fontsize=10)
    ax_pca.grid(True, linestyle='--', alpha=0.4)
    ax_pca.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Crop Label")
    st.pyplot(fig_pca)

    st.caption(
        "Notice how distinct crop varieties naturally separate in reduced 2D space, demonstrating that the original "
        "7 soil and climate features contain rich discriminative information for machine learning classification."
    )


# =============================================================================
# PAGE 5: CROP PREDICTION
# =============================================================================
elif menu == "🌱 Crop Prediction":
    st.markdown("<div class='main-header'>🌱 Crop Recommendation & Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Interactive Recommendation Engine Powered by Trained Machine Learning</div>", unsafe_allow_html=True)

    if best_model is None or label_encoder is None:
        st.warning("⚠️ Trained model not found. Please train models first:")
        if st.button("🚀 Train Model Now"):
            trigger_training(dataset_path, model_dir)
        st.stop()

    st.write(
        "Enter the soil chemical properties and local environmental/climate parameters below. "
        "The trained machine learning classifier will analyze these inputs and predict the most suitable crop."
    )

    # Input Form
    with st.form("crop_prediction_form"):
        col_soil, col_env = st.columns(2)

        with col_soil:
            st.markdown("#### 🧪 Soil Nutrient Parameters")
            n_val = st.number_input("Nitrogen (N) - kg/ha", min_value=0.0, max_value=150.0, value=90.0, step=1.0,
                                    help="Nitrogen content in the soil (typical range: 0 - 140)")
            p_val = st.number_input("Phosphorus (P) - kg/ha", min_value=5.0, max_value=150.0, value=42.0, step=1.0,
                                    help="Phosphorus content in the soil (typical range: 5 - 145)")
            k_val = st.number_input("Potassium (K) - kg/ha", min_value=5.0, max_value=210.0, value=43.0, step=1.0,
                                    help="Potassium content in the soil (typical range: 5 - 205)")
            ph_val = st.number_input("Soil pH Level", min_value=3.5, max_value=10.0, value=6.5, step=0.1,
                                     help="Soil acidity or alkalinity (typical agricultural range: 4.5 - 8.5)")

        with col_env:
            st.markdown("#### ⛅ Environmental & Climate Parameters")
            temp_val = st.number_input("Temperature (°C)", min_value=5.0, max_value=50.0, value=20.9, step=0.1,
                                       help="Ambient temperature in degrees Celsius (typical range: 8 - 45°C)")
            hum_val = st.number_input("Relative Humidity (%)", min_value=10.0, max_value=100.0, value=82.0, step=0.5,
                                      help="Atmospheric relative humidity percentage (typical range: 15 - 100%)")
            rain_val = st.number_input("Rainfall (mm)", min_value=10.0, max_value=350.0, value=202.9, step=1.0,
                                       help="Annual or seasonal rainfall in millimeters (typical range: 20 - 300 mm)")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🌱 Recommend Suitable Crop")

    # Prediction Processing
    if submitted:
        # Input Validation & Sanity Warnings
        warnings = []
        if ph_val < 4.5:
            warnings.append("Soil pH is strongly acidic (< 4.5). Most crops prefer slightly acidic to neutral soils (5.5 - 7.5).")
        elif ph_val > 8.5:
            warnings.append("Soil pH is strongly alkaline (> 8.5), which can restrict nutrient absorption.")

        if rain_val < 30.0:
            warnings.append("Rainfall is exceptionally low (< 30 mm). Supplemental irrigation will be necessary.")
        elif rain_val > 280.0:
            warnings.append("Rainfall is very high (> 280 mm). Ensure adequate drainage to avoid waterlogging.")

        for w in warnings:
            st.warning(f"⚠️ **Agricultural Note:** {w}")

        # Prepare feature vector with explicit feature names
        input_data = pd.DataFrame(
            [[n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val]],
            columns=feature_list
        )
        
        # Predict class
        pred_encoded = best_model.predict(input_data)[0]
        crop_recommended = label_encoder.inverse_transform([pred_encoded])[0]
        crop_clean = crop_recommended.strip().lower()

        # Display Prominent Recommendation Banner
        st.markdown(f"""
        <div class='recommendation-banner'>
            <div style='font-size: 1.1rem; opacity: 0.9;'>🌱 RECOMMENDED CROP</div>
            <div class='crop-name'>{crop_recommended.upper()}</div>
            <div style='font-size: 1.1rem; opacity: 0.9;'>
                Based on the provided soil and environmental conditions, the trained machine learning model recommends this crop.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Detailed Agronomic Insights
        col_rec_info, col_rec_prob = st.columns([1.2, 1])

        with col_rec_info:
            st.markdown("### 🌾 Crop Agronomic Profile")
            info = CROP_INFO.get(crop_clean, {
                "desc": "Agricultural crop suitable for the input soil and climatic parameters.",
                "season": "Seasonal",
                "soil": "Fertile Loamy Soil"
            })
            st.markdown(f"**Description:** {info['desc']}")
            st.markdown(f"**Preferred Cropping Season:** {info['season']}")
            st.markdown(f"**Typical Soil Requirement:** {info['soil']}")

        with col_rec_prob:
            st.markdown("### 📊 Prediction Confidence")
            if hasattr(best_model, "predict_proba"):
                probs = best_model.predict_proba(input_data)[0]
                top_indices = np.argsort(probs)[::-1][:3]
                
                top_crops = label_encoder.inverse_transform(top_indices)
                top_scores = probs[top_indices]

                for crop, score in zip(top_crops, top_scores):
                    st.write(f"**{crop.capitalize()}**: `{score*100:.1f}%`")
                    st.progress(float(score))
            else:
                st.info("Direct prediction from Decision Tree classifier.")

        # st.markdown("""
        # <div style='background-color: #f1f8e9; border: 1px solid #c8e6c9; border-radius: 8px; padding: 1rem; margin-top: 1.5rem;'>
        #     <b>⚠️ Important Notice:</b> This is an educational Data Science recommendation derived from historical data.
        #     Local factors such as soil micronutrients, irrigation availability, and regional weather variations should always
        #     be consulted with local agricultural experts before planting.
        # </div>
        # """, unsafe_allow_html=True)
