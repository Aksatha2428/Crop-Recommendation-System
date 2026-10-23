"""
=============================================================================
CROP RECOMMENDATION SYSTEM USING MACHINE LEARNING
A Data Science Based Approach for Crop Selection Using Soil and Environmental Parameters
Script: train_model.py
Description: Loads agricultural dataset, performs cleaning & EDA checks,
             trains 3 classification models (Decision Tree, KNN, Random Forest),
             evaluates metrics, selects the best model dynamically, and saves
             the trained model, label encoder, and comparison metrics.
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

# Ensure UTF-8 output encoding for Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def get_project_paths():
    """Determine relative project paths regardless of working directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Identify root_dir and crs_dir
    if os.path.basename(script_dir).lower() == "crop-recommendation-system":
        root_dir = os.path.dirname(script_dir)
        crs_dir = script_dir
    else:
        root_dir = script_dir
        crs_dir = os.path.join(script_dir, "Crop-Recommendation-System")
    
    possible_dataset_paths = [
        os.path.join(root_dir, "dataset", "crop_data.csv"),
        os.path.join(crs_dir, "dataset", "crop_data.csv"),
        os.path.join("dataset", "crop_data.csv"),
        os.path.join("Crop-Recommendation-System", "dataset", "crop_data.csv")
    ]
    
    dataset_path = None
    for path in possible_dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            break
            
    if dataset_path is None:
        dataset_path = os.path.join(root_dir, "dataset", "crop_data.csv")
        
    model_dir = os.path.join(root_dir, "model")
    output_dir = os.path.join(root_dir, "outputs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    crs_model_dir = os.path.join(crs_dir, "model")
    crs_output_dir = os.path.join(crs_dir, "outputs")
    os.makedirs(crs_model_dir, exist_ok=True)
    os.makedirs(crs_output_dir, exist_ok=True)
    
    return dataset_path, model_dir, output_dir, crs_model_dir, crs_output_dir


def load_and_preprocess_data(dataset_path):
    """
    Load the CSV dataset, validate columns, and perform data cleaning.
    """
    print("\n[Step 1] Loading Dataset...")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset file not found at: {dataset_path}. "
            "Please ensure dataset/crop_data.csv exists."
        )

    df = pd.read_csv(dataset_path)
    
    # Normalize column names: strip whitespace and check
    df.columns = [col.strip() for col in df.columns]
    
    # Expected columns
    expected_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    expected_target = 'label'
    
    # Case-insensitive column name alignment
    col_mapping = {}
    for col in df.columns:
        for exp in expected_features + [expected_target]:
            if col.lower() == exp.lower():
                col_mapping[col] = exp
    df = df.rename(columns=col_mapping)
    
    # Check for required columns
    missing_cols = [col for col in expected_features + [expected_target] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    print(f"-> Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"-> Features: {expected_features}")
    print(f"-> Target Variable: '{expected_target}' ({df[expected_target].nunique()} unique crops)")

    # Data Cleaning
    print("\n[Step 2] Data Understanding & Cleaning...")
    initial_rows = len(df)
    missing_count = df.isnull().sum().sum()
    print(f"-> Missing Values Found: {missing_count}")
    if missing_count > 0:
        print("   Handling missing values (dropping rows with null values)...")
        df = df.dropna()

    duplicate_count = df.duplicated().sum()
    print(f"-> Duplicate Rows Found: {duplicate_count}")
    if duplicate_count > 0:
        print("   Removing duplicate rows...")
        df = df.drop_duplicates()
        
    print(f"-> Cleaned Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    # Ensure numerical attributes are float/int
    for feature in expected_features:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
    df = df.dropna()

    return df, expected_features, expected_target


def train_and_evaluate_models(df, features, target_col):
    """
    Split data, train Decision Tree, KNN, and Random Forest,
    and calculate evaluation metrics.
    """
    print("\n[Step 3] Splitting Dataset (80% Train, 20% Test)...")
    X = df[features]
    y_raw = df[target_col]

    # Target label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"-> Training Set Size: {X_train.shape[0]} samples")
    print(f"-> Testing Set Size:  {X_test.shape[0]} samples")

    # Define the 3 required classification models
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    print("\n[Step 4] Training Classification Models & Evaluating Performance...")
    results = []
    trained_models = {}
    reports = {}
    matrices = {}

    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Predict on test set
        y_pred = model.predict(X_test)

        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        })

        # Save classification report and confusion matrix
        reports[name] = classification_report(
            y_test, y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )
        matrices[name] = confusion_matrix(y_test, y_pred)

        print(f"-> {name:<15} | Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1*100:.2f}%")

    # Create Comparison DataFrame
    comparison_df = pd.DataFrame(results)

    # Determine Best Model dynamically based on highest accuracy
    best_row = comparison_df.sort_values(by=["Accuracy", "F1 Score"], ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]

    print("\n==============================================================")
    print("                MODEL COMPARISON SUMMARY                      ")
    print("==============================================================")
    # Formatted display table
    display_df = comparison_df.copy()
    display_df["Accuracy"] = display_df["Accuracy"].apply(lambda v: f"{v*100:.2f}%")
    display_df["Precision"] = display_df["Precision"].apply(lambda v: f"{v*100:.2f}%")
    display_df["Recall"] = display_df["Recall"].apply(lambda v: f"{v*100:.2f}%")
    display_df["F1 Score"] = display_df["F1 Score"].apply(lambda v: f"{v*100:.2f}%")
    print(display_df.to_string(index=False))
    print("--------------------------------------------------------------")
    print(f"Best Performing Model: {best_model_name} (Accuracy: {best_row['Accuracy']*100:.2f}%)")
    print("==============================================================")

    # Calculate Feature Importance for Random Forest
    rf_feature_importances = {}
    if "Random Forest" in trained_models:
        rf_model = trained_models["Random Forest"]
        for feat, imp in zip(features, rf_model.feature_importances_):
            rf_feature_importances[feat] = float(imp)

    eval_data = {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "trained_models": trained_models,
        "comparison_df": comparison_df,
        "label_encoder": label_encoder,
        "reports": reports,
        "matrices": matrices,
        "rf_feature_importances": rf_feature_importances,
        "features": features,
        "X_test": X_test,
        "y_test": y_test
    }

    return eval_data


def save_artifacts(eval_data, model_dir, output_dir, crs_model_dir=None, crs_output_dir=None):
    """
    Save trained model, label encoder, comparison table, and metrics text.
    """
    print("\n[Step 5] Saving Artifacts...")
    
    dirs_to_save = [(model_dir, output_dir)]
    if crs_model_dir and crs_output_dir and crs_model_dir != model_dir:
        dirs_to_save.append((crs_model_dir, crs_output_dir))

    for m_dir, o_dir in dirs_to_save:
        best_model_path = os.path.join(m_dir, "best_model.pkl")
        label_encoder_path = os.path.join(m_dir, "label_encoder.pkl")
        metadata_path = os.path.join(m_dir, "model_metadata.pkl")
        comparison_csv_path = os.path.join(o_dir, "model_comparison.csv")
        metrics_txt_path = os.path.join(o_dir, "metrics.txt")

        # 1. Save Best Model
        joblib.dump(eval_data["best_model"], best_model_path)
        print(f"-> Saved Best Model:       {best_model_path}")

        # 2. Save Label Encoder
        joblib.dump(eval_data["label_encoder"], label_encoder_path)
        print(f"-> Saved Label Encoder:    {label_encoder_path}")

        # 3. Save Metadata (for fast Streamlit dashboard loading)
        metadata = {
            "best_model_name": eval_data["best_model_name"],
            "features": eval_data["features"],
            "comparison_df": eval_data["comparison_df"],
            "rf_feature_importances": eval_data["rf_feature_importances"],
            "reports": eval_data["reports"],
            "classes": eval_data["label_encoder"].classes_.tolist(),
            "matrices": eval_data["matrices"]
        }
        joblib.dump(metadata, metadata_path)
        print(f"-> Saved Model Metadata:   {metadata_path}")

        # 4. Save Model Comparison CSV
        eval_data["comparison_df"].to_csv(comparison_csv_path, index=False)
        print(f"-> Saved Comparison CSV:   {comparison_csv_path}")

        # 5. Save Metrics Text File
        with open(metrics_txt_path, "w", encoding="utf-8") as f:
            f.write("==============================================================\n")
            f.write("CROP RECOMMENDATION SYSTEM - MODEL EVALUATION METRICS REPORT\n")
            f.write("==============================================================\n\n")
            f.write("1. MODEL COMPARISON SUMMARY\n")
            f.write("--------------------------------------------------------------\n")
            f.write(eval_data["comparison_df"].to_string(index=False))
            f.write("\n\n")
            f.write(f"Selected Best Model: {eval_data['best_model_name']}\n\n")
            f.write("2. DETAILED CLASSIFICATION REPORTS\n")
            f.write("--------------------------------------------------------------\n")
            for m_name, report in eval_data["reports"].items():
                f.write(f"\n--- {m_name} Classification Report ---\n")
                f.write(report)
                f.write("\n")
            if eval_data["rf_feature_importances"]:
                f.write("\n3. RANDOM FOREST FEATURE IMPORTANCE\n")
                f.write("--------------------------------------------------------------\n")
                for feat, imp in eval_data["rf_feature_importances"].items():
                    f.write(f"{feat:<15}: {imp*100:.2f}%\n")

        print(f"-> Saved Metrics Report:   {metrics_txt_path}")

    print("\nAll models trained and artifacts generated successfully!\n")


def main():
    print("==============================================================")
    print("🌱 CROP RECOMMENDATION SYSTEM - MODEL TRAINING PIPELINE")
    print("   Data Science & Machine Learning Classification Engine      ")
    print("==============================================================")

    try:
        dataset_path, model_dir, output_dir, crs_m, crs_o = get_project_paths()
        df, features, target_col = load_and_preprocess_data(dataset_path)
        eval_data = train_and_evaluate_models(df, features, target_col)
        save_artifacts(eval_data, model_dir, output_dir, crs_m, crs_o)
        print("Done! You can now launch the Streamlit application using:")
        print("  streamlit run app.py")
    except Exception as e:
        print(f"\n[ERROR] Training failed: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
