import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import os

# Setup Paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root / "03_trained_model"))

# Import ModelInference
try:
    from model_inference import ModelInference
except ImportError:
    print("Error: Could not import ModelInference. Check paths.")
    sys.exit(1)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_shap_plot(model_version, output_path):
    print(f"Generating SHAP plot for {model_version}...")
    
    try:
        # Load Model
        inf = ModelInference(model_dir=str(project_root / "03_trained_model"), model_version=model_version)
        
        # Load Data
        data_path = project_root / "data/processed/kkbox_train_feature_v4.parquet"
        if not data_path.exists():
            print(f"Data not found at {data_path}")
            return
            
        # Sampling
        df = pd.read_parquet(data_path).sample(n=1000, random_state=42)
        
        # Preprocess
        X = inf.preprocess(df)
        
        # SHAP Calculation
        explainer = shap.TreeExplainer(inf.model)
        shap_values = explainer.shap_values(X)
        
        # Plotting
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, show=False, plot_size=(10, 8))
        plt.title(f"SHAP Summary: {model_version.upper()} Model", fontsize=16)
        plt.tight_layout()
        
        # Save
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot to {output_path}")
        
    except Exception as e:
        print(f"Error for {model_version}: {e}")

if __name__ == "__main__":
    ensure_dir(project_root / "images/shap")
    
    generate_shap_plot("v4", project_root / "images/shap/v4_shap_summary.png")
    generate_shap_plot("v5.2", project_root / "images/shap/v5_2_shap_summary.png")
