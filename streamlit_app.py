"""
Streamlit App Entry Point - Minimal Test
"""
import streamlit as st

# Absolute minimum - just show hello world
st.title("🇨🇳 Chinese Stock Top-10 Predictor")
st.write("Hello World! App is running.")

# Show Python info
import sys
st.write(f"Python: {sys.version}")

# Check if we can list files
from pathlib import Path
project_root = Path(__file__).parent
st.write(f"Project root: {project_root}")

# List data files
data_dir = project_root / "data"
if data_dir.exists():
    st.write("Data files found:")
    for f in sorted(data_dir.iterdir()):
        if f.is_file():
            size_mb = f.stat().st_size / 1024 / 1024
            st.write(f"  - {f.name}: {size_mb:.2f} MB")
else:
    st.error("No data directory!")

# Try loading a parquet file
st.subheader("Loading predictions...")
try:
    import pandas as pd
    pred_file = project_root / "outputs" / "top10_latest.parquet"
    if pred_file.exists():
        df = pd.read_parquet(pred_file)
        st.write(f"Loaded {len(df)} predictions")
        st.dataframe(df)
    else:
        st.warning(f"File not found: {pred_file}")
except Exception as e:
    st.error(f"Error loading predictions: {e}")
