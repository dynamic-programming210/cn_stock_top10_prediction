"""
Streamlit App Entry Point for Streamlit Cloud
Minimal entry point with maximum error handling
"""
import streamlit as st

st.set_page_config(
    page_title="🇨🇳 Chinese Stock Top-10 Predictor",
    page_icon="🇨🇳",
    layout="wide"
)

import sys
import traceback
from pathlib import Path

# Show startup info
st.write("🚀 Starting app...")
st.write(f"Python version: {sys.version}")
st.write(f"Working directory: {Path.cwd()}")

try:
    # Add project root to path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    st.write(f"✅ Project root: {project_root}")
    
    # List files to verify deployment
    st.write("📁 Files in project root:")
    files = list(project_root.iterdir())[:15]
    for f in files:
        st.write(f"  - {f.name}")
    
    # Check data directory
    data_dir = project_root / "data"
    if data_dir.exists():
        st.write("📁 Files in data/:")
        for f in list(data_dir.iterdir())[:10]:
            size_mb = f.stat().st_size / 1024 / 1024 if f.is_file() else 0
            st.write(f"  - {f.name} ({size_mb:.2f} MB)")
    else:
        st.error("❌ data/ directory not found!")
    
    # Try importing config
    st.write("🔄 Importing config...")
    from config import TOP10_LATEST_FILE, OUTPUTS_DIR
    st.write(f"✅ Config imported. TOP10_LATEST_FILE exists: {TOP10_LATEST_FILE.exists()}")
    
    # Try importing main app
    st.write("🔄 Importing app.web...")
    from app.web import main
    st.write("✅ app.web imported successfully")
    
    # Run main app
    st.write("🔄 Running main()...")
    main()
    
except Exception as e:
    st.error(f"❌ Error: {type(e).__name__}: {str(e)}")
    st.code(traceback.format_exc())
