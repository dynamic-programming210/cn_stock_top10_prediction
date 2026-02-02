"""
Streamlit App Entry Point for Streamlit Cloud
This file serves as an alternative entry point with better error handling
"""
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="🇨🇳 Chinese Stock Top-10 Predictor",
    page_icon="🇨🇳",
    layout="wide"
)

try:
    # Import and run the main app
    from app.web import main
    main()
except Exception as e:
    st.error(f"❌ Error loading application: {str(e)}")
    st.exception(e)
    
    # Show debug info
    st.markdown("---")
    st.subheader("🔍 Debug Information")
    
    st.write("**Python Path:**")
    st.code("\n".join(sys.path[:5]))
    
    st.write("**Project Files:**")
    project_root = Path(__file__).parent
    try:
        files = list(project_root.glob("**/*.py"))[:20]
        st.code("\n".join(str(f.relative_to(project_root)) for f in files))
    except Exception as fe:
        st.write(f"Error listing files: {fe}")
    
    st.write("**Data Files:**")
    data_dir = project_root / "data"
    if data_dir.exists():
        try:
            data_files = list(data_dir.glob("*"))
            for f in data_files:
                size = f.stat().st_size / 1024 / 1024 if f.is_file() else 0
                st.write(f"  {f.name}: {size:.2f} MB")
        except Exception as de:
            st.write(f"Error listing data: {de}")
    else:
        st.write("Data directory not found")
