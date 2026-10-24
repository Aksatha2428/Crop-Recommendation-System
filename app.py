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
