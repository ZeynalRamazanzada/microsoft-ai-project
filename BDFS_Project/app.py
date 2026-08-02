"""
BDFS — Streamlit Dashboard (app.py)
==================================
BDFS projesinin makine öğrenimi sonuçlarını ve görsellerini 
interaktif bir şekilde gösteren web tabanlı arayüz.
"""

import streamlit as st
import pandas as pd
import os

# Sayfa yapılandırması
st.set_page_config(
    page_title="BDFS ML Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Başlık ve Yazar Bilgisi
st.title("Behavioral Decision Fatigue Scoring (BDFS) 🧠")
st.markdown("### A Machine Learning Pipeline for Fatigue Detection")
st.markdown("**Authors:** Zeynalabdin Ramazanzada (231805121) & Alperen Sümeroğlu (231805023)")
st.markdown("---")

# Veri yükleme fonksiyonu
@st.cache_data
def load_data():
    try:
        metrics_df = pd.read_csv("results/model_evaluation_metrics.csv")
        shap_df = pd.read_csv("results/shap_feature_importance.csv")
        return metrics_df, shap_df
    except FileNotFoundError:
        st.error("Gerekli CSV dosyaları (results/) bulunamadı. Lütfen önce run_all.py'yi çalıştırın.")
        return pd.DataFrame(), pd.DataFrame()

metrics_df, shap_df = load_data()

# ----------------- YAN PANEL (SIDEBAR) -----------------
st.sidebar.header("Navigation & Settings")

if not metrics_df.empty:
    model_names = metrics_df['Model'].tolist()
    selected_model = st.sidebar.selectbox("Select a Model to View Metrics:", model_names)
else:
    selected_model = "None"

st.sidebar.markdown("---")
st.sidebar.info("This dashboard visualizes the results from the BDFS pipeline, comparing 5 machine learning models on 150,000 synthetic records.")

# ----------------- ANA EKRAN -----------------

if not metrics_df.empty and selected_model != "None":
    # 1. Model Metrikleri Kartları
    st.header(f"Model Performance: {selected_model}")
    
    # Seçilen modelin verilerini al
    model_data = metrics_df[metrics_df['Model'] == selected_model].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Accuracy", value=f"{model_data['Accuracy']:.4f}")
    with col2:
        st.metric(label="F1-Score", value=f"{model_data['F1-Score']:.4f}")
    with col3:
        st.metric(label="ROC-AUC", value=f"{model_data['ROC-AUC']:.4f}")
    with col4:
        st.metric(label="PR-AUC", value=f"{model_data['PR-AUC']:.4f}")
        
    # Modelin Confusion Matrix'i
    st.subheader(f"{selected_model} Confusion Matrix")
    cm_path = os.path.join("figures", f"{selected_model.lower()}_confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(cm_path, width=400)
    else:
        st.warning(f"Resim bulunamadı: {cm_path}")
        
    st.markdown("---")
    
# 2. Genel Grafik Karşılaştırmaları
st.header("Global Project Results")

tab1, tab2, tab3 = st.tabs(["ROC Curves", "SHAP Feature Importance", "Ablation Study"])

with tab1:
    st.subheader("Model Comparison: ROC Curves")
    roc_path = os.path.join("figures", "roc_curves_comparison.png")
    if os.path.exists(roc_path):
        st.image(roc_path, use_container_width=True)
    else:
        st.warning("ROC Curve resmi bulunamadı.")

with tab2:
    st.subheader("SHAP Interpretability: Top 10 Features (XGBoost)")
    col_img, col_data = st.columns([2, 1])
    
    with col_img:
        shap_path = os.path.join("figures", "09_shap_summary.png")
        if os.path.exists(shap_path):
            st.image(shap_path, use_container_width=True)
        else:
            st.warning("SHAP Summary resmi bulunamadı.")
            
    with col_data:
        if not shap_df.empty:
            st.dataframe(shap_df.head(10), hide_index=True)
            
            # Bar chart for Top 10
            st.bar_chart(data=shap_df.head(10), x="Feature", y="SHAP_Importance", color="#ff4b4b")
            
with tab3:
    st.subheader("Ablation Study: Feature Set Contribution")
    ablation_path = os.path.join("figures", "11_ablation_study.png")
    if os.path.exists(ablation_path):
        st.image(ablation_path, use_container_width=True)
    else:
        st.warning("Ablation Study resmi bulunamadı.")
