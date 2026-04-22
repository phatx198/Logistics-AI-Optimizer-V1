import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Logistics AI Dashboard", layout="wide")

st.title("Logistics Cost & Carbon Hybrid Optimizer")
st.markdown("Independent Research Prototype: RF-Fuzzy-MOO Model")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("User Parameters")
uploaded_file = st.sidebar.file_exists = st.sidebar.file_uploader("Upload Logistics Excel", type=["xlsx"])
cost_weight = st.sidebar.slider("Cost Priority (W)", 0.0, 1.0, 0.5)
carbon_weight = 1.0 - cost_weight

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    # Model Logic (Condensed)
    X = df[['Trip_Length', 'Cargo_Weight', 'Total_Time']]
    rf_cost = RandomForestRegressor(n_estimators=100).fit(X, df['Total_Cost'])
    rf_carb = RandomForestRegressor(n_estimators=100).fit(X, df['Carbon_Emission'])
    
    pred_c = rf_cost.predict(X)
    pred_g = rf_carb.predict(X)
    
    # Fuzzy Uncertainty
    tree_preds = np.array([t.predict(X.values) for t in rf_cost.estimators_])
    spread = np.std(tree_preds, axis=0) * 1.5 # Fixed multiplier for app
    
    # Multi-Objective Utility
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(np.column_stack([pred_c, pred_g]))
    utility = (cost_weight * norm[:,0]) + (carbon_weight * norm[:,1])

    # --- DISPLAY RESULTS ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fuzzy Cost Prediction")
        fig1, ax1 = plt.subplots()
        ax1.fill_between(range(len(df)), pred_c-spread, pred_c+spread, alpha=0.2)
        ax1.plot(pred_c, label="Prediction")
        ax1.scatter(range(len(df)), df['Total_Cost'], color='black', s=5)
        st.pyplot(fig1)

    with col2:
        st.subheader("Decision Utility Score")
        fig2, ax2 = plt.subplots()
        ax2.plot(utility, color='green', label=f'W={cost_weight}')
        st.pyplot(fig2)
        
    st.success(f"Current Policy Balance: {cost_weight*100}% Cost / {carbon_weight*100}% Green")
else:
    st.info("Please upload your Logistics_Data.xlsx file to start.")