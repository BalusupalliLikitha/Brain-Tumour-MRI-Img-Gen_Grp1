import streamlit as st
import plotly.express as px
import pandas as pd


def show_dashboard():
    st.title(" Performance Dashboard")
    st.write("Comparison of model performance with and without GAN augmentation.")

    # --- 1. METRIC CARDS ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Final Accuracy", value="94.2%", delta="11.7%")
    with col2:
        st.metric(label="FID Score", value="15.82", delta="-2.4", delta_color="inverse")
    with col3:
        st.metric(label="Precision", value="0.93", delta="0.08")

    st.write("---")

    # --- 2. ACCURACY BAR CHART ---
    st.subheader("Accuracy Gain")

    # Data for the chart
    df_acc = pd.DataFrame({
        "Model Version": ["Baseline", "GAN Augmented"],
        "Accuracy (%)": [82.5, 94.2]
    })

    fig_acc = px.bar(
        df_acc,
        x="Model Version",
        y="Accuracy (%)",
        color="Model Version",
        text_auto='.1f',
        color_discrete_sequence=["#FF4B4B", "#00CC96"]
    )

    # Make it look like your friend's zoomed chart
    fig_acc.update_layout(yaxis_range=[70, 100], template="plotly_dark")
    st.plotly_chart(fig_acc, use_container_width=True)

    st.write("---")

    # --- 3. CONFUSION MATRIX ---
    st.subheader("Confusion Matrix")

    cm_data = [[48, 2], [3, 47]]
    labels = ['Normal', 'Tumor']

    fig_cm = px.imshow(
        cm_data,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale='Blues'
    )

    fig_cm.update_layout(template="plotly_dark")
    st.plotly_chart(fig_cm, use_container_width=True)