"""
Page Scénarios - Système d'Aide à la Décision Hospitalière
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_data, preprocess_data
from src.features import build_features
from src.models.predict import forecast, load_model
from src.scenarios.simulate import (
    ScenarioParams, apply_scenario,
    create_preset_scenarios, summarize_scenario_impact
)
from src.config import CAPACITY_CONFIG

st.set_page_config(page_title="Scénarios - Hôpital", page_icon="🔮", layout="wide")

st.title("🔮 Simulation de Scénarios")
st.markdown("Comparez les prévisions de référence avec différents scénarios de crise")


@st.cache_data
def get_data():
    df = load_data()
    df = preprocess_data(df)
    df = build_features(df)
    return df


@st.cache_resource
def get_model():
    try:
        return load_model("random_forest")
    except FileNotFoundError:
        return None


try:
    df = get_data()
except FileNotFoundError as e:
    st.error(f"Fichier de données introuvable : {e}")
    st.stop()

model = get_model()

if model is None:
    st.warning("⚠️ Modèle non entraîné. Veuillez d'abord visiter la page Prévisions pour entraîner le modèle.")
    st.info("Utilisation des prévisions de base pour la simulation de scénarios...")

st.sidebar.header("Configuration du Scénario")
st.sidebar.subheader("Paramètres de Prévision")

last_date = df["date"].max()
last_date_py = last_date.to_pydatetime() if hasattr(last_date, 'to_pydatetime') else last_date
forecast_start = st.sidebar.date_input(
    "Date de début",
    value=(last_date_py + timedelta(days=1)).date(),
    format="DD/MM/YYYY"
)

horizon = st.sidebar.select_slider(
    "Horizon (jours)",
    options=[7, 14, 30, 60, 90],
    value=30
)

st.sidebar.divider()
st.sidebar.subheader("Scénarios Prédéfinis")

preset_scenarios = create_preset_scenarios()
preset_options = ["Personnalisé"] + list(preset_scenarios.keys())

selected_preset = st.sidebar.selectbox(
    "Charger un Scénario",
    preset_options,
    index=0
)

st.sidebar.divider()
st.sidebar.subheader("Paramètres du Scénario")

if selected_preset != "Personnalisé":
    preset = preset_scenarios[selected_preset]
    default_epidemic = float(preset.epidemic_intensity)
    default_staffing = float(preset.staffing_reduction)
    default_seasonal = float(preset.seasonal_multiplier)
    default_shock = float(preset.shock_day_spike)
    default_shock_day = int(preset.shock_day_index) if preset.shock_day_index is not None else 0
    default_beds = float(preset.beds_reduction)
    default_stock = float(preset.stock_reduction)
else:
    default_epidemic = 0.0
    default_staffing = 0.0
    default_seasonal = 1.0
    default_shock = 0.0
    default_shock_day = 0
    default_beds = 0.0
    default_stock = 0.0

epidemic_intensity = st.sidebar.slider(
    "Intensité Épidémique (%)",
    min_value=0.0, max_value=100.0, value=default_epidemic, step=5.0,
    help="Pourcentage d'augmentation de la demande de patients"
)

staffing_reduction = st.sidebar.slider(
    "Réduction du Personnel (%)",
    min_value=0.0, max_value=50.0, value=default_staffing, step=5.0,
    help="Pourcentage de réduction du personnel disponible"
)

seasonal_multiplier = st.sidebar.slider(
    "Multiplicateur Saisonnier",
    min_value=0.5, max_value=1.5, value=default_seasonal, step=0.1,
    help="Ajustement pour les effets saisonniers (>1 = demande plus élevée)"
)

beds_reduction = st.sidebar.slider(
    "Réduction des Lits (%)",
    min_value=0.0, max_value=30.0, value=default_beds, step=5.0,
    help="Pourcentage de réduction des lits disponibles"
)

with st.sidebar.expander("Options Avancées", expanded=default_shock > 0):
    shock_spike = st.slider(
        "Pic de Choc (%)",
        min_value=0.0, max_value=200.0, value=default_shock, step=10.0,
        help="Pourcentage de pic sur une journée"
    )

    if shock_spike > 0:
        shock_day_mode = st.radio(
            "Mode de sélection du jour",
            ["Jour relatif (J+X)", "Date exacte"],
            index=0,
            horizontal=True
        )

        if shock_day_mode == "Jour relatif (J+X)":
            shock_day = st.number_input(
                "Jour du Choc (J+)",
                min_value=0, max_value=horizon - 1, value=min(default_shock_day, horizon - 1),
                help="Numéro du jour pour le choc (0 = premier jour)"
            )
        else:
            forecast_start_ts = pd.Timestamp(forecast_start)
            shock_date = st.date_input(
                "Date du Choc",
                value=(forecast_start_ts + timedelta(days=default_shock_day)).date(),
                min_value=forecast_start,
                max_value=(forecast_start_ts + timedelta(days=horizon - 1)).date(),
                format="DD/MM/YYYY"
            )
            shock_day = (pd.Timestamp(shock_date) - forecast_start_ts).days
    else:
        shock_day = None

    stock_reduction = st.slider(
        "Réduction des Stocks (%)",
        min_value=0.0, max_value=50.0, value=default_stock, step=5.0,
        help="Pourcentage de réduction des fournitures médicales"
    )

scenario_params = ScenarioParams(
    epidemic_intensity=epidemic_intensity,
    staffing_reduction=staffing_reduction,
    seasonal_multiplier=seasonal_multiplier,
    shock_day_spike=shock_spike if shock_spike > 0 else 0.0,
    shock_day_index=shock_day if shock_spike > 0 else None,
    beds_reduction=beds_reduction,
    stock_reduction=stock_reduction
)

st.header("Analyse du Scénario")

with st.spinner("Génération des prévisions..."):
    forecast_start_ts = pd.Timestamp(forecast_start)

    if model is not None:
        baseline_df = forecast(model, forecast_start_ts, horizon)
    else:
        from src.models.predict import generate_future_dates, predict_baseline
        from src.models.train import train_baseline

        baseline_params = train_baseline(df)
        baseline_df = generate_future_dates(forecast_start_ts, horizon)
        baseline_df["predicted_admissions"] = predict_baseline(baseline_params, baseline_df)

    scenario_df = apply_scenario(baseline_df, scenario_params)

st.subheader("Résumé de l'Impact")

baseline_summary = {
    "avg_daily": baseline_df["predicted_admissions"].mean(),
    "max_daily": baseline_df["predicted_admissions"].max(),
    "total": baseline_df["predicted_admissions"].sum()
}

scenario_summary = summarize_scenario_impact(scenario_df)

col1, col2, col3, col4 = st.columns(4)

with col1:
    delta = scenario_summary["avg_daily_admissions"] - baseline_summary["avg_daily"]
    delta_pct = delta / baseline_summary["avg_daily"] * 100
    st.metric("Admissions Moy./Jour", f"{scenario_summary['avg_daily_admissions']:.0f}", delta=f"{delta_pct:+.1f}%")

with col2:
    st.metric("Jour de Pointe", f"{scenario_summary['max_daily_admissions']:.0f}",
              delta=f"{scenario_summary['max_daily_admissions'] - baseline_summary['max_daily']:.0f}")

with col3:
    st.metric("Jours en Surcapacité", f"{scenario_summary['days_overcapacity']:.0f}", delta_color="inverse")

with col4:
    st.metric("Occupation Moy.", f"{scenario_summary['avg_occupancy_rate']:.0%}",
              delta=f"{(scenario_summary['avg_occupancy_rate'] - 0.45)*100:+.0f}pp vs réf.")

st.divider()

tab1, tab2, tab3 = st.tabs(["📈 Comparaison des Prévisions", "📊 Analyse de Capacité", "📋 Détails Quotidiens"])

with tab1:
    st.subheader("Référence vs Scénario")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=baseline_df["date"], y=baseline_df["predicted_admissions"],
        mode="lines", name="Prévision de Référence", line=dict(color="#1f77b4", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=scenario_df["date"], y=scenario_df["scenario_admissions"],
        mode="lines", name="Prévision du Scénario", line=dict(color="#ff7f0e", width=2)
    ))

    fig.add_hline(y=CAPACITY_CONFIG.normal_admission_capacity, line_dash="dash", line_color="red",
                  annotation_text=f"Capacité ({CAPACITY_CONFIG.normal_admission_capacity})")

    if scenario_df["effective_capacity"].mean() < CAPACITY_CONFIG.normal_admission_capacity:
        fig.add_trace(go.Scatter(
            x=scenario_df["date"], y=scenario_df["effective_capacity"],
            mode="lines", name="Capacité Effective", line=dict(color="red", width=1, dash="dot")
        ))

    if shock_spike > 0 and shock_day is not None:
        shock_date = scenario_df["date"].iloc[shock_day] if shock_day < len(scenario_df) else None
        if shock_date is not None:
            fig.add_vline(x=shock_date, line_dash="dash", line_color="purple",
                          annotation_text=f"Choc J+{shock_day}")

    fig.update_layout(
        title="Admissions : Référence vs Scénario",
        xaxis_title="Date", yaxis_title="Admissions quotidiennes",
        hovermode="x unified", height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Variation de la Demande par rapport à la Référence")

    fig_delta = go.Figure()
    fig_delta.add_trace(go.Bar(
        x=scenario_df["date"], y=scenario_df["demand_change_pct"],
        marker_color=np.where(scenario_df["demand_change_pct"] > 0, "#ff7f0e", "#1f77b4"),
        name="Variation %"
    ))
    fig_delta.update_layout(title="Variation en Pourcentage des Admissions",
                            xaxis_title="Date", yaxis_title="Variation (%)", height=300)
    st.plotly_chart(fig_delta, width="stretch")

with tab2:
    st.subheader("Analyse de la Capacité")

    col1, col2 = st.columns(2)

    with col1:
        fig_gap = go.Figure()
        colors = np.where(scenario_df["capacity_gap"] > 0, "red", "green")
        fig_gap.add_trace(go.Bar(x=scenario_df["date"], y=scenario_df["capacity_gap"],
                                 marker_color=colors, name="Écart de Capacité"))
        fig_gap.add_hline(y=0, line_color="black", line_width=1)
        fig_gap.update_layout(title="Écart de Capacité Quotidien (Demande - Capacité)",
                              xaxis_title="Date", yaxis_title="Écart (patients)", height=400)
        st.plotly_chart(fig_gap, width="stretch")

    with col2:
        fig_occ = go.Figure()
        fig_occ.add_trace(go.Scatter(
            x=scenario_df["date"], y=scenario_df["occupancy_rate"] * 100,
            mode="lines+markers", name="Taux d'Occupation",
            line=dict(color="#2ca02c", width=2), fill="tozeroy", fillcolor="rgba(44,160,44,0.2)"
        ))
        fig_occ.add_hline(y=CAPACITY_CONFIG.critical_occupancy_threshold * 100,
                          line_dash="dash", line_color="red", annotation_text="Critique (85%)")
        fig_occ.add_hline(y=CAPACITY_CONFIG.warning_occupancy_threshold * 100,
                          line_dash="dash", line_color="orange", annotation_text="Alerte (75%)")
        fig_occ.update_layout(title="Taux d'Occupation des Lits",
                              xaxis_title="Date", yaxis_title="Occupation (%)",
                              yaxis=dict(range=[0, 110]), height=400)
        st.plotly_chart(fig_occ, width="stretch")

    st.subheader("Impact sur les Ressources")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Impact Personnel**")
        base_staff = CAPACITY_CONFIG.total_staff
        eff_staff = scenario_df["effective_staff"].mean()
        st.write(f"- Personnel de base : {base_staff}")
        st.write(f"- Personnel effectif : {eff_staff:.0f}")
        st.write(f"- Réduction : {base_staff - eff_staff:.0f} ({staffing_reduction:.0f}%)")

    with col2:
        st.markdown("**Impact Lits**")
        base_beds = CAPACITY_CONFIG.total_beds
        eff_beds = scenario_df["effective_beds"].mean()
        st.write(f"- Lits de base : {base_beds}")
        st.write(f"- Lits effectifs : {eff_beds:.0f}")
        st.write(f"- Réduction : {base_beds - eff_beds:.0f} ({beds_reduction:.0f}%)")

    with col3:
        st.markdown("**Impact Stocks**")
        eff_stock = scenario_df["effective_stock_pct"].mean()
        st.write(f"- Stock de base : 75%")
        st.write(f"- Stock effectif : {eff_stock:.0f}%")
        st.write(f"- État : {'⚠️ Bas' if eff_stock < 50 else '✅ OK'}")

with tab3:
    st.subheader("Détails des Prévisions Quotidiennes")

    detail_df = scenario_df[[
        "date", "baseline_admissions", "scenario_admissions",
        "effective_capacity", "capacity_gap", "occupancy_rate",
        "is_overcapacity", "is_critical"
    ]].copy()

    detail_df["date"] = detail_df["date"].dt.strftime("%d/%m/%Y")
    detail_df.columns = ["Date", "Référence", "Scénario", "Capacité", "Écart", "Occupation", "Surcapacité", "Critique"]

    detail_df["Référence"] = detail_df["Référence"].round(0).astype(int)
    detail_df["Scénario"] = detail_df["Scénario"].round(0).astype(int)
    detail_df["Capacité"] = detail_df["Capacité"].round(0).astype(int)
    detail_df["Écart"] = detail_df["Écart"].round(0).astype(int)
    detail_df["Occupation"] = (detail_df["Occupation"] * 100).round(1).astype(str) + "%"

    def highlight_overcapacity(row):
        if row["Surcapacité"]:
            return ["background-color: #ffcccc"] * len(row)
        elif row["Critique"]:
            return ["background-color: #fff3cd"] * len(row)
        return [""] * len(row)

    styled_df = detail_df.style.apply(highlight_overcapacity, axis=1)
    st.dataframe(styled_df, width="stretch", height=400)

    csv_data = detail_df.to_csv(index=False, sep=";")
    st.download_button(
        label="📥 Télécharger les Données du Scénario (CSV)",
        data=csv_data.encode("utf-8"),
        file_name=f"scenario_{selected_preset.lower().replace(' ', '_').replace('é', 'e')}.csv",
        mime="text/csv"
    )

st.divider()
st.header("Comparaison Multi-Scénarios")

with st.expander("Comparer Plusieurs Scénarios"):
    st.markdown("Comparez tous les scénarios prédéfinis côte à côte :")

    comparison_data = []

    for name, preset in preset_scenarios.items():
        if name == "Référence":
            continue

        scenario_result = apply_scenario(baseline_df, preset)
        summary = summarize_scenario_impact(scenario_result)

        comparison_data.append({
            "Scénario": name,
            "Admissions Moy.": f"{summary['avg_daily_admissions']:.0f}",
            "Jour de Pointe": f"{summary['max_daily_admissions']:.0f}",
            "Jours Surcapacité": int(summary["days_overcapacity"]),
            "Jours Critiques": int(summary["days_critical"]),
            "Écart Max": f"{summary['max_capacity_gap']:.0f}",
            "Occupation Moy.": f"{summary['avg_occupancy_rate']:.0%}"
        })

    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, width="stretch")

    fig_compare = px.bar(
        comparison_df, x="Scénario", y="Jours Surcapacité", color="Jours Critiques",
        title="Comparaison des Risques par Scénario",
        labels={"Jours Surcapacité": "Jours en Surcapacité", "Jours Critiques": "Jours Critiques"}
    )
    st.plotly_chart(fig_compare, width="stretch")
