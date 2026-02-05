"""
Page Recommandations - Système d'Aide à la Décision Hospitalière
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
from src.scenarios.simulate import ScenarioParams, apply_scenario, create_preset_scenarios
from src.reco.recommend import generate_recommendations, get_priority_actions, summarize_recommendations, Priority
from src.config import CAPACITY_CONFIG

st.set_page_config(page_title="Recommandations - Hôpital", page_icon="📋", layout="wide")

st.title("📋 Recommandations d'Actions")
st.markdown("Obtenez des suggestions actionnables basées sur l'analyse des prévisions et scénarios")


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

st.sidebar.header("Configuration")

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
    value=14
)

st.sidebar.divider()
st.sidebar.subheader("Scénario")

preset_scenarios = create_preset_scenarios()
selected_scenario = st.sidebar.selectbox(
    "Sélectionner un scénario",
    list(preset_scenarios.keys()),
    index=2
)

scenario_params = preset_scenarios[selected_scenario]

with st.sidebar.expander("Détails du scénario"):
    st.write(f"- Épidémie : +{scenario_params.epidemic_intensity}%")
    st.write(f"- Réduction personnel : {scenario_params.staffing_reduction}%")
    st.write(f"- Saisonnier : x{scenario_params.seasonal_multiplier}")
    if scenario_params.shock_day_spike > 0:
        st.write(f"- Pic de choc : +{scenario_params.shock_day_spike}%")
    st.write(f"- Réduction lits : {scenario_params.beds_reduction}%")
    st.write(f"- Réduction stocks : {scenario_params.stock_reduction}%")

st.sidebar.divider()
st.sidebar.subheader("Filtres")

priority_filter = st.sidebar.multiselect(
    "Niveaux de priorité",
    ["CRITIQUE", "HAUTE", "MOYENNE", "BASSE"],
    default=["CRITIQUE", "HAUTE", "MOYENNE"]
)

with st.spinner("Génération des recommandations..."):
    forecast_start_ts = pd.Timestamp(forecast_start)

    if model is not None:
        baseline_df = forecast(model, forecast_start_ts, horizon)
    else:
        from src.models.predict import generate_future_dates, predict_baseline
        from src.models.train import train_baseline

        baseline_params = train_baseline(df)
        baseline_df = generate_future_dates(forecast_start_ts, horizon)
        baseline_df["hospital_id"] = "PITIE"
        baseline_df["predicted_admissions"] = predict_baseline(baseline_params, baseline_df)

    scenario_df = apply_scenario(baseline_df, scenario_params)
    recommendations_df = generate_recommendations(scenario_df)

    if priority_filter:
        recommendations_df = recommendations_df[recommendations_df["priority"].isin(priority_filter)]

st.header("Résumé Exécutif")

summary = summarize_recommendations(recommendations_df)

col1, col2, col3, col4 = st.columns(4)

with col1:
    critical_count = summary.get("critical_count", 0)
    st.metric("Actions Critiques", critical_count, delta_color="inverse")

with col2:
    high_count = summary.get("high_count", 0)
    st.metric("Priorité Haute", high_count)

with col3:
    total_recs = summary.get("total_recommendations", 0)
    st.metric("Total Recommandations", total_recs)

with col4:
    issue_days = summary.get("dates_with_issues", 0)
    st.metric("Jours avec Problèmes", f"{issue_days}/{horizon}")

if critical_count > 0:
    st.error(f"""
    ⚠️ **ALERTE** : {critical_count} actions critiques requises !

    Le scénario sélectionné "{selected_scenario}" génère des situations nécessitant
    une attention immédiate. Examinez les recommandations ci-dessous et préparez des plans de contingence.
    """)
elif high_count > 0:
    st.warning(f"""
    **Attention** : {high_count} actions de haute priorité identifiées.

    Examinez et planifiez les actions recommandées pour maintenir la capacité opérationnelle.
    """)
else:
    st.success("""
    ✅ **Tout va bien** : Aucun problème critique identifié pour ce scénario.

    Continuez la surveillance et les opérations standard.
    """)

st.divider()

st.header("Plan d'Action")

critical_recs = recommendations_df[recommendations_df["priority"] == "CRITIQUE"]
high_recs = recommendations_df[recommendations_df["priority"] == "HAUTE"]
medium_recs = recommendations_df[recommendations_df["priority"] == "MOYENNE"]
low_recs = recommendations_df[recommendations_df["priority"] == "BASSE"]

def format_date(d):
    if hasattr(d, 'strftime'):
        return d.strftime('%d/%m/%Y')
    return str(d)

if len(critical_recs) > 0 and "CRITIQUE" in priority_filter:
    st.subheader("🚨 Actions Critiques")
    for _, rec in critical_recs.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([2, 3, 1])
            with col1:
                st.write(f"**{format_date(rec['date'])}**")
            with col2:
                st.write(f"🔴 {rec['action']}")
                st.caption(rec['description'])
            with col3:
                if rec['quantity'] is not None:
                    st.write(f"{rec['quantity']:.0f} {rec['unit'] or ''}")

if len(high_recs) > 0 and "HAUTE" in priority_filter:
    st.subheader("⚠️ Actions Priorité Haute")
    for _, rec in high_recs.iterrows():
        with st.container():
            col1, col2, col3 = st.columns([2, 3, 1])
            with col1:
                st.write(f"**{format_date(rec['date'])}**")
            with col2:
                st.write(f"🟠 {rec['action']}")
                st.caption(rec['description'])
            with col3:
                if rec['quantity'] is not None:
                    st.write(f"{rec['quantity']:.0f} {rec['unit'] or ''}")

if len(medium_recs) > 0 and "MOYENNE" in priority_filter:
    with st.expander(f"📌 Actions Priorité Moyenne ({len(medium_recs)})"):
        for _, rec in medium_recs.iterrows():
            col1, col2, col3 = st.columns([2, 3, 1])
            with col1:
                st.write(f"{format_date(rec['date'])}")
            with col2:
                st.write(f"🟡 {rec['action']}")
                st.caption(rec['description'])
            with col3:
                if rec['quantity'] is not None:
                    st.write(f"{rec['quantity']:.0f} {rec['unit'] or ''}")

if len(low_recs) > 0 and "BASSE" in priority_filter:
    with st.expander(f"ℹ️ Éléments à Surveiller ({len(low_recs)})"):
        st.write("Opérations normales ces jours-là - continuez la surveillance standard.")

st.divider()

st.header("Tableau Complet des Recommandations")

display_df = recommendations_df.copy()
if "date" in display_df.columns:
    display_df["date"] = display_df["date"].apply(format_date)

def color_priority(val):
    colors = {
        "CRITIQUE": "background-color: #f8d7da",
        "HAUTE": "background-color: #fff3cd",
        "MOYENNE": "background-color: #d1ecf1",
        "BASSE": "background-color: #d4edda"
    }
    return colors.get(val, "")

styled_df = display_df.style.map(color_priority, subset=["priority"])
st.dataframe(styled_df, width="stretch", height=400)

col1, col2 = st.columns(2)

with col1:
    csv_data = recommendations_df.to_csv(index=False, sep=";")
    st.download_button(
        label="📥 Télécharger les Recommandations (CSV)",
        data=csv_data.encode("utf-8"),
        file_name=f"recommandations_{selected_scenario.lower().replace(' ', '_').replace('é', 'e')}.csv",
        mime="text/csv"
    )

with col2:
    report = f"""
SYSTÈME D'AIDE À LA DÉCISION HOSPITALIÈRE - RAPPORT D'ACTIONS
=============================================================

Scénario : {selected_scenario}
Période : {forecast_start.strftime('%d/%m/%Y')} au {(forecast_start_ts + timedelta(days=horizon-1)).strftime('%d/%m/%Y')}
Généré le : {datetime.now().strftime('%d/%m/%Y %H:%M')}

RÉSUMÉ
------
- Actions Critiques : {critical_count}
- Priorité Haute : {high_count}
- Total Recommandations : {total_recs}
- Jours avec Problèmes : {issue_days}/{horizon}

PARAMÈTRES DU SCÉNARIO
----------------------
- Intensité Épidémique : +{scenario_params.epidemic_intensity}%
- Réduction Personnel : {scenario_params.staffing_reduction}%
- Multiplicateur Saisonnier : x{scenario_params.seasonal_multiplier}
- Réduction Lits : {scenario_params.beds_reduction}%
- Réduction Stocks : {scenario_params.stock_reduction}%

ACTIONS CRITIQUES
-----------------
"""
    for _, rec in critical_recs.iterrows():
        date_str = format_date(rec['date'])
        report += f"\n[{date_str}] {rec['action']}\n  → {rec['description']}\n"

    report += """
ACTIONS PRIORITÉ HAUTE
----------------------
"""
    for _, rec in high_recs.iterrows():
        date_str = format_date(rec['date'])
        report += f"\n[{date_str}] {rec['action']}\n  → {rec['description']}\n"

    st.download_button(
        label="📄 Télécharger le Rapport (TXT)",
        data=report.encode("utf-8"),
        file_name=f"rapport_actions_{selected_scenario.lower().replace(' ', '_').replace('é', 'e')}.txt",
        mime="text/plain"
    )

st.divider()

st.header("Visualisations d'Analyse")

tab1, tab2 = st.tabs(["📊 Actions par Type", "📈 Chronologie"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        action_counts = recommendations_df["action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Nombre"]

        fig_actions = px.bar(
            action_counts.head(10), x="Nombre", y="Action", orientation="h",
            title="Actions les Plus Recommandées", color="Nombre", color_continuous_scale="Oranges"
        )
        fig_actions.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_actions, width="stretch")

    with col2:
        priority_counts = recommendations_df["priority"].value_counts().reset_index()
        priority_counts.columns = ["Priorité", "Nombre"]

        colors = {"CRITIQUE": "#dc3545", "HAUTE": "#ffc107", "MOYENNE": "#17a2b8", "BASSE": "#28a745"}

        fig_priority = px.pie(
            priority_counts, values="Nombre", names="Priorité",
            title="Recommandations par Priorité", color="Priorité", color_discrete_map=colors
        )
        fig_priority.update_layout(height=400)
        st.plotly_chart(fig_priority, width="stretch")

with tab2:
    timeline_df = recommendations_df.copy()
    priority_values = {"CRITIQUE": 4, "HAUTE": 3, "MOYENNE": 2, "BASSE": 1}
    timeline_df["priority_value"] = timeline_df["priority"].map(priority_values)

    fig_timeline = px.scatter(
        timeline_df, x="date", y="action", color="priority", size="priority_value",
        title="Chronologie des Recommandations",
        color_discrete_map={"CRITIQUE": "#dc3545", "HAUTE": "#ffc107", "MOYENNE": "#17a2b8", "BASSE": "#28a745"},
        hover_data=["description"],
        labels={"date": "Date", "action": "Action", "priority": "Priorité"}
    )
    fig_timeline.update_layout(height=500)
    st.plotly_chart(fig_timeline, width="stretch")

st.header("Résumé des Besoins en Ressources")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🛏️ Lits")
    bed_actions = recommendations_df[recommendations_df["action"].str.contains("lits", case=False, na=False)]
    if len(bed_actions) > 0:
        total_beds = bed_actions["quantity"].sum()
        st.metric("Lits Supplémentaires Nécessaires", f"{total_beds:.0f}" if pd.notna(total_beds) else "N/A")
        st.write(f"Sur {len(bed_actions)} recommandations")
    else:
        st.write("Aucun ajustement de lits nécessaire")

with col2:
    st.subheader("👥 Personnel")
    staff_actions = recommendations_df[
        recommendations_df["action"].str.contains("personnel|heures", case=False, na=False)
    ]
    if len(staff_actions) > 0:
        overtime_hours = staff_actions[staff_actions["unit"] == "heures"]["quantity"].sum()
        st.metric("Heures Supplémentaires", f"{overtime_hours:.0f}" if pd.notna(overtime_hours) else "N/A")
        st.write(f"Sur {len(staff_actions)} recommandations")
    else:
        st.write("Aucun ajustement de personnel nécessaire")

with col3:
    st.subheader("📦 Fournitures")
    stock_actions = recommendations_df[
        recommendations_df["action"].str.contains("fournitures|stock", case=False, na=False)
    ]
    if len(stock_actions) > 0:
        st.metric("Alertes de Stock", len(stock_actions))
        st.write("Réapprovisionnement recommandé")
    else:
        st.write("Niveaux de stock adéquats")

st.divider()

with st.expander("📖 Règles de Décision"):
    st.markdown("""
    Les recommandations sont générées selon les règles suivantes :

    | Condition | Seuil | Action | Priorité |
    |-----------|-------|--------|----------|
    | Surcapacité | Demande > Capacité + 50 | Protocole de surcharge + Renfort externe | CRITIQUE |
    | Surcapacité | Demande > Capacité | Ajouter lits temporaires | HAUTE |
    | Pénurie personnel | Personnel < 70% | Demander renfort | CRITIQUE |
    | Pénurie personnel | Personnel < 85% | Heures supplémentaires | HAUTE |
    | Occupation élevée | > 80% | Reporter interventions programmées | MOYENNE |
    | Occupation élevée | > 75% | Alerter administration | MOYENNE |
    | Stocks bas | < 40% | Réapprovisionnement urgent | HAUTE |
    | Stocks bas | < 50% | Réapprovisionnement | MOYENNE |
    | Situation critique | Occupation > 85% | Redistribuer patients | HAUTE |
    """)
