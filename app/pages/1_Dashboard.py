"""
Page Tableau de Bord - Système d'Aide à la Décision Hospitalière

Exploration interactive des données hospitalières historiques avec KPIs et visualisations.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_data, preprocess_data

st.set_page_config(
    page_title="Tableau de Bord - Hôpital",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Tableau de Bord Hospitalier")
st.markdown("Explorez les données historiques et les indicateurs clés de performance")


# Load and cache data
@st.cache_data
def get_data():
    """Charger et prétraiter les données hospitalières."""
    df = load_data()
    df = preprocess_data(df)
    return df


# Load data
try:
    df = get_data()
except FileNotFoundError as e:
    st.error(f"Fichier de données introuvable : {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("Filtres")

# Date range filter
min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Plage de dates",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
    filtered_df = df[mask].copy()
else:
    filtered_df = df.copy()

# Aggregation level
agg_level = st.sidebar.selectbox(
    "Agrégation",
    ["Journalier", "Hebdomadaire", "Mensuel"],
    index=0
)

# Variable selection for detailed charts
var_labels = {
    "total_admissions": "Admissions totales",
    "emergency_admissions": "Admissions urgences",
    "available_beds": "Lits disponibles",
    "available_staff": "Personnel disponible",
    "bed_occupancy_rate": "Taux d'occupation",
    "waiting_time_avg_min": "Temps d'attente moyen"
}
selected_var = st.sidebar.selectbox(
    "Variable à analyser",
    list(var_labels.keys()),
    format_func=lambda x: var_labels[x],
    index=0
)

st.sidebar.divider()
st.sidebar.info(f"Affichage de {len(filtered_df):,} enregistrements")

# KPI Section
st.header("Indicateurs Clés de Performance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    avg_admissions = filtered_df["total_admissions"].mean()
    st.metric(
        "Admissions Moy./Jour",
        f"{avg_admissions:.0f}",
        delta=f"{avg_admissions - df['total_admissions'].mean():.0f} vs global"
    )

with col2:
    avg_emergency = filtered_df["emergency_admissions"].mean()
    pct_emergency = avg_emergency / avg_admissions * 100 if avg_admissions > 0 else 0
    st.metric(
        "Urgences Moy.",
        f"{avg_emergency:.0f}",
        delta=f"{pct_emergency:.0f}% du total"
    )

with col3:
    avg_beds = filtered_df["available_beds"].mean()
    st.metric(
        "Lits Disponibles Moy.",
        f"{avg_beds:.0f}"
    )

with col4:
    avg_occupancy = filtered_df["bed_occupancy_rate"].mean() * 100
    st.metric(
        "Taux d'Occupation Moy.",
        f"{avg_occupancy:.1f}%"
    )

with col5:
    avg_wait = filtered_df["waiting_time_avg_min"].mean()
    st.metric(
        "Temps d'Attente Moy.",
        f"{avg_wait:.0f} min"
    )

st.divider()

# Time Series Section
st.header("Analyse des Séries Temporelles")


def aggregate_data(data, level):
    """Agréger les données par niveau temporel."""
    if level == "Hebdomadaire":
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        return data.set_index("date")[numeric_cols].resample("W").mean().reset_index()
    elif level == "Mensuel":
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        return data.set_index("date")[numeric_cols].resample("MS").mean().reset_index()
    return data


agg_df = aggregate_data(filtered_df, agg_level)

# Main time series chart
fig_ts = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.1,
    subplot_titles=("Admissions Totales", "Taux d'Occupation des Lits")
)

# Admissions with 7-day rolling average
fig_ts.add_trace(
    go.Scatter(
        x=agg_df["date"],
        y=agg_df["total_admissions"],
        mode="lines",
        name="Admissions",
        line=dict(color="#1f77b4", width=1),
        opacity=0.7
    ),
    row=1, col=1
)

if agg_level == "Journalier" and len(agg_df) > 7:
    rolling = agg_df["total_admissions"].rolling(window=7).mean()
    fig_ts.add_trace(
        go.Scatter(
            x=agg_df["date"],
            y=rolling,
            mode="lines",
            name="Moyenne mobile 7j",
            line=dict(color="#ff7f0e", width=2)
        ),
        row=1, col=1
    )

# Occupancy rate
fig_ts.add_trace(
    go.Scatter(
        x=agg_df["date"],
        y=agg_df["bed_occupancy_rate"] * 100,
        mode="lines",
        name="Occupation %",
        line=dict(color="#2ca02c", width=1.5)
    ),
    row=2, col=1
)

# Add threshold lines
fig_ts.add_hline(y=85, line_dash="dash", line_color="red",
                 annotation_text="Critique (85%)", row=2, col=1)
fig_ts.add_hline(y=75, line_dash="dash", line_color="orange",
                 annotation_text="Alerte (75%)", row=2, col=1)

fig_ts.update_layout(
    height=500,
    showlegend=True,
    hovermode="x unified"
)
fig_ts.update_yaxes(title_text="Admissions", row=1, col=1)
fig_ts.update_yaxes(title_text="Occupation %", row=2, col=1)

st.plotly_chart(fig_ts, width="stretch")

st.divider()

# Distribution and Pattern Analysis
st.header("Distribution et Tendances")

col1, col2 = st.columns(2)

with col1:
    # Histogram
    fig_hist = px.histogram(
        filtered_df,
        x=selected_var,
        nbins=30,
        title=f"Distribution de {var_labels[selected_var]}",
        color_discrete_sequence=["#1f77b4"]
    )
    fig_hist.add_vline(
        x=filtered_df[selected_var].mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Moyenne: {filtered_df[selected_var].mean():.1f}"
    )
    st.plotly_chart(fig_hist, width="stretch")

with col2:
    # Box plot by day of week
    fig_box = px.box(
        filtered_df,
        x="dow",
        y="total_admissions",
        title="Admissions par Jour de la Semaine",
        labels={"dow": "Jour (0=Lun, 6=Dim)", "total_admissions": "Admissions"},
        color_discrete_sequence=["#2ca02c"]
    )
    st.plotly_chart(fig_box, width="stretch")

# Seasonal patterns
col1, col2 = st.columns(2)

with col1:
    # Monthly averages
    monthly_avg = filtered_df.groupby("month")["total_admissions"].mean().reset_index()
    fig_monthly = px.bar(
        monthly_avg,
        x="month",
        y="total_admissions",
        title="Admissions Moyennes par Mois",
        labels={"month": "Mois", "total_admissions": "Admissions Moy."},
        color_discrete_sequence=["#ff7f0e"]
    )
    st.plotly_chart(fig_monthly, width="stretch")

with col2:
    # Season averages
    if "season" in filtered_df.columns:
        season_avg = filtered_df.groupby("season")["total_admissions"].mean().reset_index()
        season_order = ["winter", "spring", "summer", "autumn"]
        season_labels = {"winter": "Hiver", "spring": "Printemps", "summer": "Été", "autumn": "Automne"}
        season_avg["season"] = pd.Categorical(season_avg["season"], categories=season_order, ordered=True)
        season_avg = season_avg.sort_values("season")
        season_avg["season_fr"] = season_avg["season"].map(season_labels)

        fig_season = px.bar(
            season_avg,
            x="season_fr",
            y="total_admissions",
            title="Admissions Moyennes par Saison",
            labels={"season_fr": "Saison", "total_admissions": "Admissions Moy."},
            color_discrete_sequence=["#9467bd"]
        )
        st.plotly_chart(fig_season, width="stretch")

st.divider()

# Correlation Analysis
st.header("Analyse des Corrélations")

# Select numeric columns for correlation
numeric_cols = [
    "total_admissions", "emergency_admissions", "pediatric_admissions",
    "icu_admissions", "available_beds", "available_staff",
    "bed_occupancy_rate", "waiting_time_avg_min", "epidemic_level",
    "temperature_c", "staff_absence_rate", "medical_stock_level_pct"
]
numeric_cols = [c for c in numeric_cols if c in filtered_df.columns]

corr_matrix = filtered_df[numeric_cols].corr()

fig_corr = px.imshow(
    corr_matrix,
    labels=dict(color="Corrélation"),
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="Matrice de Corrélation"
)
fig_corr.update_layout(height=600)

st.plotly_chart(fig_corr, width="stretch")

# Top correlations with target
st.subheader("Principales Corrélations avec les Admissions Totales")
target_corr = corr_matrix["total_admissions"].drop("total_admissions").sort_values(key=abs, ascending=False)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Corrélations Positives**")
    positive = target_corr[target_corr > 0].head(5)
    for var, corr in positive.items():
        st.write(f"- {var}: **{corr:.3f}**")

with col2:
    st.markdown("**Corrélations Négatives**")
    negative = target_corr[target_corr < 0].head(5)
    for var, corr in negative.items():
        st.write(f"- {var}: **{corr:.3f}**")

st.divider()

# Scatter plot exploration
st.header("Explorateur de Relations")

col1, col2 = st.columns([1, 3])

with col1:
    x_var = st.selectbox("Variable X", numeric_cols, index=numeric_cols.index("epidemic_level") if "epidemic_level" in numeric_cols else 0)
    y_var = st.selectbox("Variable Y", numeric_cols, index=0)
    color_var = st.selectbox("Colorer par", ["Aucun", "season", "dow", "month"], index=0)

with col2:
    color = None if color_var == "Aucun" else color_var

    fig_scatter = px.scatter(
        filtered_df,
        x=x_var,
        y=y_var,
        color=color,
        opacity=0.6,
        title=f"{y_var.replace('_', ' ').title()} vs {x_var.replace('_', ' ').title()}"
    )
    st.plotly_chart(fig_scatter, width="stretch")

# Data table
with st.expander("📄 Voir les Données Brutes"):
    st.dataframe(
        filtered_df[["date", "total_admissions", "emergency_admissions",
                     "available_beds", "available_staff", "bed_occupancy_rate"]].head(100),
        width="stretch"
    )

# Download option
st.download_button(
    label="📥 Télécharger les Données Filtrées (CSV)",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name="donnees_hopital_filtrees.csv",
    mime="text/csv"
)
