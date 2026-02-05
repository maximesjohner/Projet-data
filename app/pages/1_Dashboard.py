"""
Page Tableau de Bord - Système d'Aide à la Décision Hospitalière
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_data, preprocess_data

st.set_page_config(page_title="Tableau de Bord - Hôpital", page_icon="📊", layout="wide")

st.title("📊 Tableau de Bord Hospitalier")
st.markdown("Explorez les données historiques et les indicateurs clés de performance")


@st.cache_data
def get_data():
    df = load_data()
    df = preprocess_data(df)
    return df


try:
    df = get_data()
except FileNotFoundError as e:
    st.error(f"Fichier de données introuvable : {e}")
    st.stop()

st.sidebar.header("Filtres")

min_date = df["date"].min().date()
max_date = df["date"].max().date()

date_range = st.sidebar.date_input(
    "Plage de dates",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    format="DD/MM/YYYY"
)

if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
    filtered_df = df[mask].copy()
else:
    filtered_df = df.copy()

agg_level = st.sidebar.selectbox("Agrégation", ["Journalier", "Hebdomadaire", "Mensuel"], index=0)

var_labels = {
    "total_admissions": "Admissions totales",
    "emergency_admissions": "Admissions urgences",
    "pediatric_admissions": "Admissions pédiatrie",
    "icu_admissions": "Admissions réanimation",
    "available_beds": "Lits disponibles",
    "available_staff": "Personnel disponible",
    "bed_occupancy_rate": "Taux d'occupation",
    "waiting_time_avg_min": "Temps d'attente (min)",
    "medical_stock_level_pct": "Stock médical (%)",
    "estimated_cost_per_day": "Coût estimé (€/jour)",
    "patient_satisfaction_score": "Satisfaction patient"
}
selected_var = st.sidebar.selectbox(
    "Variable à analyser",
    list(var_labels.keys()),
    format_func=lambda x: var_labels[x],
    index=0
)

st.sidebar.divider()

st.sidebar.subheader("Filtres d'Événements")
event_filters = {}
if "epidemic_level" in filtered_df.columns:
    event_filters["epidemic"] = st.sidebar.checkbox("Épidémie (niveau > 0)", value=False)
if "heatwave_event" in filtered_df.columns:
    event_filters["heatwave"] = st.sidebar.checkbox("Canicule", value=False)
if "strike_level" in filtered_df.columns:
    event_filters["strike"] = st.sidebar.checkbox("Grève", value=False)
if "accident_event" in filtered_df.columns:
    event_filters["accident"] = st.sidebar.checkbox("Accident majeur", value=False)

if any(event_filters.values()):
    # Use OR logic: show days with ANY of the selected events
    event_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
    if event_filters.get("epidemic"):
        event_mask |= filtered_df["epidemic_level"] > 0
    if event_filters.get("heatwave"):
        event_mask |= filtered_df["heatwave_event"] == 1
    if event_filters.get("strike"):
        event_mask |= filtered_df["strike_level"] > 0
    if event_filters.get("accident"):
        event_mask |= filtered_df["accident_event"] == 1
    filtered_df = filtered_df[event_mask]

# Handle empty filtered data
if len(filtered_df) == 0:
    st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
    st.stop()

st.sidebar.divider()
st.sidebar.info(f"Affichage de {len(filtered_df):,} enregistrements")

st.header("Indicateurs Clés de Performance")

row1 = st.columns(6)
row2 = st.columns(5)

with row1[0]:
    avg_admissions = filtered_df["total_admissions"].mean()
    st.metric("Admissions Moy./Jour", f"{avg_admissions:.0f}",
              delta=f"{avg_admissions - df['total_admissions'].mean():.0f} vs global")

with row1[1]:
    avg_emergency = filtered_df["emergency_admissions"].mean()
    st.metric("Urgences Moy.", f"{avg_emergency:.0f}")

with row1[2]:
    if "pediatric_admissions" in filtered_df.columns:
        avg_ped = filtered_df["pediatric_admissions"].mean()
        st.metric("Pédiatrie Moy.", f"{avg_ped:.0f}")

with row1[3]:
    if "icu_admissions" in filtered_df.columns:
        avg_icu = filtered_df["icu_admissions"].mean()
        st.metric("Réanimation Moy.", f"{avg_icu:.0f}")

with row1[4]:
    avg_beds = filtered_df["available_beds"].mean()
    st.metric("Lits Disponibles Moy.", f"{avg_beds:.0f}")

with row1[5]:
    if "available_staff" in filtered_df.columns:
        avg_staff = filtered_df["available_staff"].mean()
        st.metric("Personnel Moy.", f"{avg_staff:.0f}")

with row2[0]:
    avg_occupancy = filtered_df["bed_occupancy_rate"].mean() * 100
    st.metric("Taux d'Occupation Moy.", f"{avg_occupancy:.1f}%")

with row2[1]:
    avg_wait = filtered_df["waiting_time_avg_min"].mean()
    st.metric("Temps d'Attente Moy.", f"{avg_wait:.0f} min")

with row2[2]:
    if "medical_stock_level_pct" in filtered_df.columns:
        avg_stock = filtered_df["medical_stock_level_pct"].mean()
        st.metric("Stock Médical Moy.", f"{avg_stock:.0f}%")

with row2[3]:
    if "estimated_cost_per_day" in filtered_df.columns:
        avg_cost = filtered_df["estimated_cost_per_day"].mean()
        st.metric("Coût Moy./Jour", f"{avg_cost/1000:.0f}k €")

with row2[4]:
    if "patient_satisfaction_score" in filtered_df.columns:
        avg_sat = filtered_df["patient_satisfaction_score"].mean()
        st.metric("Satisfaction Patient", f"{avg_sat:.1f}/10")

st.divider()

st.header("Analyse des Séries Temporelles")


def aggregate_data(data, level):
    if level == "Hebdomadaire":
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        return data.set_index("date")[numeric_cols].resample("W").mean().reset_index()
    elif level == "Mensuel":
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        return data.set_index("date")[numeric_cols].resample("MS").mean().reset_index()
    return data


agg_df = aggregate_data(filtered_df, agg_level)

fig_ts = make_subplots(
    rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.1,
    subplot_titles=("Admissions Totales", "Taux d'Occupation (%)",
                    "Temps d'Attente (min)", "Lits et Personnel")
)

fig_ts.add_trace(go.Scatter(
    x=agg_df["date"], y=agg_df["total_admissions"],
    mode="lines", name="Admissions", line=dict(color="#1f77b4", width=1), opacity=0.7
), row=1, col=1)

if agg_level == "Journalier" and len(agg_df) > 7:
    rolling = agg_df["total_admissions"].rolling(window=7).mean()
    fig_ts.add_trace(go.Scatter(
        x=agg_df["date"], y=rolling,
        mode="lines", name="Moyenne mobile 7j", line=dict(color="#ff7f0e", width=2)
    ), row=1, col=1)

fig_ts.add_trace(go.Scatter(
    x=agg_df["date"], y=agg_df["bed_occupancy_rate"] * 100,
    mode="lines", name="Occupation %", line=dict(color="#2ca02c", width=1.5)
), row=1, col=2)

fig_ts.add_hline(y=85, line_dash="dash", line_color="red", row=1, col=2)
fig_ts.add_hline(y=75, line_dash="dash", line_color="orange", row=1, col=2)

fig_ts.add_trace(go.Scatter(
    x=agg_df["date"], y=agg_df["waiting_time_avg_min"],
    mode="lines", name="Temps d'attente", line=dict(color="#9467bd", width=1.5)
), row=2, col=1)

fig_ts.add_trace(go.Scatter(
    x=agg_df["date"], y=agg_df["available_beds"],
    mode="lines", name="Lits", line=dict(color="#17becf", width=1.5)
), row=2, col=2)

if "available_staff" in agg_df.columns:
    fig_ts.add_trace(go.Scatter(
        x=agg_df["date"], y=agg_df["available_staff"],
        mode="lines", name="Personnel", line=dict(color="#bcbd22", width=1.5)
    ), row=2, col=2)

fig_ts.update_layout(height=600, showlegend=True, hovermode="x unified")
fig_ts.update_yaxes(title_text="Admissions", row=1, col=1)
fig_ts.update_yaxes(title_text="Occupation %", row=1, col=2)
fig_ts.update_yaxes(title_text="Minutes", row=2, col=1)
fig_ts.update_yaxes(title_text="Effectif", row=2, col=2)

st.plotly_chart(fig_ts, width="stretch")

if "medical_stock_level_pct" in agg_df.columns:
    st.subheader("Niveau de Stock Médical")
    fig_stock = go.Figure()
    fig_stock.add_trace(go.Scatter(
        x=agg_df["date"], y=agg_df["medical_stock_level_pct"],
        mode="lines", name="Stock %", fill="tozeroy",
        line=dict(color="#e377c2", width=1.5), fillcolor="rgba(227,119,194,0.2)"
    ))
    fig_stock.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Seuil d'alerte (50%)")
    fig_stock.update_layout(title="Évolution du Stock Médical", xaxis_title="Date",
                            yaxis_title="Stock (%)", height=300)
    st.plotly_chart(fig_stock, width="stretch")

st.divider()

st.header("Distribution et Tendances")

col1, col2 = st.columns(2)

with col1:
    fig_hist = px.histogram(
        filtered_df, x=selected_var, nbins=30,
        title=f"Distribution de {var_labels[selected_var]}",
        color_discrete_sequence=["#1f77b4"],
        labels={selected_var: var_labels[selected_var]}
    )
    fig_hist.add_vline(
        x=filtered_df[selected_var].mean(),
        line_dash="dash", line_color="red",
        annotation_text=f"Moyenne: {filtered_df[selected_var].mean():.1f}"
    )
    st.plotly_chart(fig_hist, width="stretch")

with col2:
    dow_labels = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
    box_df = filtered_df.copy()
    box_df["jour"] = box_df["dow"].map(lambda x: dow_labels[x])

    fig_box = px.box(
        box_df, x="jour", y="total_admissions",
        title="Admissions par Jour de la Semaine",
        labels={"jour": "Jour", "total_admissions": "Admissions"},
        color_discrete_sequence=["#2ca02c"],
        category_orders={"jour": dow_labels}
    )
    st.plotly_chart(fig_box, width="stretch")

col1, col2 = st.columns(2)

with col1:
    month_names = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    monthly_avg = filtered_df.groupby("month")["total_admissions"].mean().reset_index()
    monthly_avg["mois"] = monthly_avg["month"].map(lambda x: month_names[x-1])

    fig_monthly = px.bar(
        monthly_avg, x="mois", y="total_admissions",
        title="Admissions Moyennes par Mois",
        labels={"mois": "Mois", "total_admissions": "Admissions Moy."},
        color_discrete_sequence=["#ff7f0e"],
        category_orders={"mois": month_names}
    )
    st.plotly_chart(fig_monthly, width="stretch")

with col2:
    if "season" in filtered_df.columns:
        season_labels = {"winter": "Hiver", "spring": "Printemps", "summer": "Été", "autumn": "Automne"}
        season_order = ["Hiver", "Printemps", "Été", "Automne"]

        season_avg = filtered_df.groupby("season", observed=True)["total_admissions"].mean().reset_index()
        season_avg["saison"] = season_avg["season"].map(season_labels)

        fig_season = px.bar(
            season_avg, x="saison", y="total_admissions",
            title="Admissions Moyennes par Saison",
            labels={"saison": "Saison", "total_admissions": "Admissions Moy."},
            color_discrete_sequence=["#9467bd"],
            category_orders={"saison": season_order}
        )
        st.plotly_chart(fig_season, width="stretch")

st.divider()

st.header("Analyse des Corrélations")

numeric_cols = [
    "total_admissions", "emergency_admissions", "pediatric_admissions",
    "icu_admissions", "available_beds", "available_staff",
    "bed_occupancy_rate", "waiting_time_avg_min", "epidemic_level",
    "temperature_c", "staff_absence_rate", "medical_stock_level_pct"
]
numeric_cols = [c for c in numeric_cols if c in filtered_df.columns]

corr_matrix = filtered_df[numeric_cols].corr()

var_labels_corr = {
    "total_admissions": "Admissions", "emergency_admissions": "Urgences",
    "pediatric_admissions": "Pédiatrie", "icu_admissions": "Réanimation",
    "available_beds": "Lits", "available_staff": "Personnel",
    "bed_occupancy_rate": "Occupation", "waiting_time_avg_min": "Attente",
    "epidemic_level": "Épidémie", "temperature_c": "Température",
    "staff_absence_rate": "Absence", "medical_stock_level_pct": "Stock"
}

corr_labels = [var_labels_corr.get(c, c) for c in corr_matrix.columns]

fig_corr = px.imshow(
    corr_matrix,
    labels=dict(color="Corrélation"),
    x=corr_labels, y=corr_labels,
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="Matrice de Corrélation"
)
fig_corr.update_layout(height=600)

st.plotly_chart(fig_corr, width="stretch")

st.subheader("Principales Corrélations avec les Admissions Totales")
target_corr = corr_matrix["total_admissions"].drop("total_admissions").sort_values(key=abs, ascending=False)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Corrélations Positives**")
    positive = target_corr[target_corr > 0].head(5)
    for var, corr in positive.items():
        st.write(f"- {var_labels_corr.get(var, var)}: **{corr:.3f}**")

with col2:
    st.markdown("**Corrélations Négatives**")
    negative = target_corr[target_corr < 0].head(5)
    for var, corr in negative.items():
        st.write(f"- {var_labels_corr.get(var, var)}: **{corr:.3f}**")

st.divider()

st.header("Explorateur de Relations")

col1, col2 = st.columns([1, 3])

with col1:
    x_var = st.selectbox("Variable X", numeric_cols,
                         index=numeric_cols.index("epidemic_level") if "epidemic_level" in numeric_cols else 0,
                         format_func=lambda x: var_labels_corr.get(x, x))
    y_var = st.selectbox("Variable Y", numeric_cols, index=0,
                         format_func=lambda x: var_labels_corr.get(x, x))
    color_options = ["Aucun", "season", "dow", "month"]
    color_labels = {"Aucun": "Aucun", "season": "Saison", "dow": "Jour", "month": "Mois"}
    color_var = st.selectbox("Colorer par", color_options, index=0,
                             format_func=lambda x: color_labels.get(x, x))

with col2:
    color = None if color_var == "Aucun" else color_var

    fig_scatter = px.scatter(
        filtered_df, x=x_var, y=y_var, color=color, opacity=0.6,
        title=f"{var_labels_corr.get(y_var, y_var)} vs {var_labels_corr.get(x_var, x_var)}",
        labels={x_var: var_labels_corr.get(x_var, x_var), y_var: var_labels_corr.get(y_var, y_var)}
    )
    st.plotly_chart(fig_scatter, width="stretch")

with st.expander("📄 Voir les Données Brutes"):
    display_cols = ["date", "total_admissions", "emergency_admissions",
                    "available_beds", "available_staff", "bed_occupancy_rate"]
    display_cols = [c for c in display_cols if c in filtered_df.columns]
    display_df = filtered_df[display_cols].head(100).copy()
    display_df["date"] = display_df["date"].dt.strftime("%d/%m/%Y")
    st.dataframe(display_df, width="stretch")

csv_export = filtered_df.copy()
csv_export["date"] = csv_export["date"].dt.strftime("%d/%m/%Y")
csv_data = csv_export.to_csv(index=False, sep=";")
st.download_button(
    label="📥 Télécharger les Données Filtrées (CSV)",
    data=csv_data.encode('utf-8'),
    file_name="donnees_hopital_filtrees.csv",
    mime="text/csv"
)
