"""
Page Prévisions - Système d'Aide à la Décision Hospitalière

Génération et visualisation des prédictions d'admissions hospitalières futures.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_data, preprocess_data
from src.features import build_features, get_feature_columns
from src.models.train import train_model, train_baseline, evaluate_model, save_model, save_baseline
from src.models.predict import predict, predict_baseline, load_model, load_baseline, forecast
from src.data.preprocess import get_train_test_split

st.set_page_config(
    page_title="Prévisions - Hôpital",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Prévisions des Admissions")
st.markdown("Prédisez les admissions hospitalières futures grâce au machine learning")


# Load and cache data
@st.cache_data
def get_data():
    """Charger et prétraiter les données hospitalières."""
    df = load_data()
    df = preprocess_data(df)
    df = build_features(df)
    return df


@st.cache_resource
def get_trained_model(_df):
    """Entraîner et mettre en cache le modèle."""
    from src.features.build_features import prepare_model_data

    train_df, test_df = get_train_test_split(_df)
    X_train, y_train = prepare_model_data(train_df)
    X_test, y_test = prepare_model_data(test_df)

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, "random_forest", metrics)

    return model, metrics, test_df


@st.cache_data
def get_baseline_params(_df):
    """Obtenir les paramètres du modèle de base."""
    params = train_baseline(_df)
    save_baseline(params, "baseline")
    return params


# Load data
try:
    df = get_data()
except FileNotFoundError as e:
    st.error(f"Fichier de données introuvable : {e}")
    st.stop()

# Sidebar controls
st.sidebar.header("Paramètres de Prévision")

# Model selection
model_choice = st.sidebar.radio(
    "Modèle",
    ["Random Forest (ML)", "Modèle de Base (Saisonnier)"],
    index=0
)

# Forecast horizon
horizon = st.sidebar.slider(
    "Horizon de prévision (jours)",
    min_value=7,
    max_value=90,
    value=30,
    step=7
)

# Start date for forecast
last_date = df["date"].max()
# Convert to Python datetime for reliable date arithmetic
last_date_py = last_date.to_pydatetime() if hasattr(last_date, 'to_pydatetime') else last_date
forecast_start = st.sidebar.date_input(
    "Date de début de prévision",
    value=(last_date_py + timedelta(days=1)).date(),
    min_value=last_date_py.date()
)

st.sidebar.divider()

# Train model button
train_model_btn = st.sidebar.button("🔄 Réentraîner le Modèle", type="secondary")

# Main content
tab1, tab2, tab3 = st.tabs(["📈 Prévisions", "📊 Performance du Modèle", "🔧 Détails du Modèle"])

with tab1:
    st.header("Résultats des Prévisions")

    # Handle retrain button
    if train_model_btn:
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Cache vidé ! Réentraînement du modèle...")
        st.rerun()

    # Get or train model
    with st.spinner("Préparation du modèle..."):
        try:
            if model_choice == "Random Forest (ML)":
                model, metrics, test_df = get_trained_model(df)
                baseline_params = get_baseline_params(df)
            else:
                baseline_params = get_baseline_params(df)
                model, metrics, test_df = None, None, None

        except Exception as e:
            st.error(f"Erreur lors de la préparation du modèle : {e}")
            st.stop()

    # Generate forecast
    with st.spinner("Génération des prévisions..."):
        forecast_start_ts = pd.Timestamp(forecast_start)

        if model_choice == "Random Forest (ML)" and model is not None:
            forecast_df = forecast(model, forecast_start_ts, horizon)
            forecast_col = "predicted_admissions"
        else:
            # Baseline forecast
            from src.models.predict import generate_future_dates
            forecast_df = generate_future_dates(forecast_start_ts, horizon)
            forecast_df["predicted_admissions"] = predict_baseline(baseline_params, forecast_df)
            forecast_col = "predicted_admissions"

    # Display forecast metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_pred = forecast_df[forecast_col].mean()
        st.metric("Prévision Moy./Jour", f"{avg_pred:.0f}")

    with col2:
        max_pred = forecast_df[forecast_col].max()
        st.metric("Jour de Pointe", f"{max_pred:.0f}")

    with col3:
        total_pred = forecast_df[forecast_col].sum()
        st.metric("Total Admissions", f"{total_pred:,.0f}")

    with col4:
        if metrics:
            st.metric("R² du Modèle", f"{metrics['R2']:.2%}")
        else:
            st.metric("Type de Modèle", "Base")

    # Forecast plot
    st.subheader("Visualisation des Prévisions")

    # Historical data for context
    historical_days = min(90, len(df))
    historical_df = df.tail(historical_days).copy()

    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_df["date"],
        y=historical_df["total_admissions"],
        mode="lines",
        name="Historique",
        line=dict(color="#1f77b4", width=1.5)
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df["date"],
        y=forecast_df[forecast_col],
        mode="lines+markers",
        name="Prévision",
        line=dict(color="#ff7f0e", width=2, dash="dash"),
        marker=dict(size=4)
    ))

    # Add confidence band (simple approach: +/- 10% for visualization)
    upper = forecast_df[forecast_col] * 1.15
    lower = forecast_df[forecast_col] * 0.85

    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
        y=pd.concat([upper, lower[::-1]]),
        fill="toself",
        fillcolor="rgba(255,127,14,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Intervalle de confiance (±15%)",
        showlegend=True
    ))

    # Add vertical line at forecast start
    fig.add_shape(
        type="line",
        x0=forecast_start_ts,
        x1=forecast_start_ts,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(dash="dash", color="gray")
    )
    fig.add_annotation(
        x=forecast_start_ts,
        y=1,
        yref="paper",
        text="Début des prévisions",
        showarrow=False,
        yanchor="bottom"
    )

    fig.update_layout(
        title="Données Historiques et Prévisions",
        xaxis_title="Date",
        yaxis_title="Admissions Totales",
        hovermode="x unified",
        height=500
    )

    st.plotly_chart(fig, width="stretch")

    # Forecast table
    with st.expander("📋 Voir les Données de Prévision"):
        display_df = forecast_df[["date", forecast_col]].copy()
        display_df.columns = ["Date", "Admissions Prévues"]
        display_df["Admissions Prévues"] = display_df["Admissions Prévues"].round(0).astype(int)
        st.dataframe(display_df, width="stretch")

    # Download forecast
    csv_data = forecast_df[["date", forecast_col]].to_csv(index=False)
    st.download_button(
        label="📥 Télécharger les Prévisions (CSV)",
        data=csv_data.encode("utf-8"),
        file_name=f"previsions_{horizon}jours.csv",
        mime="text/csv"
    )

with tab2:
    st.header("Performance du Modèle")

    if model is not None and metrics is not None:
        # Performance metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("MAE", f"{metrics['MAE']:.2f}",
                      help="Erreur Absolue Moyenne - erreur de prédiction moyenne")

        with col2:
            st.metric("RMSE", f"{metrics['RMSE']:.2f}",
                      help="Racine de l'Erreur Quadratique Moyenne - pénalise les grandes erreurs")

        with col3:
            st.metric("Score R²", f"{metrics['R2']:.2%}",
                      help="Coefficient de détermination - variance expliquée")

        st.divider()

        # Actual vs Predicted plot
        st.subheader("Réel vs Prédit (Ensemble de Test)")

        from src.features.build_features import prepare_model_data
        X_test, y_test = prepare_model_data(test_df)
        predictions = predict(model, X_test)

        test_results = pd.DataFrame({
            "date": test_df["date"].values,
            "actual": y_test.values,
            "predicted": predictions
        })

        fig_compare = go.Figure()

        fig_compare.add_trace(go.Scatter(
            x=test_results["date"],
            y=test_results["actual"],
            mode="lines",
            name="Réel",
            line=dict(color="#1f77b4")
        ))

        fig_compare.add_trace(go.Scatter(
            x=test_results["date"],
            y=test_results["predicted"],
            mode="lines",
            name="Prédit",
            line=dict(color="#ff7f0e", dash="dash")
        ))

        fig_compare.update_layout(
            title="Prédictions du Modèle vs Valeurs Réelles",
            xaxis_title="Date",
            yaxis_title="Admissions",
            hovermode="x unified",
            height=400
        )

        st.plotly_chart(fig_compare, width="stretch")

        # Residuals plot
        st.subheader("Erreurs de Prédiction (Résidus)")

        test_results["residual"] = test_results["actual"] - test_results["predicted"]

        fig_residuals = px.histogram(
            test_results,
            x="residual",
            nbins=30,
            title="Distribution des Erreurs de Prédiction",
            labels={"residual": "Erreur (Réel - Prédit)"}
        )
        fig_residuals.add_vline(x=0, line_dash="dash", line_color="red")

        st.plotly_chart(fig_residuals, width="stretch")

        # Error statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Erreur Moyenne", f"{test_results['residual'].mean():.2f}")

        with col2:
            st.metric("Écart-type Erreur", f"{test_results['residual'].std():.2f}")

        with col3:
            within_10pct = (abs(test_results['residual']) < test_results['actual'] * 0.1).mean()
            st.metric("Dans ±10%", f"{within_10pct:.1%}")

    else:
        st.info("Sélectionnez 'Random Forest (ML)' pour voir les métriques de performance.")

        # Show baseline info
        st.subheader("Modèle de Base")
        st.markdown("""
        Le modèle de base utilise une prévision **saisonnière naïve** :
        - Moyenne des admissions par jour de la semaine
        - Moyenne des admissions par mois
        - Estimation combinée des deux patterns

        Cela fournit une prévision simple mais interprétable.
        """)

with tab3:
    st.header("Détails du Modèle")

    if model_choice == "Random Forest (ML)":
        st.subheader("Random Forest Regressor")

        st.markdown("""
        **Algorithme** : Random Forest avec 300 arbres

        **Prétraitement** :
        - Variables numériques : Imputation par la médiane
        - Variables catégorielles : Imputation par le mode + Encodage one-hot

        **Validation** : Séparation train/test temporelle (80/20)
        - Pas de mélange pour éviter les fuites de données
        - Ensemble de test = 20% des données les plus récentes
        """)

        st.subheader("Variables Utilisées")

        feature_cols = get_feature_columns(df)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Variables Temporelles**")
            temporal = ["month", "dow", "is_weekend", "season", "day_of_week"]
            for f in temporal:
                if f in feature_cols:
                    st.write(f"- {f}")

        with col2:
            st.markdown("**Variables Opérationnelles**")
            operational = [f for f in feature_cols if f not in temporal]
            for f in operational[:8]:
                st.write(f"- {f}")

        # Feature importance (if available)
        if model is not None:
            st.subheader("Importance des Variables")

            try:
                rf_model = model.named_steps["rf"]
                preprocessor = model.named_steps["prep"]

                # Get feature names after preprocessing
                feature_names = []
                for name, trans, cols in preprocessor.transformers_:
                    if name == "num":
                        feature_names.extend(cols)
                    elif name == "cat":
                        if hasattr(trans.named_steps["onehot"], "get_feature_names_out"):
                            cat_names = trans.named_steps["onehot"].get_feature_names_out(cols)
                            feature_names.extend(cat_names)

                importances = rf_model.feature_importances_

                if len(feature_names) == len(importances):
                    importance_df = pd.DataFrame({
                        "feature": feature_names,
                        "importance": importances
                    }).sort_values("importance", ascending=True).tail(15)

                    fig_imp = px.bar(
                        importance_df,
                        x="importance",
                        y="feature",
                        orientation="h",
                        title="Top 15 des Variables les Plus Importantes"
                    )
                    st.plotly_chart(fig_imp, width="stretch")

            except Exception as e:
                st.warning(f"Impossible d'extraire l'importance des variables : {e}")

    else:
        st.subheader("Modèle de Base (Saisonnier Naïf)")

        st.markdown("""
        **Méthode** : Combine les patterns jour de semaine et mois

        La prédiction pour une date donnée est la moyenne de :
        - La moyenne historique pour ce jour de la semaine
        - La moyenne historique pour ce mois

        Cette approche simple capture les patterns de saisonnalité de base.
        """)

        if baseline_params:
            st.subheader("Moyennes par Jour de la Semaine")

            dow_means = baseline_params.get("dow_means", {})
            dow_df = pd.DataFrame({
                "Jour": ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"],
                "Admissions Moy.": [dow_means.get(str(i), 0) for i in range(7)]
            })

            fig_dow = px.bar(dow_df, x="Jour", y="Admissions Moy.",
                             title="Admissions Moyennes par Jour de la Semaine")
            st.plotly_chart(fig_dow, width="stretch")

            st.subheader("Moyennes Mensuelles")

            month_means = baseline_params.get("month_means", {})
            month_df = pd.DataFrame({
                "Mois": list(range(1, 13)),
                "Admissions Moy.": [month_means.get(str(i), 0) for i in range(1, 13)]
            })

            fig_month = px.bar(month_df, x="Mois", y="Admissions Moy.",
                               title="Admissions Moyennes par Mois")
            st.plotly_chart(fig_month, width="stretch")
