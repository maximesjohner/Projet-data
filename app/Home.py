import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration
st.set_page_config(
    page_title="Aide à la Décision Hospitalière",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🏥 Système d\'Aide à la Décision Hospitalière</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Tableau de bord interactif pour la planification des capacités, les prévisions et l\'analyse de scénarios</p>', unsafe_allow_html=True)

st.divider()

# Introduction
st.markdown("""
## Bienvenue

Cette application aide les administrateurs et planificateurs hospitaliers à prendre des décisions
basées sur les données concernant l'allocation des ressources, la planification des capacités et la réponse aux crises.

**Fonctionnalités principales :**
- 📊 **Tableau de bord** : Explorez les données historiques avec des visualisations interactives
- 📈 **Prévisions** : Prédisez les admissions futures grâce au machine learning
- 🔮 **Simulation de scénarios** : Testez des scénarios hypothétiques (épidémies, grèves, pics saisonniers)
- 📋 **Recommandations** : Obtenez des suggestions concrètes pour la gestion des capacités
""")

# System overview in columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Aperçu des Données")
    st.markdown("""
    Le système analyse les données hospitalières quotidiennes incluant :
    - **Admissions** : Totales, urgences, pédiatrie, réanimation
    - **Capacité** : Lits disponibles, effectifs du personnel
    - **Facteurs externes** : Météo, épidémies, événements
    - **Opérations** : Temps d'attente, chirurgies, niveaux de stock
    """)

with col2:
    st.markdown("### 🎯 Cible de Prédiction")
    st.markdown("""
    La cible principale de prédiction est le **nombre total d'admissions quotidiennes**.

    Cette métrique est cruciale pour :
    - La planification de l'allocation des lits
    - La programmation du personnel
    - L'approvisionnement en ressources
    - La préparation aux urgences
    """)

st.divider()

# Quick start guide
st.markdown("### 🚀 Guide de Démarrage Rapide")

tabs = st.tabs(["1. Tableau de bord", "2. Prévisions", "3. Scénarios", "4. Recommandations"])

with tabs[0]:
    st.markdown("""
    **Page Tableau de Bord**

    Explorez les données historiques avec :
    - Indicateurs clés de performance (KPI)
    - Graphiques temporels des admissions
    - Analyse des distributions
    - Cartes de corrélation

    Utilisez les filtres de la barre latérale pour sélectionner les plages de dates et les variables d'intérêt.
    """)

with tabs[1]:
    st.markdown("""
    **Page Prévisions**

    Générez des prédictions pour les admissions futures :
    - Sélectionnez un horizon de prévision (7-90 jours)
    - Choisissez entre le modèle de base et le modèle ML
    - Visualisez les prédictions avec les bandes de confiance
    - Comparez les métriques de performance des modèles
    """)

with tabs[2]:
    st.markdown("""
    **Page Scénarios**

    Simulez différentes situations :
    - **Épidémie** : Augmentation de la demande de 10-50%
    - **Pénurie de personnel** : Réduction de la capacité de 10-40%
    - **Effets saisonniers** : Ajustement pour les pics hivernaux
    - **Événements chocs** : Modélisation de pics soudains

    Comparez les scénarios côte à côte avec les prévisions de référence.
    """)

with tabs[3]:
    st.markdown("""
    **Page Recommandations**

    Obtenez des informations exploitables :
    - Recommandations classées par priorité
    - Actions spécifiques (ajout de lits, personnel, etc.)
    - Besoins en ressources quantifiés
    - Rapports exportables
    """)

st.divider()

# Assumptions and methodology
with st.expander("📖 Hypothèses et Méthodologie"):
    st.markdown("""
    ### Données
    - Jeu de données : Opérations hospitalières quotidiennes (2023-2028)
    - 2 192 enregistrements avec 29 variables
    - Aucune valeur manquante dans les données sources

    ### Modèle
    - **Algorithme** : Random Forest Regressor
    - **Variables** : Temporelles (jour, mois, saison) + opérationnelles (personnel, lits, événements)
    - **Validation** : Séparation train/test temporelle (80/20) pour éviter les fuites de données
    - **Performance** : R² = 0.82 sur l'ensemble de test (ensemble de variables réalistes)

    ### Hypothèses de Capacité
    - Lits totaux : ~1 500
    - Personnel total : ~430
    - Capacité d'admission normale : ~450/jour
    - Seuil d'occupation critique : 85%
    - Seuil d'alerte : 75%

    ### Logique des Scénarios
    Les scénarios appliquent des multiplicateurs déterministes aux prévisions de référence :
    - Épidémie : Augmente la demande prévue proportionnellement
    - Réduction du personnel : Diminue la capacité effective
    - Multiplicateur saisonnier : Ajuste selon les périodes de l'année
    - Pic choc : Modélise des événements de pointe ponctuels
    """)

# Footer
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Navigation**")
    st.markdown("Utilisez la barre latérale pour naviguer entre les pages →")

with col2:
    st.markdown("**État des Données**")
    try:
        from src.data import load_data
        df = load_data()
        st.success(f"✅ Données chargées : {len(df):,} enregistrements")
    except Exception as e:
        st.error(f"❌ Données non disponibles : {e}")

with col3:
    st.markdown("**État du Modèle**")
    try:
        from src.models import load_model
        model = load_model()
        st.success("✅ Modèle prêt")
    except Exception:
        st.warning("⚠️ Modèle non entraîné - visitez la page Prévisions pour l'entraîner")

# Sidebar info
with st.sidebar:
    st.markdown("### À propos")
    st.info("""
    **Système d'Aide à la Décision Hospitalière**

    Un projet académique pour la
    planification des capacités hospitalières
    et l'analyse de scénarios.

    Version 1.0.0
    """)

    st.markdown("### Navigation")
    st.markdown("""
    1. 📊 Tableau de bord
    2. 📈 Prévisions
    3. 🔮 Scénarios
    4. 📋 Recommandations
    """)
