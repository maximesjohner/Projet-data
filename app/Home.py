"""
Page Accueil - Système d'Aide à la Décision Hospitalière
"""
import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import check_data_exists

st.set_page_config(
    page_title="Aide à la Décision Hospitalière",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #666; margin-bottom: 2rem; }
    .workflow-step { padding: 1rem; border-radius: 8px; text-align: center; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🏥 Système d\'Aide à la Décision Hospitalière</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Tableau de bord interactif pour la planification des capacités, les prévisions et l\'analyse de scénarios</p>', unsafe_allow_html=True)

if not check_data_exists():
    st.error("⚠️ **Aucune donnée disponible**")
    st.warning("""
    Les données n'ont pas encore été générées. Pour utiliser l'application, exécutez d'abord :

    ```bash
    python run.py generate
    ```
    """)
    st.stop()

st.divider()

st.markdown("## Bienvenue")

st.markdown("""
Cette application aide les administrateurs et planificateurs hospitaliers à prendre des décisions
basées sur les données concernant l'allocation des ressources, la planification des capacités et la réponse aux crises.
""")

st.info("⚠️ **Données synthétiques** - Cette application utilise des données générées à des fins de démonstration.")

st.markdown("### 🔄 Workflow Décisionnel")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="workflow-step" style="background-color: #e3f2fd;">
        <h3>👁️ Observer</h3>
        <p>Explorez les données historiques</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="workflow-step" style="background-color: #fff3e0;">
        <h3>📈 Prévoir</h3>
        <p>Prédisez les admissions futures</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="workflow-step" style="background-color: #f3e5f5;">
        <h3>🔮 Simuler</h3>
        <p>Testez différents scénarios</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="workflow-step" style="background-color: #e8f5e9;">
        <h3>✅ Décider</h3>
        <p>Obtenez des recommandations</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.markdown("### 📚 Fonctionnalités")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **📊 Tableau de bord**
    - Indicateurs clés de performance (KPIs)
    - Visualisations interactives
    - Analyse des corrélations
    - Filtrage par événements (épidémie, grève, canicule)

    **📈 Prévisions**
    - Modèle ML (Random Forest)
    - Horizon de prévision : 7 à 90 jours
    - Métriques de performance (R², MAE, RMSE, MAPE)
    - Comparaison avec modèle de référence
    """)

with col2:
    st.markdown("""
    **🔮 Simulation de Scénarios**
    - Scénarios prédéfinis (épidémie, grève, pic hivernal)
    - Personnalisation des paramètres
    - Analyse de l'impact sur la capacité
    - Comparaison multi-scénarios

    **📋 Recommandations**
    - Actions prioritaires (critique, haute, moyenne, basse)
    - Plan d'action détaillé
    - Besoins en ressources (lits, personnel, fournitures)
    - Règles de décision transparentes
    """)

st.divider()

st.markdown("### 📊 Aperçu des Données")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    Le système analyse les données hospitalières quotidiennes incluant :
    - **Admissions** : Totales, urgences, pédiatrie, réanimation
    - **Capacité** : Lits disponibles, effectifs du personnel
    - **Facteurs externes** : Météo, épidémies, événements
    - **Opérations** : Temps d'attente, chirurgies, niveaux de stock
    """)

with col2:
    st.markdown("""
    **🎯 Cible de Prédiction**

    La cible principale de prédiction est le **nombre total d'admissions quotidiennes**.

    Cette métrique est cruciale pour :
    - La planification de l'allocation des lits
    - La programmation du personnel
    - L'approvisionnement en ressources
    """)

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🧭 Navigation**")
    st.markdown("Utilisez la barre latérale pour naviguer entre les pages →")

with col2:
    st.markdown("**📅 État des Données**")
    try:
        from src.data import load_data
        data = load_data()
        date_min = data["date"].min().strftime("%d/%m/%Y")
        date_max = data["date"].max().strftime("%d/%m/%Y")
        st.success(f"✅ {len(data):,} jours ({date_min} → {date_max})")
    except Exception as e:
        st.error(f"❌ Erreur : {e}")

with col3:
    st.markdown("**🤖 État du Modèle**")
    try:
        from src.models import load_model
        model = load_model()
        st.success("✅ Modèle prêt")
    except Exception:
        st.warning("⚠️ Non entraîné - visitez la page Prévisions")

with st.sidebar:
    st.markdown("### À propos")
    st.info("""
    **Système d'Aide à la Décision Hospitalière**

    Planification des capacités et analyse de scénarios.

    Version 1.0.0
    """)

    st.markdown("### Navigation")
    st.markdown("""
    1. 📊 Tableau de bord
    2. 📈 Prévisions
    3. 🔮 Scénarios
    4. 📋 Recommandations
    """)
