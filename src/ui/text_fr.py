"""
Constantes de texte en français pour l'interface utilisateur.
Centralise toutes les chaînes pour éviter les régressions et garantir la cohérence.
"""

# === NAVIGATION ===
NAV_HOME = "Accueil"
NAV_DASHBOARD = "Tableau de bord"
NAV_FORECAST = "Prévisions"
NAV_SCENARIOS = "Scénarios"
NAV_RECOMMENDATIONS = "Recommandations"

# === BOUTONS ===
BTN_DOWNLOAD = "Télécharger"
BTN_DOWNLOAD_CSV = "📥 Télécharger (CSV)"
BTN_DOWNLOAD_REPORT = "📄 Télécharger le Rapport (TXT)"
BTN_RETRAIN = "🔄 Réentraîner le Modèle"
BTN_REFRESH = "Rafraîchir"
BTN_RUN = "Lancer"
BTN_ACCESS_DASHBOARD = "Accéder au tableau de bord"

# === ACTIONS DE RECOMMANDATION ===
ACTION_ADD_BEDS = "Ajouter des lits temporaires"
ACTION_ADD_STAFF = "Demander du personnel supplémentaire"
ACTION_OVERTIME = "Autoriser les heures supplémentaires"
ACTION_REDISTRIBUTE = "Redistribuer les patients"
ACTION_STOCK_REPLENISH = "Réapprovisionner les fournitures médicales"
ACTION_DELAY_ELECTIVE = "Reporter les interventions non urgentes"
ACTION_ALERT_ADMIN = "Alerter l'administration"
ACTION_ACTIVATE_SURGE = "Activer le protocole de surcharge"
ACTION_EXTERNAL_SUPPORT = "Demander un renfort externe"
ACTION_MONITOR = "Continuer la surveillance"

# === DESCRIPTIONS DE RECOMMANDATION ===
DESC_SURGE_PROTOCOL = "Activer le protocole de surcharge - {gap} au-dessus de la capacité"
DESC_EXTERNAL_SUPPORT = "Demander le soutien des hôpitaux voisins"
DESC_ADD_BEDS = "Ajouter des lits temporaires"
DESC_CRITICAL_STAFF = "Pénurie critique de personnel"
DESC_OVERTIME = "Autoriser les heures supplémentaires"
DESC_OCCUPANCY = "Occupation à {rate}"
DESC_DELAY_ELECTIVE = "Envisager de reporter les interventions programmées"
DESC_STOCK_LOW = "Fournitures à {level}%"
DESC_REDISTRIBUTE = "Redistribuer les patients entre les services"
DESC_NORMAL = "Opérations normales"

# === UNITÉS ===
UNIT_BEDS = "lits"
UNIT_STAFF = "personnel"
UNIT_HOURS = "heures"
UNIT_TRANSFERS = "transferts"
UNIT_STOCK_PCT = "% stock"
UNIT_OCCUPANCY_PCT = "% occupation"
UNIT_MINUTES = "min"
UNIT_EURO_DAY = "€/jour"

# === PRIORITÉS ===
PRIORITY_CRITICAL = "CRITIQUE"
PRIORITY_HIGH = "HAUTE"
PRIORITY_MEDIUM = "MOYENNE"
PRIORITY_LOW = "BASSE"

PRIORITY_MAP = {
    "CRITICAL": PRIORITY_CRITICAL,
    "HIGH": PRIORITY_HIGH,
    "MEDIUM": PRIORITY_MEDIUM,
    "LOW": PRIORITY_LOW
}

# === SCÉNARIOS ===
SCENARIO_REFERENCE = "Référence"
SCENARIO_MILD_EPIDEMIC = "Épidémie Légère"
SCENARIO_SEVERE_EPIDEMIC = "Épidémie Sévère"
SCENARIO_STAFF_STRIKE = "Grève du Personnel"
SCENARIO_WINTER_PEAK = "Pic Hivernal"
SCENARIO_SUMMER_HEATWAVE = "Canicule Estivale"
SCENARIO_MAJOR_ACCIDENT = "Accident Majeur"
SCENARIO_CUSTOM = "Personnalisé"

# === LABELS DE GRAPHIQUES ===
CHART_DATE = "Date"
CHART_ADMISSIONS = "Admissions"
CHART_DAILY_ADMISSIONS = "Admissions quotidiennes"
CHART_TOTAL_ADMISSIONS = "Admissions totales"
CHART_OCCUPANCY = "Occupation (%)"
CHART_OCCUPANCY_RATE = "Taux d'occupation"
CHART_CAPACITY = "Capacité"
CHART_EFFECTIVE_CAPACITY = "Capacité effective"
CHART_FORECAST = "Prévision"
CHART_HISTORICAL = "Historique"
CHART_REFERENCE = "Référence"
CHART_SCENARIO = "Scénario"
CHART_CONFIDENCE_INTERVAL = "Intervalle de confiance (±15%)"
CHART_VARIATION = "Variation (%)"
CHART_GAP = "Écart (patients)"
CHART_WAITING_TIME = "Temps d'attente (min)"
CHART_STOCK_LEVEL = "Niveau de stock (%)"
CHART_STAFF = "Personnel"
CHART_BEDS = "Lits"

# === MÉTRIQUES ===
METRIC_AVG_DAILY = "Admissions Moy./Jour"
METRIC_PEAK_DAY = "Jour de Pointe"
METRIC_TOTAL = "Total Admissions"
METRIC_R2 = "Score R²"
METRIC_MAE = "MAE"
METRIC_RMSE = "RMSE"
METRIC_MAPE = "MAPE"
METRIC_DAYS_OVERCAPACITY = "Jours en Surcapacité"
METRIC_AVG_OCCUPANCY = "Occupation Moy."
METRIC_CRITICAL_ACTIONS = "Actions Critiques"
METRIC_HIGH_PRIORITY = "Priorité Haute"
METRIC_TOTAL_RECOMMENDATIONS = "Total Recommandations"
METRIC_PROBLEM_DAYS = "Jours avec Problèmes"

# === MESSAGES ===
MSG_NO_DATA = "Aucune donnée disponible"
MSG_FILE_NOT_FOUND = "Fichier de données introuvable"
MSG_MODEL_NOT_TRAINED = "Modèle non entraîné"
MSG_MODEL_READY = "Modèle prêt"
MSG_LOADING = "Chargement..."
MSG_GENERATING = "Génération des prévisions..."
MSG_GENERATING_RECO = "Génération des recommandations..."
MSG_CACHE_CLEARED = "Cache vidé ! Réentraînement du modèle..."
MSG_SYNTHETIC_DATA = "Données synthétiques - Démonstrateur"

# === TITRES DE PAGES ===
TITLE_HOME = "🏥 Système d'Aide à la Décision Hospitalière"
TITLE_DASHBOARD = "📊 Tableau de Bord Hospitalier"
TITLE_FORECAST = "📈 Prévisions des Admissions"
TITLE_SCENARIOS = "🔮 Simulation de Scénarios"
TITLE_RECOMMENDATIONS = "📋 Recommandations d'Actions"

# === SECTIONS ===
SECTION_KPI = "Indicateurs Clés de Performance"
SECTION_TIME_SERIES = "Analyse des Séries Temporelles"
SECTION_DISTRIBUTION = "Distribution et Tendances"
SECTION_CORRELATION = "Analyse des Corrélations"
SECTION_EXPLORER = "Explorateur de Relations"
SECTION_IMPACT_SUMMARY = "Résumé de l'Impact"
SECTION_EXECUTIVE_SUMMARY = "Résumé Exécutif"
SECTION_ACTION_PLAN = "Plan d'Action"
SECTION_RESOURCE_NEEDS = "Résumé des Besoins en Ressources"

# === FILTRES ===
FILTER_DATE_RANGE = "Plage de dates"
FILTER_AGGREGATION = "Agrégation"
FILTER_VARIABLE = "Variable à analyser"
FILTER_PRIORITY = "Niveaux de priorité"
FILTER_HORIZON = "Horizon (jours)"
FILTER_START_DATE = "Date de début"
FILTER_MODEL = "Modèle"

# === AGRÉGATIONS ===
AGG_DAILY = "Journalier"
AGG_WEEKLY = "Hebdomadaire"
AGG_MONTHLY = "Mensuel"

# === MODÈLES ===
MODEL_RANDOM_FOREST = "Modèle Final (ML)"
MODEL_BASELINE = "Modèle de Référence (Saisonnier)"

# === JOURS DE LA SEMAINE ===
DAYS_SHORT = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
DAYS_FULL = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]

# === MOIS ===
MONTHS = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]

# === SAISONS ===
SEASON_WINTER = "Hiver"
SEASON_SPRING = "Printemps"
SEASON_SUMMER = "Été"
SEASON_AUTUMN = "Automne"

SEASON_MAP = {
    "winter": SEASON_WINTER,
    "spring": SEASON_SPRING,
    "summer": SEASON_SUMMER,
    "autumn": SEASON_AUTUMN
}

# === ALERTES ===
ALERT_CRITICAL = "⚠️ **ALERTE** : {count} actions critiques requises !"
ALERT_HIGH = "**Attention** : {count} actions de haute priorité identifiées."
ALERT_OK = "✅ **Tout va bien** : Aucun problème critique identifié pour ce scénario."

# === RÈGLES DE DÉCISION ===
DECISION_RULES_TITLE = "Règles de Décision"
DECISION_RULES_TEXT = """
Les recommandations sont générées selon les règles suivantes :
- **Surcapacité** : Si demande > capacité effective → Ajouter lits / Activer protocole surcharge
- **Pénurie personnel** : Si personnel < 85% → Heures supplémentaires ; < 70% → Renfort critique
- **Occupation élevée** : Si > 75% → Alerte ; > 80% → Reporter interventions programmées
- **Stocks bas** : Si < 50% → Réapprovisionnement prioritaire ; < 40% → Réapprovisionnement critique
- **Situation critique** : Si occupation > 85% → Redistribuer les patients
"""

# === WORKFLOW ===
WORKFLOW_STEPS = ["Observer", "Prévoir", "Simuler", "Décider"]

# === FORMAT DATE ===
DATE_FORMAT_DISPLAY = "%d/%m/%Y"
DATE_FORMAT_ISO = "%Y-%m-%d"
