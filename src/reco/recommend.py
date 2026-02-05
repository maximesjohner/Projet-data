import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import CAPACITY_CONFIG, logger


class Priority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    def __str__(self): return self.name

    def to_fr(self) -> str:
        mapping = {"CRITICAL": "CRITIQUE", "HIGH": "HAUTE", "MEDIUM": "MOYENNE", "LOW": "BASSE"}
        return mapping.get(self.name, self.name)


class ActionType(Enum):
    ADD_BEDS = "Ajouter des lits temporaires"
    ADD_STAFF = "Demander du personnel supplémentaire"
    OVERTIME = "Autoriser les heures supplémentaires"
    REDISTRIBUTE = "Redistribuer les patients"
    STOCK_REPLENISH = "Réapprovisionner les fournitures"
    DELAY_ELECTIVE = "Reporter les interventions programmées"
    ALERT_ADMIN = "Alerter l'administration"
    ACTIVATE_SURGE = "Activer le protocole de surcharge"
    EXTERNAL_SUPPORT = "Demander un renfort externe"
    MONITOR = "Continuer la surveillance"
    def __str__(self): return self.value


@dataclass
class Recommendation:
    date: pd.Timestamp
    action_type: ActionType
    priority: Priority
    description: str
    quantity: Optional[float] = None
    unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"date": self.date, "action": str(self.action_type), "priority": self.priority.to_fr(),
                "description": self.description, "quantity": self.quantity, "unit": self.unit}


def generate_recommendations(scenario_df: pd.DataFrame, capacity_config=None) -> pd.DataFrame:
    logger.info("Génération des recommandations...")
    cfg = capacity_config or CAPACITY_CONFIG
    recommendations = []

    for idx, row in scenario_df.iterrows():
        date = row.get("date", idx)
        capacity_gap = row.get("capacity_gap", 0)
        occupancy_rate = row.get("occupancy_rate", 0)
        effective_staff = row.get("effective_staff", cfg.total_staff)
        effective_stock = row.get("effective_stock_pct", 75)
        is_overcapacity = row.get("is_overcapacity", False)
        is_critical = row.get("is_critical", False)
        day_recs = []

        if is_overcapacity and capacity_gap > 0:
            extra_beds = int(np.ceil(capacity_gap * 1.1))
            if capacity_gap > 50:
                day_recs.append(Recommendation(date, ActionType.ACTIVATE_SURGE, Priority.CRITICAL,
                    f"Activer le protocole de surcharge - {int(capacity_gap)} au-dessus de la capacité", extra_beds, "lits"))
                day_recs.append(Recommendation(date, ActionType.EXTERNAL_SUPPORT, Priority.CRITICAL,
                    "Demander le soutien des hôpitaux voisins", int(capacity_gap * 0.3), "transferts"))
            else:
                day_recs.append(Recommendation(date, ActionType.ADD_BEDS, Priority.HIGH,
                    "Ajouter des lits temporaires", extra_beds, "lits"))

        staff_ratio = effective_staff / cfg.total_staff
        if staff_ratio < 0.85:
            staff_needed = int((cfg.total_staff - effective_staff) * 0.8)
            if staff_ratio < 0.7:
                day_recs.append(Recommendation(date, ActionType.ADD_STAFF, Priority.CRITICAL,
                    "Pénurie critique de personnel", staff_needed, "personnel"))
            else:
                day_recs.append(Recommendation(date, ActionType.OVERTIME, Priority.HIGH,
                    "Autoriser les heures supplémentaires", staff_needed * 4, "heures"))

        if not is_overcapacity and occupancy_rate > cfg.warning_occupancy_threshold:
            day_recs.append(Recommendation(date, ActionType.ALERT_ADMIN, Priority.MEDIUM,
                f"Occupation à {occupancy_rate:.0%}", occupancy_rate * 100, "% occupation"))
            if occupancy_rate > 0.8:
                day_recs.append(Recommendation(date, ActionType.DELAY_ELECTIVE, Priority.MEDIUM,
                    "Envisager de reporter les interventions programmées"))

        if effective_stock < cfg.min_stock_level:
            day_recs.append(Recommendation(date, ActionType.STOCK_REPLENISH,
                Priority.HIGH if effective_stock < 40 else Priority.MEDIUM,
                f"Fournitures à {effective_stock:.0f}%", cfg.min_stock_level - effective_stock + 20, "% stock"))

        if is_critical:
            day_recs.append(Recommendation(date, ActionType.REDISTRIBUTE, Priority.HIGH,
                "Redistribuer les patients entre les services"))

        if not day_recs:
            day_recs.append(Recommendation(date, ActionType.MONITOR, Priority.LOW, "Opérations normales"))

        recommendations.extend(day_recs)

    recs_df = pd.DataFrame([r.to_dict() for r in recommendations])
    critical = (recs_df["priority"] == "CRITIQUE").sum()
    high = (recs_df["priority"] == "HAUTE").sum()
    logger.info(f"Recommandations: {len(recs_df)} total, {critical} critiques, {high} haute priorité")
    return recs_df


def get_priority_actions(recommendations_df: pd.DataFrame, max_priority: Priority = Priority.HIGH) -> pd.DataFrame:
    priority_order = {"CRITIQUE": 1, "HAUTE": 2, "MOYENNE": 3, "BASSE": 4}
    filtered = recommendations_df[
        recommendations_df["priority"].apply(lambda x: priority_order.get(x, 4) <= max_priority.value)
    ].copy()
    filtered["_pv"] = filtered["priority"].apply(lambda x: priority_order.get(x, 4))
    return filtered.sort_values(["_pv", "date"]).drop(columns=["_pv"])


def summarize_recommendations(recommendations_df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "total_recommendations": len(recommendations_df),
        "by_priority": recommendations_df["priority"].value_counts().to_dict(),
        "critical_count": (recommendations_df["priority"] == "CRITIQUE").sum(),
        "high_count": (recommendations_df["priority"] == "HAUTE").sum(),
        "dates_with_issues": recommendations_df[
            recommendations_df["priority"].isin(["CRITIQUE", "HAUTE"])
        ]["date"].nunique()
    }
