import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
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

    def __str__(self):
        return self.name


class ActionType(Enum):
    ADD_BEDS = "Add temporary beds"
    ADD_STAFF = "Request additional staff"
    OVERTIME = "Authorize overtime"
    REDISTRIBUTE = "Redistribute patients"
    STOCK_REPLENISH = "Replenish medical supplies"
    DELAY_ELECTIVE = "Delay elective procedures"
    ALERT_ADMIN = "Alert administration"
    ACTIVATE_SURGE = "Activate surge protocol"
    EXTERNAL_SUPPORT = "Request external support"
    MONITOR = "Continue monitoring"

    def __str__(self):
        return self.value


@dataclass
class Recommendation:
    date: pd.Timestamp
    action_type: ActionType
    priority: Priority
    description: str
    quantity: Optional[float] = None
    unit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "action": str(self.action_type),
            "priority": str(self.priority),
            "description": self.description,
            "quantity": self.quantity,
            "unit": self.unit
        }


def generate_recommendations(scenario_df: pd.DataFrame, capacity_config: Optional[CAPACITY_CONFIG] = None) -> pd.DataFrame:
    logger.info("Génération des recommandations...")

    if capacity_config is None:
        capacity_config = CAPACITY_CONFIG

    recommendations = []

    for idx, row in scenario_df.iterrows():
        date = row.get("date", idx)
        predicted_load = row.get("scenario_admissions", row.get("predicted_admissions", 0))
        capacity_gap = row.get("capacity_gap", 0)
        occupancy_rate = row.get("occupancy_rate", 0)
        effective_staff = row.get("effective_staff", capacity_config.total_staff)
        effective_stock = row.get("effective_stock_pct", 75)
        is_overcapacity = row.get("is_overcapacity", False)
        is_critical = row.get("is_critical", False)

        day_recs = []

        if is_overcapacity and capacity_gap > 0:
            extra_beds_needed = int(np.ceil(capacity_gap * 1.1))
            if capacity_gap > 50:
                day_recs.append(Recommendation(
                    date=date, action_type=ActionType.ACTIVATE_SURGE, priority=Priority.CRITICAL,
                    description=f"Activate surge protocol - {int(capacity_gap)} patients over capacity",
                    quantity=extra_beds_needed, unit="beds"
                ))
                day_recs.append(Recommendation(
                    date=date, action_type=ActionType.EXTERNAL_SUPPORT, priority=Priority.CRITICAL,
                    description="Request support from neighboring hospitals",
                    quantity=int(capacity_gap * 0.3), unit="patient transfers"
                ))
            else:
                day_recs.append(Recommendation(
                    date=date, action_type=ActionType.ADD_BEDS, priority=Priority.HIGH,
                    description="Add temporary beds to handle overflow",
                    quantity=extra_beds_needed, unit="beds"
                ))

        staff_ratio = effective_staff / capacity_config.total_staff
        if staff_ratio < 0.85:
            staff_needed = int((capacity_config.total_staff - effective_staff) * 0.8)
            if staff_ratio < 0.7:
                day_recs.append(Recommendation(
                    date=date, action_type=ActionType.ADD_STAFF, priority=Priority.CRITICAL,
                    description="Critical staff shortage - request emergency staffing",
                    quantity=staff_needed, unit="staff members"
                ))
            else:
                day_recs.append(Recommendation(
                    date=date, action_type=ActionType.OVERTIME, priority=Priority.HIGH,
                    description="Authorize overtime to cover staff gaps",
                    quantity=staff_needed * 4, unit="overtime hours"
                ))

        if not is_overcapacity and occupancy_rate > capacity_config.warning_occupancy_threshold:
            day_recs.append(Recommendation(
                date=date, action_type=ActionType.ALERT_ADMIN, priority=Priority.MEDIUM,
                description=f"Occupancy at {occupancy_rate:.0%} - approaching capacity",
                quantity=occupancy_rate * 100, unit="% occupancy"
            ))
            if occupancy_rate > 0.8:
                day_recs.append(Recommendation(
                    date=date, action_type=ActionType.DELAY_ELECTIVE, priority=Priority.MEDIUM,
                    description="Consider postponing non-urgent elective procedures",
                    quantity=None, unit=None
                ))

        if effective_stock < capacity_config.min_stock_level:
            day_recs.append(Recommendation(
                date=date, action_type=ActionType.STOCK_REPLENISH,
                priority=Priority.HIGH if effective_stock < 40 else Priority.MEDIUM,
                description=f"Medical supplies at {effective_stock:.0f}% - below minimum threshold",
                quantity=capacity_config.min_stock_level - effective_stock + 20, unit="% of normal stock"
            ))

        if is_critical:
            day_recs.append(Recommendation(
                date=date, action_type=ActionType.REDISTRIBUTE, priority=Priority.HIGH,
                description="Redistribute patients across departments to balance load",
                quantity=None, unit=None
            ))

        if not day_recs:
            day_recs.append(Recommendation(
                date=date, action_type=ActionType.MONITOR, priority=Priority.LOW,
                description="Operations normal - continue standard monitoring",
                quantity=None, unit=None
            ))

        recommendations.extend(day_recs)

    recs_df = pd.DataFrame([r.to_dict() for r in recommendations])

    critical = (recs_df["priority"] == "CRITICAL").sum()
    high = (recs_df["priority"] == "HIGH").sum()
    logger.info(f"Recommandations générées: {len(recs_df)} total, {critical} critiques, {high} haute priorité")

    return recs_df


def get_priority_actions(recommendations_df: pd.DataFrame, max_priority: Priority = Priority.HIGH) -> pd.DataFrame:
    priority_order = {str(p): p.value for p in Priority}
    max_value = max_priority.value

    filtered = recommendations_df[
        recommendations_df["priority"].apply(lambda x: priority_order.get(x, 4) <= max_value)
    ].copy()

    filtered["priority_value"] = filtered["priority"].apply(lambda x: priority_order.get(x, 4))
    filtered = filtered.sort_values(["priority_value", "date"]).drop(columns=["priority_value"])

    return filtered


def summarize_recommendations(recommendations_df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "total_recommendations": len(recommendations_df),
        "by_priority": recommendations_df["priority"].value_counts().to_dict(),
        "by_action": recommendations_df["action"].value_counts().to_dict(),
        "critical_count": (recommendations_df["priority"] == "CRITICAL").sum(),
        "high_count": (recommendations_df["priority"] == "HIGH").sum(),
        "dates_with_issues": recommendations_df[
            recommendations_df["priority"].isin(["CRITICAL", "HIGH"])
        ]["date"].nunique()
    }
