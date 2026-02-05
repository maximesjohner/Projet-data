"""Schema inference from existing dataset."""
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


class ColumnType(Enum):
    DATE = "date"
    COUNT = "count"
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    RATIO = "ratio"


@dataclass
class ColumnSchema:
    """Schema for a single column."""
    name: str
    col_type: ColumnType
    dtype: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    na_rate: float = 0.0
    unique_values: Optional[List[Any]] = None
    distribution: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.col_type.value,
            "dtype": self.dtype,
            "min": self.min_val,
            "max": self.max_val,
            "mean": self.mean,
            "std": self.std,
            "na_rate": self.na_rate,
            "unique_values": self.unique_values,
        }


def detect_column_type(series: pd.Series, col_name: str) -> ColumnType:
    """Detect the semantic type of a column."""
    if col_name == "date":
        return ColumnType.DATE

    if series.dtype == "object":
        return ColumnType.CATEGORICAL

    unique = series.nunique()
    vmin, vmax = series.min(), series.max()

    if unique == 2 and vmin == 0 and vmax == 1:
        return ColumnType.BINARY

    if "rate" in col_name or "pct" in col_name or "ratio" in col_name or "score" in col_name:
        return ColumnType.RATIO

    if series.dtype in ["int64", "int32"] and vmin >= 0:
        return ColumnType.COUNT

    return ColumnType.CONTINUOUS


def infer_column_schema(series: pd.Series, col_name: str) -> ColumnSchema:
    """Infer schema for a single column."""
    col_type = detect_column_type(series, col_name)
    dtype = str(series.dtype)
    na_rate = series.isna().mean()

    schema = ColumnSchema(
        name=col_name,
        col_type=col_type,
        dtype=dtype,
        na_rate=na_rate,
    )

    if col_type == ColumnType.CATEGORICAL:
        schema.unique_values = list(series.dropna().unique())
    elif col_type != ColumnType.DATE:
        schema.min_val = float(series.min())
        schema.max_val = float(series.max())
        schema.mean = float(series.mean())
        schema.std = float(series.std())

        if col_type == ColumnType.COUNT:
            schema.distribution = "poisson" if schema.std**2 < schema.mean * 2 else "normal"
        else:
            schema.distribution = "normal"

    return schema


def infer_schema(df: pd.DataFrame) -> Dict[str, ColumnSchema]:
    """Infer schema for entire dataframe."""
    schema = {}
    for col in df.columns:
        schema[col] = infer_column_schema(df[col], col)
    return schema


def compute_correlations(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    """Compute correlations between numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    target_cols = [c for c in target_cols if c in numeric_df.columns]
    if not target_cols:
        return pd.DataFrame()
    return numeric_df[target_cols].corr()


def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """Complete analysis of dataset."""
    schema = infer_schema(df)

    date_cols = [k for k, v in schema.items() if v.col_type == ColumnType.DATE]
    count_cols = [k for k, v in schema.items() if v.col_type == ColumnType.COUNT]
    continuous_cols = [k for k, v in schema.items() if v.col_type == ColumnType.CONTINUOUS]
    categorical_cols = [k for k, v in schema.items() if v.col_type == ColumnType.CATEGORICAL]
    binary_cols = [k for k, v in schema.items() if v.col_type == ColumnType.BINARY]
    ratio_cols = [k for k, v in schema.items() if v.col_type == ColumnType.RATIO]

    main_cols = ["total_admissions", "available_beds", "available_staff"]
    correlations = compute_correlations(df, main_cols + count_cols[:5])

    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "schema": schema,
        "date_columns": date_cols,
        "count_columns": count_cols,
        "continuous_columns": continuous_cols,
        "categorical_columns": categorical_cols,
        "binary_columns": binary_cols,
        "ratio_columns": ratio_cols,
        "correlations": correlations,
    }


def print_schema_report(analysis: Dict[str, Any]) -> None:
    """Print schema analysis report."""
    print("=" * 60)
    print("DATASET SCHEMA ANALYSIS")
    print("=" * 60)
    print(f"\nRows: {analysis['n_rows']}, Columns: {analysis['n_cols']}")

    print(f"\nDate columns: {analysis['date_columns']}")
    print(f"Count columns ({len(analysis['count_columns'])}): {analysis['count_columns'][:5]}...")
    print(f"Continuous columns: {analysis['continuous_columns']}")
    print(f"Categorical columns: {analysis['categorical_columns']}")
    print(f"Binary columns: {analysis['binary_columns']}")
    print(f"Ratio columns: {analysis['ratio_columns']}")

    print("\n--- Column Details ---")
    for name, col_schema in analysis["schema"].items():
        if col_schema.col_type != ColumnType.CATEGORICAL:
            print(f"{name}: {col_schema.col_type.value}, "
                  f"range=[{col_schema.min_val}, {col_schema.max_val}], "
                  f"mean={col_schema.mean:.2f}" if col_schema.mean else "")
        else:
            print(f"{name}: {col_schema.col_type.value}, values={col_schema.unique_values}")
