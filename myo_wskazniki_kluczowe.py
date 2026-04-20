#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Uruchom
# python .\myo_wskazniki_kluczowe.py "badanie urazowe_Myo_clean.csv"

def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
        dtype={"Activity Names": "string"},
    )


def get_dt(df: pd.DataFrame) -> Optional[float]:
    if "Time,s" not in df.columns or len(df) < 2:
        return None
    t = pd.to_numeric(df["Time,s"], errors="coerce").dropna()
    if len(t) < 2:
        return None
    dt = t.diff().dropna()
    if len(dt) == 0:
        return None
    return float(dt.median())


def num_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() == 0:
        return None
    return s


def exposure_metrics(df: pd.DataFrame, col: str, threshold: float) -> Optional[Dict[str, float]]:
    s = num_series(df, col)
    if s is None:
        return None

    dt = get_dt(df)
    valid = s.dropna()
    if len(valid) == 0:
        return None

    mask = valid.abs() > threshold
    n = int(mask.sum())
    pct = float(mask.mean() * 100.0)
    time_s = float(n * dt) if dt is not None else None

    return {
        "column": col,
        "metric": f"pct_time_abs_gt_{threshold}",
        "value": pct,
        "unit": "%",
        "n_samples_above": n,
        "estimated_time_s": time_s,
    }


def percentile_metric(df: pd.DataFrame, col: str, q: float, abs_value: bool = True) -> Optional[Dict[str, float]]:
    s = num_series(df, col)
    if s is None:
        return None

    valid = s.dropna()
    if abs_value:
        valid = valid.abs()

    if len(valid) == 0:
        return None

    return {
        "column": col,
        "metric": f"p{int(q * 100)}" + ("_abs" if abs_value else ""),
        "value": float(valid.quantile(q)),
        "unit": "deg",
        "n_samples_above": None,
        "estimated_time_s": None,
    }


def median_abs_metric(df: pd.DataFrame, col: str) -> Optional[Dict[str, float]]:
    s = num_series(df, col)
    if s is None:
        return None

    valid = s.dropna().abs()
    if len(valid) == 0:
        return None

    return {
        "column": col,
        "metric": "median_abs",
        "value": float(valid.median()),
        "unit": "deg",
        "n_samples_above": None,
        "estimated_time_s": None,
    }


def aggregate_mean(metrics: List[Dict[str, float]], metric_name: str, output_column: str, unit: str) -> Optional[Dict[str, float]]:
    vals = [m["value"] for m in metrics if m is not None and m["metric"] == metric_name]
    if not vals:
        return None

    return {
        "column": output_column,
        "metric": metric_name + "_mean",
        "value": float(sum(vals) / len(vals)),
        "unit": unit,
        "n_samples_above": None,
        "estimated_time_s": None,
    }


def build_key_metrics(df: pd.DataFrame) -> pd.DataFrame:
    results: List[Optional[Dict[str, float]]] = []

    # Ekspozycje czasowe – najbardziej użyteczne do interpretacji ergonomicznej
    exposure_rules = [
        ("Cervical Flexion,deg", 20),
        ("Lumbar Flexion,deg", 20),
        ("Thoracic Flexion,deg", 20),
        ("Shoulder Flexion LT,deg", 60),
        ("Shoulder Flexion RT,deg", 60),
        ("Shoulder Abduction LT,deg", 60),
        ("Shoulder Abduction RT,deg", 60),
    ]

    for col, thr in exposure_rules:
        results.append(exposure_metrics(df, col, thr))

    # Percentyle / mediany – stabilniejsze niż sam max i range
    p95_targets = [
        "Cervical Flexion,deg",
        "Lumbar Flexion,deg",
        "Thoracic Flexion,deg",
        "Shoulder Flexion LT,deg",
        "Shoulder Flexion RT,deg",
        "Shoulder Abduction LT,deg",
        "Shoulder Abduction RT,deg",
    ]

    med_targets = [
        "Cervical Flexion,deg",
        "Lumbar Flexion,deg",
        "Thoracic Flexion,deg",
        "Shoulder Flexion LT,deg",
        "Shoulder Flexion RT,deg",
    ]

    p95_metrics = []
    for col in p95_targets:
        m = percentile_metric(df, col, 0.95, abs_value=True)
        results.append(m)
        if m is not None:
            p95_metrics.append(m)

    median_metrics = []
    for col in med_targets:
        m = median_abs_metric(df, col)
        results.append(m)
        if m is not None:
            median_metrics.append(m)

    # Proste agregaty regionalne
    trunk_exposures = [
        exposure_metrics(df, "Lumbar Flexion,deg", 20),
        exposure_metrics(df, "Thoracic Flexion,deg", 20),
    ]
    shoulder_exposures = [
        exposure_metrics(df, "Shoulder Flexion LT,deg", 60),
        exposure_metrics(df, "Shoulder Flexion RT,deg", 60),
        exposure_metrics(df, "Shoulder Abduction LT,deg", 60),
        exposure_metrics(df, "Shoulder Abduction RT,deg", 60),
    ]

    trunk_p95 = [
        percentile_metric(df, "Lumbar Flexion,deg", 0.95, abs_value=True),
        percentile_metric(df, "Thoracic Flexion,deg", 0.95, abs_value=True),
    ]
    shoulder_p95 = [
        percentile_metric(df, "Shoulder Flexion LT,deg", 0.95, abs_value=True),
        percentile_metric(df, "Shoulder Flexion RT,deg", 0.95, abs_value=True),
        percentile_metric(df, "Shoulder Abduction LT,deg", 0.95, abs_value=True),
        percentile_metric(df, "Shoulder Abduction RT,deg", 0.95, abs_value=True),
    ]

    results.append(
        aggregate_mean(trunk_exposures, "pct_time_abs_gt_20", "TRUNK_LOAD", "%")
    )
    results.append(
        aggregate_mean(shoulder_exposures, "pct_time_abs_gt_60", "SHOULDER_LOAD", "%")
    )
    results.append(
        aggregate_mean(trunk_p95, "p95_abs", "TRUNK_P95", "deg")
    )
    results.append(
        aggregate_mean(shoulder_p95, "p95_abs", "SHOULDER_P95", "deg")
    )

    rows = [r for r in results if r is not None]
    return pd.DataFrame(rows)


def print_report(metrics: pd.DataFrame) -> None:
    print("=== KLUCZOWE WSKAŹNIKI MYOMOTION ===")
    if metrics.empty:
        print("Brak wyników.")
        return

    order = [
        "TRUNK_LOAD",
        "SHOULDER_LOAD",
        "TRUNK_P95",
        "SHOULDER_P95",
        "Cervical Flexion,deg",
        "Lumbar Flexion,deg",
        "Thoracic Flexion,deg",
        "Shoulder Flexion LT,deg",
        "Shoulder Flexion RT,deg",
        "Shoulder Abduction LT,deg",
        "Shoulder Abduction RT,deg",
    ]

    tmp = metrics.copy()
    tmp["order"] = tmp["column"].apply(lambda x: order.index(x) if x in order else 999)
    tmp = tmp.sort_values(["order", "column", "metric"]).drop(columns=["order"])

    for _, row in tmp.iterrows():
        val = row["value"]
        unit = row["unit"]
        metric = row["metric"]
        col = row["column"]

        if pd.notna(row.get("estimated_time_s", None)):
            print(f"- {col} | {metric}: {val:.2f} {unit} (~{row['estimated_time_s']:.2f} s)")
        else:
            print(f"- {col} | {metric}: {val:.2f} {unit}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Liczenie kluczowych wskaźników MyoMotion z cleaned CSV")
    parser.add_argument("input_csv", type=Path, help="Ścieżka do pliku MyoMotion clean CSV")
    parser.add_argument("--out", type=Path, default=Path("myo_key_metrics.csv"), help="Ścieżka do pliku wynikowego CSV")
    args = parser.parse_args()

    df = load_data(args.input_csv)
    metrics = build_key_metrics(df)
    metrics.to_csv(args.out, index=False, encoding="utf-8-sig")

    print_report(metrics)
    print()
    print(f"Zapisano: {args.out}")


if __name__ == "__main__":
    main()