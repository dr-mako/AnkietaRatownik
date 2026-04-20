#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Uruchomienie
# python .\analiza_myo_motion.py "badanie urazowe_Myo_clean.csv"

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, dtype={"Activity Names": "string"})
    return df


def estimate_dt(df: pd.DataFrame) -> float:
    if "Time,s" not in df.columns or len(df) < 2:
        return 0.0
    t = pd.to_numeric(df["Time,s"], errors="coerce").dropna()
    if len(t) < 2:
        return 0.0
    dt = t.diff().dropna()
    if len(dt) == 0:
        return 0.0
    return float(dt.median())


def numeric_columns(df: pd.DataFrame) -> List[str]:
    out = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            out.append(col)
    return out


def build_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for col in numeric_columns(df):
        if col == "Activities":
            continue

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue

        rows.append(
            {
                "column": col,
                "n": int(len(s)),
                "min": float(s.min()),
                "p5": float(s.quantile(0.05)),
                "median": float(s.median()),
                "mean": float(s.mean()),
                "p95": float(s.quantile(0.95)),
                "max": float(s.max()),
                "range": float(s.max() - s.min()),
                "std": float(s.std()) if len(s) > 1 else 0.0,
            }
        )

    return pd.DataFrame(rows)


def threshold_report(df: pd.DataFrame, dt: float) -> pd.DataFrame:
    """
    Proste progi robocze / ergonomiczne.
    To nie jest jeszcze pełna walidowana skala kliniczna,
    tylko techniczny raport ekspozycji.
    """
    rules = [
        ("Cervical Flexion,deg", 20),
        ("Cervical Flexion,deg", 40),
        ("Lumbar Flexion,deg", 20),
        ("Lumbar Flexion,deg", 45),
        ("Thoracic Flexion,deg", 20),
        ("Thoracic Flexion,deg", 45),
        ("Shoulder Flexion LT,deg", 60),
        ("Shoulder Flexion RT,deg", 60),
        ("Shoulder Abduction LT,deg", 60),
        ("Shoulder Abduction RT,deg", 60),
        ("Hip Flexion LT,deg", 45),
        ("Hip Flexion RT,deg", 45),
    ]

    rows = []
    for col, threshold in rules:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        mask = s.abs() > threshold
        n = int(mask.sum())
        pct = float(mask.mean() * 100.0) if len(mask) else 0.0
        time_s = float(n * dt) if dt > 0 else None

        rows.append(
            {
                "column": col,
                "threshold_abs_deg": threshold,
                "n_samples_above": n,
                "pct_time_above": pct,
                "estimated_time_s_above": time_s,
            }
        )

    return pd.DataFrame(rows)


def regional_report(df: pd.DataFrame) -> pd.DataFrame:
    regions: Dict[str, List[str]] = {
        "neck": [
            "Cervical Flexion,deg",
            "Cervical Lateral - RT,deg",
            "Cervical Axial - RT,deg",
        ],
        "trunk": [
            "Thoracic Flexion,deg",
            "Thoracic Lateral - RT,deg",
            "Thoracic Axial - RT,deg",
            "Lumbar Flexion,deg",
            "Lumbar Lateral - RT,deg",
            "Lumbar Axial - RT,deg",
        ],
        "shoulder_left": [
            "Shoulder Total Flexion LT,deg",
            "Shoulder Flexion LT,deg",
            "Shoulder Abduction LT,deg",
            "Shoulder Rotation - out LT,deg",
        ],
        "shoulder_right": [
            "Shoulder Total Flexion RT,deg",
            "Shoulder Flexion RT,deg",
            "Shoulder Abduction RT,deg",
            "Shoulder Rotation - out RT,deg",
        ],
        "hip_left": [
            "Hip Flexion LT,deg",
            "Hip Abduction LT,deg",
            "Hip Rotation - out LT,deg",
        ],
        "hip_right": [
            "Hip Flexion RT,deg",
            "Hip Abduction RT,deg",
            "Hip Rotation - out RT,deg",
        ],
        "knee_left": [
            "Knee Flexion LT,deg",
            "Knee rotation - out LT,deg",
            "Knee abduction LT,deg",
        ],
        "knee_right": [
            "Knee Flexion RT,deg",
            "Knee rotation - out RT,deg",
            "Knee abduction RT,deg",
        ],
    }

    rows = []

    for region, cols in regions.items():
        existing = [c for c in cols if c in df.columns]
        if not existing:
            continue

        values = []
        for col in existing:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s):
                values.append(
                    {
                        "column": col,
                        "median_abs": float(s.abs().median()),
                        "p95_abs": float(s.abs().quantile(0.95)),
                        "range": float(s.max() - s.min()),
                    }
                )

        if not values:
            continue

        tmp = pd.DataFrame(values)
        rows.append(
            {
                "region": region,
                "n_columns": int(len(existing)),
                "median_abs_mean": float(tmp["median_abs"].mean()),
                "p95_abs_mean": float(tmp["p95_abs"].mean()),
                "range_mean": float(tmp["range"].mean()),
            }
        )

    return pd.DataFrame(rows)


def print_short_report(df: pd.DataFrame, summary: pd.DataFrame, threshold: pd.DataFrame, regional: pd.DataFrame, dt: float) -> None:
    print("=== ANALIZA MYOMOTION ===")
    print(f"Liczba rekordów: {len(df)}")
    print(f"Liczba kolumn: {len(df.columns)}")
    print(f"Mediana dt [s]: {dt}")

    if "Time,s" in df.columns and len(df) > 0:
        t = pd.to_numeric(df["Time,s"], errors="coerce").dropna()
        if len(t) > 0:
            print(f"Start [s]: {float(t.iloc[0])}")
            print(f"Koniec [s]: {float(t.iloc[-1])}")
            print(f"Czas trwania [s]: {float(t.iloc[-1] - t.iloc[0])}")

    print()
    print("Największe zakresy ruchu (top 10):")
    if not summary.empty:
        top = summary.sort_values("range", ascending=False).head(10)
        for _, row in top.iterrows():
            print(f"  - {row['column']}: range={row['range']:.3f}, p95={row['p95']:.3f}")

    print()
    print("Ekspozycja powyżej progów:")
    if not threshold.empty:
        for _, row in threshold.iterrows():
            print(
                f"  - {row['column']} > {row['threshold_abs_deg']}°: "
                f"{row['pct_time_above']:.2f}% czasu "
                f"(~{row['estimated_time_s_above']:.2f} s)"
            )

    print()
    print("Podsumowanie regionalne:")
    if not regional.empty:
        for _, row in regional.iterrows():
            print(
                f"  - {row['region']}: "
                f"median_abs_mean={row['median_abs_mean']:.2f}, "
                f"p95_abs_mean={row['p95_abs_mean']:.2f}, "
                f"range_mean={row['range_mean']:.2f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza cleaned CSV z MyoMotion")
    parser.add_argument("input_csv", type=Path, help="Ścieżka do cleaned CSV")
    parser.add_argument("--outdir", type=Path, default=Path("."), help="Katalog wynikowy")
    args = parser.parse_args()

    df = load_data(args.input_csv)
    dt = estimate_dt(df)

    summary = build_summary_table(df)
    threshold = threshold_report(df, dt)
    regional = regional_report(df)

    args.outdir.mkdir(parents=True, exist_ok=True)

    summary_path = args.outdir / "myo_summary_all_columns.csv"
    threshold_path = args.outdir / "myo_threshold_report.csv"
    regional_path = args.outdir / "myo_regional_report.csv"

    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    threshold.to_csv(threshold_path, index=False, encoding="utf-8-sig")
    regional.to_csv(regional_path, index=False, encoding="utf-8-sig")

    print_short_report(df, summary, threshold, regional, dt)
    print()
    print(f"Zapisano: {summary_path}")
    print(f"Zapisano: {threshold_path}")
    print(f"Zapisano: {regional_path}")


if __name__ == "__main__":
    main()