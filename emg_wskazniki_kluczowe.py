#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Uruchom
# python .\emg_wskazniki_kluczowe.py "badanie urazowe-1_emg.slk"

def parse_sylk(path: Path) -> pd.DataFrame:
    cells: Dict[Tuple[int, int], object] = {}
    max_row = 0
    max_col = 0
    current_row = None
    current_col = None

    text = path.read_text(encoding="utf-8", errors="ignore")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("C;"):
            continue

        parts = line.split(";")
        row = current_row
        col = current_col
        value = None

        for part in parts[1:]:
            if part.startswith("Y"):
                try:
                    row = int(part[1:])
                except ValueError:
                    pass
            elif part.startswith("X"):
                try:
                    col = int(part[1:])
                except ValueError:
                    pass
            elif part.startswith("K"):
                raw_val = part[1:]
                if len(raw_val) >= 2 and raw_val[0] == '"' and raw_val[-1] == '"':
                    raw_val = raw_val[1:-1].replace('""', '"')
                value = raw_val

        current_row = row
        current_col = col

        if row is not None and col is not None:
            cells[(row, col)] = value
            max_row = max(max_row, row)
            max_col = max(max_col, col)

    if max_row == 0 or max_col == 0:
        raise RuntimeError(f"Nie udało się sparsować pliku SYLK: {path}")

    data = []
    for r in range(1, max_row + 1):
        row_vals = []
        for c in range(1, max_col + 1):
            row_vals.append(cells.get((r, c), None))
        data.append(row_vals)

    return pd.DataFrame(data)


def clean_cell(x):
    if x is None:
        return None
    s = str(x).strip()
    return s if s != "" else None


def try_parse_number(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s == "":
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return x


def make_unique_headers(headers):
    counts = {}
    unique = []
    for h in headers:
        key = h if h is not None else "Unnamed"
        if key not in counts:
            counts[key] = 1
            unique.append(key)
        else:
            counts[key] += 1
            unique.append(f"{key}__{counts[key]}")
    return unique


def build_dataframe(raw: pd.DataFrame) -> Tuple[Dict[str, object], pd.DataFrame]:
    metadata = {
        "Name": raw.iat[0, 1] if raw.shape[0] > 0 and raw.shape[1] > 1 else None,
        "Frequency": raw.iat[1, 1] if raw.shape[0] > 1 and raw.shape[1] > 1 else None,
        "Date": raw.iat[2, 1] if raw.shape[0] > 2 and raw.shape[1] > 1 else None,
    }

    header_row_idx = 3
    headers_raw = [clean_cell(v) for v in raw.iloc[header_row_idx].tolist()]
    keep_idx = [i for i, h in enumerate(headers_raw) if h is not None]
    headers = [headers_raw[i] for i in keep_idx]

    data = raw.iloc[header_row_idx + 1:].copy()
    data = data.iloc[:, keep_idx]
    data.columns = make_unique_headers(headers)
    data = data.dropna(how="all").reset_index(drop=True)

    return metadata, data


def clean_emg_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in df.columns:
        df[col] = df[col].map(try_parse_number)

    if "Time,s" not in df.columns:
        raise RuntimeError("Brak kolumny 'Time,s' w EMG.")

    df["Time,s"] = pd.to_numeric(df["Time,s"], errors="coerce")
    df = df[df["Time,s"].notna()].copy()

    if "Activity Names" in df.columns:
        df["Activity Names"] = df["Activity Names"].astype("string")

    if "Activities" in df.columns:
        converted = pd.to_numeric(df["Activities"], errors="coerce")
        if converted.notna().any():
            df["Activities"] = converted

    # tylko kanały EMG + podstawowe kolumny organizacyjne
    keep_base = {"Time,s", "Activities", "Activity Names"}
    emg_cols = [c for c in df.columns if str(c).lower().endswith(",uv")]
    keep_cols = [c for c in df.columns if c in keep_base or c in emg_cols]
    df = df[keep_cols].copy()

    # usuwamy ewentualne duplikaty czasu
    df = df.sort_values("Time,s").drop_duplicates(subset=["Time,s"], keep="first")
    df = df.reset_index(drop=True)

    return df


def get_dt(df: pd.DataFrame) -> Optional[float]:
    t = pd.to_numeric(df["Time,s"], errors="coerce").dropna()
    if len(t) < 2:
        return None
    dt = t.diff().dropna()
    if len(dt) == 0:
        return None
    return float(dt.median())


def emg_channels(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).lower().endswith(",uv")]


def summarize_channel(df: pd.DataFrame, col: str) -> Optional[Dict[str, float]]:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return None

    abs_s = s.abs()
    q95 = float(abs_s.quantile(0.95))
    q50 = float(abs_s.quantile(0.50))
    mean_abs = float(abs_s.mean())
    rms = float((s.pow(2).mean()) ** 0.5)

    return {
        "channel": col,
        "metric": "summary",
        "mean_abs_uV": mean_abs,
        "median_abs_uV": q50,
        "p95_abs_uV": q95,
        "rms_uV": rms,
    }


def threshold_metric(df: pd.DataFrame, col: str, threshold: float) -> Optional[Dict[str, float]]:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return None

    dt = get_dt(df)
    abs_s = s.abs()
    mask = abs_s > threshold
    n = int(mask.sum())
    pct = float(mask.mean() * 100.0)
    time_s = float(n * dt) if dt is not None else None

    return {
        "channel": col,
        "metric": f"pct_time_abs_gt_{threshold}uV",
        "mean_abs_uV": None,
        "median_abs_uV": None,
        "p95_abs_uV": None,
        "rms_uV": None,
        "pct_time_above": pct,
        "estimated_time_s": time_s,
        "threshold_uV": threshold,
    }


def build_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    threshold_rows = []

    thresholds = [20.0, 50.0, 100.0]

    for ch in emg_channels(df):
        s = summarize_channel(df, ch)
        if s is not None:
            summary_rows.append(s)

        for thr in thresholds:
            t = threshold_metric(df, ch, thr)
            if t is not None:
                threshold_rows.append(t)

    return pd.DataFrame(summary_rows), pd.DataFrame(threshold_rows)


def build_global_aggregates(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows = []
    for metric_col, out_name in [
        ("mean_abs_uV", "EMG_GLOBAL_MEAN_ABS"),
        ("median_abs_uV", "EMG_GLOBAL_MEDIAN_ABS"),
        ("p95_abs_uV", "EMG_GLOBAL_P95_ABS"),
        ("rms_uV", "EMG_GLOBAL_RMS"),
    ]:
        vals = pd.to_numeric(summary_df[metric_col], errors="coerce").dropna()
        if len(vals):
            rows.append(
                {
                    "channel": out_name,
                    "metric": metric_col,
                    "value": float(vals.mean()),
                    "unit": "uV",
                }
            )

    return pd.DataFrame(rows)


def print_report(metadata: Dict[str, object], df: pd.DataFrame, summary_df: pd.DataFrame, threshold_df: pd.DataFrame, global_df: pd.DataFrame) -> None:
    print("=== KLUCZOWE WSKAŹNIKI EMG ===")
    print(f"Name: {metadata.get('Name')}")
    print(f"Frequency (metadata): {metadata.get('Frequency')}")
    print(f"Date: {metadata.get('Date')}")
    print(f"Liczba rekordów: {len(df)}")
    print(f"Liczba kanałów EMG: {len(emg_channels(df))}")

    dt = get_dt(df)
    if dt is not None:
        print(f"Mediana dt [s]: {dt}")
        print(f"Szac. freq [Hz]: {1.0 / dt}")

    print()

    if not global_df.empty:
        print("Agregaty globalne:")
        for _, row in global_df.iterrows():
            print(f"- {row['channel']} | {row['metric']}: {row['value']:.2f} {row['unit']}")
        print()

    if not summary_df.empty:
        top = summary_df.sort_values("p95_abs_uV", ascending=False).head(8)
        print("Kanały o najwyższym p95_abs_uV:")
        for _, row in top.iterrows():
            print(
                f"- {row['channel']}: "
                f"mean_abs={row['mean_abs_uV']:.2f} uV, "
                f"median_abs={row['median_abs_uV']:.2f} uV, "
                f"p95_abs={row['p95_abs_uV']:.2f} uV, "
                f"rms={row['rms_uV']:.2f} uV"
            )
        print()

    if not threshold_df.empty:
        print("Ekspozycja czasowa > 50 uV:")
        tmp = threshold_df[threshold_df["threshold_uV"] == 50.0].copy()
        tmp = tmp.sort_values("pct_time_above", ascending=False)
        for _, row in tmp.iterrows():
            print(
                f"- {row['channel']}: {row['pct_time_above']:.2f}% "
                f"(~{row['estimated_time_s']:.2f} s)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Liczenie kluczowych wskaźników EMG z pliku .slk")
    parser.add_argument("input_slk", type=Path, help="Ścieżka do pliku EMG .slk")
    parser.add_argument("--outdir", type=Path, default=Path("."), help="Katalog wynikowy")
    args = parser.parse_args()

    raw = parse_sylk(args.input_slk)
    metadata, df = build_dataframe(raw)
    df = clean_emg_dataframe(df)

    summary_df, threshold_df = build_metrics(df)
    global_df = build_global_aggregates(summary_df)

    args.outdir.mkdir(parents=True, exist_ok=True)

    clean_csv = args.outdir / "emg_clean.csv"
    summary_csv = args.outdir / "emg_summary_channels.csv"
    threshold_csv = args.outdir / "emg_thresholds.csv"
    global_csv = args.outdir / "emg_key_metrics.csv"

    df.to_csv(clean_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    threshold_df.to_csv(threshold_csv, index=False, encoding="utf-8-sig")
    global_df.to_csv(global_csv, index=False, encoding="utf-8-sig")

    print_report(metadata, df, summary_df, threshold_df, global_df)
    print()
    print(f"Zapisano: {clean_csv}")
    print(f"Zapisano: {summary_csv}")
    print(f"Zapisano: {threshold_csv}")
    print(f"Zapisano: {global_csv}")


if __name__ == "__main__":
    main()