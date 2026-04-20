#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

'''
Ten plik ma robić automatyczny audyt obu eksportów — czyli nie analizę kliniczną jeszcze, tylko techniczne sprawdzenie, z czym w ogóle mamy do czynienia.

W praktyce miał:

wczytać oba .slk,
znaleźć wiersz Time,s,
wyciągnąć metadane (Name, Frequency, Date),
policzyć liczbę rekordów i kolumn,
sprawdzić oś czasu i oszacować realną częstotliwość,
wypisać markery Activities / Activity Names,
policzyć braki danych w kolumnach,
porównać EMG z MyoMotion


Uruchom:
python .\audyt_emg_myo_bez_xlrd.py "badanie urazowe-1_emg.slk" "badanie urazowe_Myo.slk"
'''

def parse_sylk(path: Path) -> pd.DataFrame:
    cells: Dict[Tuple[int, int], Any] = {}
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

    data = []
    for r in range(1, max_row + 1):
        row_vals = []
        for c in range(1, max_col + 1):
            row_vals.append(cells.get((r, c), None))
        data.append(row_vals)

    return pd.DataFrame(data)


def load_raw_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".slk":
        return parse_sylk(path)

    last_err = None
    for sep in ["\t", ";", ","]:
        try:
            return pd.read_csv(path, header=None, sep=sep, dtype=str, encoding="utf-8")
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Nie udało się wczytać pliku: {path}\n{last_err}")


def find_header_row(raw: pd.DataFrame) -> int:
    for i in range(len(raw)):
        first = raw.iloc[i, 0]
        if pd.notna(first) and str(first).strip() == "Time,s":
            return i
    raise ValueError("Nie znaleziono wiersza nagłówka 'Time,s'.")


def clean_cell(x: Any) -> Any:
    if pd.isna(x):
        return None
    if isinstance(x, str):
        x = x.strip()
        return x if x != "" else None
    return x


def parse_metadata(raw: pd.DataFrame, header_row: int) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for i in range(header_row):
        row = [clean_cell(v) for v in raw.iloc[i].tolist()]
        if not row or row[0] is None:
            continue
        key = str(row[0])
        value = None
        for item in row[1:]:
            if item is not None:
                value = item
                break
        meta[key] = value
    return meta


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


def build_data_frame(raw: pd.DataFrame, header_row: int) -> pd.DataFrame:
    headers = [clean_cell(v) for v in raw.iloc[header_row].tolist()]
    headers = [h for h in headers if h is not None]

    data = raw.iloc[header_row + 1:].copy()
    data = data.iloc[:, :len(headers)]
    data.columns = make_unique_headers(headers)
    data = data.dropna(how="all").reset_index(drop=True)
    return data


def to_float_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": None, "None": None, "nan": None})
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def summarize_file(path: Path) -> Dict[str, Any]:
    raw = load_raw_table(path)
    header_row = find_header_row(raw)
    metadata = parse_metadata(raw, header_row)
    df = build_data_frame(raw, header_row)

    summary: Dict[str, Any] = {
        "file": str(path),
        "metadata": metadata,
        "header_row_index_0based": header_row,
        "n_rows_raw": int(raw.shape[0]),
        "n_cols_raw": int(raw.shape[1]),
        "n_records": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
    }

    if "Time,s" in df.columns:
        t = to_float_series(df["Time,s"])
        valid_t = t.dropna()
        summary["time"] = {
            "n_valid": int(valid_t.shape[0]),
            "start_s": float(valid_t.iloc[0]) if len(valid_t) else None,
            "end_s": float(valid_t.iloc[-1]) if len(valid_t) else None,
            "duration_s": float(valid_t.iloc[-1] - valid_t.iloc[0]) if len(valid_t) >= 2 else None,
        }
        dt = valid_t.diff().dropna()
        positive_dt = dt[dt > 0]
        summary["time"]["median_dt_s"] = float(positive_dt.median()) if len(positive_dt) else None
        summary["time"]["estimated_frequency_hz"] = (
            float(1.0 / positive_dt.median()) if len(positive_dt) and positive_dt.median() != 0 else None
        )
        summary["time"]["n_missing_time"] = int(t.isna().sum())
        summary["time"]["n_nonmonotonic_steps"] = int((dt <= 0).sum()) if len(dt) else 0

    activities_info: Dict[str, Any] = {}
    if "Activities" in df.columns:
        a = df["Activities"]
        activities_info["non_empty_count"] = int(a.notna().sum())
        activities_info["unique_values"] = sorted([str(v) for v in pd.Series(a.dropna().unique()).tolist()])

    if "Activity Names" in df.columns:
        an = df["Activity Names"]
        an_clean = an.dropna()
        activities_info["activity_name_non_empty_count"] = int(an_clean.shape[0])
        activities_info["activity_names_unique"] = sorted([str(v) for v in pd.Series(an_clean.unique()).tolist()])

    if activities_info:
        summary["activities"] = activities_info

    missing = {}
    for i, col in enumerate(df.columns):
        missing[str(col)] = int(df.iloc[:, i].isna().sum())
    summary["missing_per_column"] = missing

    cols = [str(c) for c in df.columns]
    summary["column_groups"] = {
        "uv": [c for c in cols if "uV" in c],
        "deg": [c for c in cols if "deg" in c],
        "rot": [c for c in cols if "Rot " in c or c.endswith("Rot X,") or c.endswith("Rot Y,") or c.endswith("Rot Z,")],
    }

    return summary


def compare_summaries(emg: Dict[str, Any], myo: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    emg_date = emg.get("metadata", {}).get("Date")
    myo_date = myo.get("metadata", {}).get("Date")
    out["same_date"] = emg_date == myo_date
    out["emg_date"] = emg_date
    out["myo_date"] = myo_date

    emg_dur = emg.get("time", {}).get("duration_s")
    myo_dur = myo.get("time", {}).get("duration_s")
    out["duration_difference_s"] = (
        float(emg_dur - myo_dur) if emg_dur is not None and myo_dur is not None else None
    )
    return out


def print_summary(name: str, s: Dict[str, Any]) -> None:
    print(f"\n=== {name} ===")
    print(f"Plik: {s['file']}")
    print("Metadata:")
    for k, v in s.get("metadata", {}).items():
        print(f"  - {k}: {v}")
    print(f"Liczba rekordów: {s.get('n_records')}")
    print(f"Liczba kolumn: {s.get('n_columns')}")

    time = s.get("time", {})
    if time:
        print("Czas:")
        print(f"  - start [s]: {time.get('start_s')}")
        print(f"  - koniec [s]: {time.get('end_s')}")
        print(f"  - duration [s]: {time.get('duration_s')}")
        print(f"  - mediana dt [s]: {time.get('median_dt_s')}")
        print(f"  - est. freq [Hz]: {time.get('estimated_frequency_hz')}")
        print(f"  - braków w Time,s: {time.get('n_missing_time')}")
        print(f"  - kroków niemonotonicznych: {time.get('n_nonmonotonic_steps')}")

    acts = s.get("activities", {})
    if acts:
        print("Aktywności:")
        for k, v in acts.items():
            print(f"  - {k}: {v}")

    groups = s.get("column_groups", {})
    if groups:
        print("Grupy kolumn:")
        print(f"  - uV: {len(groups.get('uv', []))}")
        print(f"  - deg: {len(groups.get('deg', []))}")
        print(f"  - rot: {len(groups.get('rot', []))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audyt plików EMG i MyoMotion.")
    parser.add_argument("emg", type=Path, help="Ścieżka do pliku EMG")
    parser.add_argument("myo", type=Path, help="Ścieżka do pliku MyoMotion")
    parser.add_argument("--json", type=Path, default=None, help="Opcjonalna ścieżka do raportu JSON")
    args = parser.parse_args()

    emg_summary = summarize_file(args.emg)
    myo_summary = summarize_file(args.myo)
    comparison = compare_summaries(emg_summary, myo_summary)

    print_summary("EMG", emg_summary)
    print_summary("MYOMOTION", myo_summary)

    print("\n=== PORÓWNANIE ===")
    for k, v in comparison.items():
        print(f"- {k}: {v}")

    if args.json:
        payload = {
            "emg": emg_summary,
            "myomotion": myo_summary,
            "comparison": comparison,
        }
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nZapisano raport JSON do: {args.json}")


if __name__ == "__main__":
    main()