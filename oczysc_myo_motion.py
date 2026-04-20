#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
oczysc_myo_motion
Oczyszczanie pliku MyoMotion z eksportu SYLK (.slk).
"""
#Uruchomienie:
#python .\oczysc_myo_motion.py "badanie urazowe_Myo.slk"

#Opcjonalnie można podać własną nazwę pliku wyjściowego:
#python .\oczysc_myo_motion.py "badanie urazowe_Myo.slk" --out "myo_clean.csv"


from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


CELL_RE = re.compile(r"([A-Z]+)([0-9]+)")


def col_letters_to_index(col_letters: str) -> int:
    """Zamienia np. A -> 1, B -> 2, AA -> 27."""
    result = 0
    for ch in col_letters:
        result = result * 26 + (ord(ch.upper()) - ord("A") + 1)
    return result


def parse_sylk(path: Path) -> pd.DataFrame:
    """
    Prosty parser SYLK:
    - czyta rekordy C;...
    - buduje siatkę [wiersz, kolumna]
    - zwraca DataFrame bez zależności od xlrd/openpyxl
    """
    cells: Dict[Tuple[int, int], object] = {}
    current_row: Optional[int] = None
    max_row = 0
    max_col = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line.startswith("C;"):
                continue

            parts = line.split(";")
            row = current_row
            col = None
            value = None

            for part in parts[1:]:
                if part.startswith("Y"):
                    try:
                        row = int(part[1:])
                        current_row = row
                    except ValueError:
                        pass
                elif part.startswith("X"):
                    try:
                        col = int(part[1:])
                    except ValueError:
                        pass
                elif part.startswith("K"):
                    raw_value = part[1:]
                    if raw_value.startswith('"') and raw_value.endswith('"'):
                        value = raw_value[1:-1].replace('""', '"')
                    else:
                        value = raw_value

            if row is None or col is None:
                continue

            cells[(row, col)] = value
            max_row = max(max_row, row)
            max_col = max(max_col, col)

    if max_row == 0 or max_col == 0:
        raise RuntimeError(f"Nie udało się odczytać danych z pliku SYLK: {path}")

    data: List[List[object]] = []
    for r in range(1, max_row + 1):
        row_values = []
        for c in range(1, max_col + 1):
            row_values.append(cells.get((r, c), None))
        data.append(row_values)

    return pd.DataFrame(data)


def try_parse_number(x):
    """Próbuje zamienić tekst na liczbę; zostawia tekst, jeśli się nie da."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return x
    s = str(x).strip()
    if s == "":
        return None

    # zamiana przecinka dziesiętnego na kropkę, jeśli występuje
    s2 = s.replace(",", ".")

    try:
        val = float(s2)
        if math.isfinite(val):
            return val
        return None
    except ValueError:
        return x


def build_dataframe(raw: pd.DataFrame) -> Tuple[Dict[str, object], pd.DataFrame]:
    """
    Zakłada układ:
    wiersz 1: Name / wartość
    wiersz 2: Frequency / wartość
    wiersz 3: Date / wartość
    wiersz 4: nagłówki
    wiersz 5+: dane
    """
    metadata = {
        "Name": raw.iat[0, 1] if raw.shape[0] > 0 and raw.shape[1] > 1 else None,
        "Frequency": raw.iat[1, 1] if raw.shape[0] > 1 and raw.shape[1] > 1 else None,
        "Date": raw.iat[2, 1] if raw.shape[0] > 2 and raw.shape[1] > 1 else None,
    }

    header_row_idx = 3
    if raw.shape[0] <= header_row_idx:
        raise RuntimeError("Plik nie zawiera wiersza nagłówków.")

    headers = []
    for i, val in enumerate(raw.iloc[header_row_idx].tolist(), start=1):
        if val is None or str(val).strip() == "":
            headers.append(f"col_{i}")
        else:
            headers.append(str(val).strip())

    data = raw.iloc[header_row_idx + 1 :].copy()
    data.columns = headers
    data = data.reset_index(drop=True)

    return metadata, data


def clean_myo_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # usuwamy całkowicie puste wiersze
    df = df.dropna(how="all").reset_index(drop=True)

    # wstępna konwersja wartości
    for col in df.columns:
        df[col] = df[col].map(try_parse_number)

    # Time,s jest kolumną kluczową
    if "Time,s" not in df.columns:
        raise RuntimeError("Brak kolumny 'Time,s'.")

    df["Time,s"] = pd.to_numeric(df["Time,s"], errors="coerce")

    # usuwamy rekordy bez czasu
    df = df[df["Time,s"].notna()].copy()

    # usuwamy rekordy techniczne typu min / max w Activity Names
    if "Activity Names" in df.columns:
        mask_bad = (
            df["Activity Names"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"min", "max", "minimum", "maximum"})
        )
        df = df[~mask_bad].copy()

    # usuwamy rekordy techniczne w Activities
    if "Activities" in df.columns:
        mask_bad2 = (
            df["Activities"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"min", "max", "minimum", "maximum"})
        )
        df = df[~mask_bad2].copy()

    # sortowanie po czasie i usunięcie duplikatów czasu
    df = df.sort_values("Time,s").drop_duplicates(subset=["Time,s"], keep="first")
    df = df.reset_index(drop=True)

    # konwersja wszystkich możliwych kolumn liczbowych poza opisowymi
    text_like = {"Activity Names"}
    for col in df.columns:
        if col not in text_like:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                df[col] = converted

    return df


def summarize(df: pd.DataFrame, metadata: Dict[str, object]) -> str:
    lines: List[str] = []

    lines.append("=== PODSUMOWANIE MYOMOTION ===")
    lines.append(f"Name: {metadata.get('Name')}")
    lines.append(f"Frequency: {metadata.get('Frequency')}")
    lines.append(f"Date: {metadata.get('Date')}")
    lines.append(f"Liczba rekordów po czyszczeniu: {len(df)}")
    lines.append(f"Liczba kolumn: {len(df.columns)}")

    if "Time,s" in df.columns and len(df) > 1:
        t = pd.to_numeric(df["Time,s"], errors="coerce").dropna()
        if len(t) > 1:
            dt = t.diff().dropna()
            lines.append(f"Start [s]: {t.iloc[0]}")
            lines.append(f"Koniec [s]: {t.iloc[-1]}")
            lines.append(f"Czas trwania [s]: {t.iloc[-1] - t.iloc[0]}")
            lines.append(f"Mediana dt [s]: {dt.median()}")
            if dt.median() and dt.median() > 0:
                lines.append(f"Szac. częstotliwość [Hz]: {1.0 / dt.median()}")

    deg_cols = [c for c in df.columns if "deg" in str(c).lower()]
    rot_cols = [
        c for c in df.columns
        if str(c).endswith("Rot X,")
        or str(c).endswith("Rot Y,")
        or str(c).endswith("Rot Z,")
    ]

    lines.append(f"Kolumny 'deg': {len(deg_cols)}")
    lines.append(f"Kolumny 'rot': {len(rot_cols)}")

    if "Activity Names" in df.columns:
        uniq = sorted(
            {
                str(x).strip()
                for x in df["Activity Names"].dropna().tolist()
                if str(x).strip() != ""
            }
        )
        lines.append(f"Unikalne Activity Names: {uniq[:20]}")

    return "\n".join(lines)


def default_output_name(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + "_clean.csv")


def main():
    parser = argparse.ArgumentParser(description="Oczyszczanie pliku MyoMotion (.slk)")
    parser.add_argument("input", help="Ścieżka do pliku .slk z MyoMotion")
    parser.add_argument("--out", help="Ścieżka do pliku wynikowego .csv", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {input_path}")

    output_path = Path(args.out) if args.out else default_output_name(input_path)

    raw = parse_sylk(input_path)
    metadata, df = build_dataframe(raw)
    clean_df = clean_myo_dataframe(df)

    clean_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(summarize(clean_df, metadata))
    print()
    print(f"Zapisano plik: {output_path}")


if __name__ == "__main__":
    main()