# =========================================
# IMPORTY
# =========================================

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr

# =========================================
# CONFIG
# =========================================

FILE_PATH = "Exept/dane.xlsx"
N_CLUSTERS = 4
RANDOM_STATE = 42

# =========================================
# UTILSY
# =========================================

def normalize_text(x):
    if isinstance(x, str):
        return x.strip().lower()
    return x


def find_col(df, keyword_list):
    for col in df.columns:
        for k in keyword_list:
            if k in col:
                return col
    raise ValueError(f"Nie znaleziono kolumny dla: {keyword_list}")


# =========================================
# MAPOWANIA
# =========================================

binary_map = {"tak": 1, "nie": 0, "może": 0.5}

scale_map = {
    "mały": 1,
    "średni": 2,
    "duży": 3,
    "bardzo duży": 4
}

condition_map = {
    "zła": 1,
    "nie mam zdania": 2,
    "dobra": 3,
    "bardzo dobra": 4
}

ciezkosc_map = {
    "bardzo lekka": 1,
    "lekka": 1,
    "średnio ciężka": 2,
    "ciężka": 3,
    "bardzo ciężka": 4
}

# =========================================
# WCZYTANIE DANYCH
# =========================================

def load_data(path):
    df = pd.read_excel(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.apply(lambda col: col.map(normalize_text))
    return df


# =========================================
# PREPROCESSING
# =========================================

def preprocess(df):

    df = df.copy()

    # mapowanie kolumn
    df["stres"] = df[find_col(df, ["stres"])]
    df["zmeczenie"] = df[find_col(df, ["zmęczen", "zmeczen"])]
    df["kondycja"] = df[find_col(df, ["kondycj"])]
    df["ciezkosc_pracy"] = df[find_col(df, ["ciężko", "ciezko"])]
    df["halas"] = df[find_col(df, ["hałas", "halas"])]
    df["drgania"] = df[find_col(df, ["drgan"])]

    # mapowanie wartości (kolumna po kolumnie!)
    df["stres"] = df["stres"].replace(scale_map)
    df["zmeczenie"] = df["zmeczenie"].replace(scale_map)
    df["kondycja"] = df["kondycja"].replace(condition_map)
    df["ciezkosc_pracy"] = df["ciezkosc_pracy"].replace(ciezkosc_map)
    df["halas"] = df["halas"].replace(scale_map)
    df["drgania"] = df["drgania"].replace(scale_map)

    df["halas"] = df["halas"].replace(binary_map)
    df["drgania"] = df["drgania"].replace(binary_map)

    return df


# =========================================
# BÓL
# =========================================

def process_pain(df):

    pain_col = [c for c in df.columns if "dolegliwości" in c][0]

    def parse_pain(x):
        if not isinstance(x, str):
            return []
        return re.split("[,;]", x.lower())

    def classify(parts):
        plecy = 0
        konczyny = 0

        for p in parts:
            if any(k in p for k in ["plec", "szyi"]):
                plecy += 1
            elif any(k in p for k in ["rąk", "nóg", "bark"]):
                konczyny += 1

        return plecy, konczyny

    totals = []

    for val in df[pain_col]:
        parts = parse_pain(val)
        totals.append(len(parts))

    df["bol_total"] = totals
    return df


# =========================================
# FUZZY MODEL
# =========================================

def sigmoid(x, c=0.5, k=10):
    return 1 / (1 + np.exp(-k * (x - c)))


def normalize_quantile(X):
    X = X.copy()
    for col in X.columns:
        q1 = X[col].quantile(0.05)
        q9 = X[col].quantile(0.95)
        X[col] = ((X[col] - q1) / (q9 - q1)).clip(0, 1)
    return X


def compute_W(row):

    stres = sigmoid(row["stres"])
    zmeczenie = sigmoid(row["zmeczenie"])
    kondycja = sigmoid(row["kondycja"])

    mental = 0.4 * stres + 0.4 * zmeczenie + 0.2 * kondycja
    mental += 0.4 * stres * zmeczenie

    physical = (
        0.6 * sigmoid(row["ciezkosc_pracy"]) +
        0.3 * sigmoid(row["halas"])
    )

    # 🔥 interakcja między blokami
    interaction = 0.2 * zmeczenie * sigmoid(row["ciezkosc_pracy"])

    W = np.power(0.7 * mental + 0.3 * physical + interaction, 1.0)

    return W


# =========================================
# MODEL GŁÓWNY
# =========================================

def build_model(df):

    features = ["stres", "zmeczenie", "kondycja", "ciezkosc_pracy", "halas", "drgania"]

    X = df[features].apply(pd.to_numeric, errors="coerce").dropna()

    print("\n=== DIAGNOZA X ===")
    print("shape:", X.shape)

    print("\nNaN per kolumna:")
    print(df[features].isna().sum())

    print("\nUnikalne wartości:")
    for col in features:
        print(col, df[col].unique())

    if X.shape[0] == 0:
        raise ValueError("Brak danych po preprocessingu — sprawdź mapowanie wartości")

    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X)

    print("\nExplained variance:")
    print(pca.explained_variance_ratio_)

    # clustering
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_pca[:, :3])

    print("\nSilhouette score:", silhouette_score(X_pca[:, :3], clusters))

    df_struct = df.loc[X.index].copy()
    df_struct["cluster"] = clusters

    # fuzzy
    X_fuzzy = X.copy()

    # odwrócenie kondycji
    X_fuzzy["kondycja"] = 5 - X_fuzzy["kondycja"]

    X_fuzzy = normalize_quantile(X_fuzzy)

    df_fuzzy = df.loc[X_fuzzy.index].copy()

    df_fuzzy["W_raw"] = X_fuzzy.apply(compute_W, axis=1)

    W = df_fuzzy["W_raw"]
    df_fuzzy["W"] = (W - W.min()) / (W.max() - W.min() + 1e-6)

    df_fuzzy["risk_level"] = pd.qcut(
        df_fuzzy["W"],
        q=4,
        labels=["low", "moderate", "high", "critical"],
        duplicates="drop"
    )

    print("\n=== ROZKŁAD RISK LEVEL ===")
    print(df_fuzzy["risk_level"].value_counts())

    return df_struct, df_fuzzy, X_fuzzy
# =========================================
# MODEL GŁÓWNY
# =========================================

def build_model(df):

    from scipy.stats import spearmanr

    features = ["stres", "zmeczenie", "kondycja", "ciezkosc_pracy", "halas"]

    # =========================================
    # DANE
    # =========================================

    X = df[features].apply(pd.to_numeric, errors="coerce").dropna()

    print("\n=== DIAGNOZA X ===")
    print("shape:", X.shape)

    print("\nNaN per kolumna:")
    print(df[features].isna().sum())

    print("\nUnikalne wartości:")
    for col in features:
        print(col, df[col].unique())

    if X.shape[0] == 0:
        raise ValueError("Brak danych po preprocessingu — sprawdź mapowanie wartości")

    # =========================================
    # PCA
    # =========================================

    pca = PCA()
    X_pca = pca.fit_transform(X)

    print("\nExplained variance:")
    print(pca.explained_variance_ratio_)

    # zapis PCA do df
    df_pca = df.loc[X.index].copy()
    df_pca["W_pca"] = X_pca[:, 0]

    # normalizacja PCA (0–1)
    Wp = df_pca["W_pca"]
    df_pca["W_pca_norm"] = (Wp - Wp.min()) / (Wp.max() - Wp.min() + 1e-6)

    # =========================================
    # KLASTRY
    # =========================================

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_pca[:, :3])

    print("\nSilhouette score:", silhouette_score(X_pca[:, :3], clusters))

    df_struct = df.loc[X.index].copy()
    df_struct["cluster"] = clusters

    # =========================================
    # FUZZY
    # =========================================

    X_fuzzy = X.copy()

    # odwrócenie kondycji
    X_fuzzy["kondycja"] = 5 - X_fuzzy["kondycja"]

    # normalizacja
    X_fuzzy = normalize_quantile(X_fuzzy)

    df_fuzzy = df.loc[X_fuzzy.index].copy()

    df_fuzzy["W_raw"] = X_fuzzy.apply(compute_W, axis=1)

    W = df_fuzzy["W_raw"]
    df_fuzzy["W"] = (W - W.min()) / (W.max() - W.min() + 1e-6)

    # =========================================
    # POZIOMY RYZYKA
    # =========================================

    df_fuzzy["risk_level"] = pd.qcut(
        df_fuzzy["W"],
        q=4,
        labels=["low", "moderate", "high", "critical"],
        duplicates="drop"
    )

    print("\n=== ROZKŁAD RISK LEVEL ===")
    print(df_fuzzy["risk_level"].value_counts())

    # =========================================
    # PORÓWNANIE PCA vs FUZZY
    # =========================================

    df_compare = df_fuzzy[["W"]].join(
        df_pca[["W_pca_norm"]],
        how="inner"
    )

    print("\n=== CHECK długości ===")
    print(df_compare.shape)

    rho, p = spearmanr(df_compare["W"], df_compare["W_pca_norm"])

    print("\n=== PCA vs FUZZY ===")
    print(f"rho={rho:.3f}, p={p:.5f}")

    # =========================================
    # RETURN
    # =========================================

    return df_struct, df_fuzzy, X_fuzzy, df_pca

# =========================================
# WYKRESY
# =========================================

def plot_results(df_fuzzy, X_fuzzy, df_struct):

    # histogram W
    plt.figure()
    df_fuzzy["W"].hist()
    plt.title("Rozkład W")
    plt.show()

    # korelacje
    corrs = []

    for col in X_fuzzy.columns:
        rho, _ = spearmanr(df_fuzzy["W"], X_fuzzy[col])
        corrs.append((col, rho))

    names = [c[0] for c in corrs]
    values = [c[1] for c in corrs]

    plt.figure()
    plt.bar(names, values)
    plt.title("Spearman correlations")
    plt.xticks(rotation=45)
    plt.show()

    # klastry vs W
    merged = df_struct.join(df_fuzzy["W"], how="inner")

    plt.figure()
    merged.boxplot(column="W", by="cluster")
    plt.title("W vs cluster")
    plt.suptitle("")
    plt.show()

    # ból vs W
    if "bol_total" in df_fuzzy.columns:
        plt.figure()
        plt.scatter(df_fuzzy["bol_total"], df_fuzzy["W"])
        plt.title("Ból vs W")
        plt.xlabel("bol_total")
        plt.ylabel("W")
        plt.show()


# =========================================
# WALIDACJA
# =========================================

def validate(df_fuzzy, X_fuzzy):

    print("\n=== KORELACJE ===")

    for col in X_fuzzy.columns:
        rho, p = spearmanr(df_fuzzy["W"], X_fuzzy[col])
        print(f"{col}: rho={rho:.3f}, p={p:.5f}")

    if "bol_total" in df_fuzzy.columns:
        print("\n=== W vs BÓL ===")
        print(df_fuzzy[["W", "bol_total"]].corr())


# =========================================
# MAIN
# =========================================

def main():

    df = load_data(FILE_PATH)

    df = preprocess(df)
    df = process_pain(df)

    df_struct, df_fuzzy, X_fuzzy, df_pca = build_model(df)

    # =========================
    # PORÓWNANIE PCA vs FUZZY
    # =========================

    df_compare = df_fuzzy[["W", "bol_total"]].join(
        df_pca[["W_pca_norm"]],
        how="inner"
    )

    # różnice
    df_compare["diff"] = df_compare["W"] - df_compare["W_pca_norm"]

    print("\n=== NAJWIĘKSZE RÓŻNICE PCA vs FUZZY ===")
    print(df_compare.sort_values("diff", key=abs, ascending=False).head(10))


    # =========================
    # PCA vs FUZZY vs BÓL
    # =========================

    rho_pca, _ = spearmanr(df_compare["W_pca_norm"], df_compare["bol_total"])
    rho_fuzzy, _ = spearmanr(df_compare["W"], df_compare["bol_total"])

    print("\n=== PCA vs FUZZY vs BÓL ===")
    print(f"PCA:   rho={rho_pca:.3f}")
    print(f"Fuzzy: rho={rho_fuzzy:.3f}")

    validate(df_fuzzy, X_fuzzy)
    plot_results(df_fuzzy, X_fuzzy, df_struct)


if __name__ == "__main__":
    main()