# step 2

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, kruskal


# =========================================
# 1. LOAD DATA
# =========================================

df = pd.read_excel("Exept/dane.xlsx")

# =========================================
# 2. BASIC CLEANING
# =========================================

def normalize_text(x):
    if isinstance(x, str):
        return x.strip().lower()
    return x

df = df.apply(lambda col: col.map(normalize_text))

# =========================================
# 3. MAP ORDINAL RESPONSES
# =========================================

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
    "lekka": 1,
    "średnio ciężka": 2,
    "ciężka": 3,
    "bardzo ciężka": 4
}

df = df.replace(scale_map)
df = df.replace(condition_map)

# =========================================
# 4. CLEAN COLUMN NAMES
# =========================================

def clean_column_name(col):
    return col.lower().strip().replace("\t", "").replace("  ", " ")

df.columns = [clean_column_name(col) for col in df.columns]

# =========================================
# 5. RENAME VARIABLES
# =========================================

column_map = {
    "8. jak ocenia pani/pan poziom odczuwalnego stresu?": "stres",
    "9. jak ocenia pani/pan swoją kondycję psychofizyczną?": "kondycja",
    "10. jaki poziom zmęczenia odczuwa pani/pan pod koniec pracy ?": "zmeczenie",
    "37.jak oceniasz ciężkość wykonywanej pracy?": "ciezkosc_pracy",
    "43. czy hałas na stanowisku pracy jest uciążliwy?": "halas",
    "42. czy na stanowisku pracy są odczuwalne drgania?": "drgania"
}

df = df.rename(columns=column_map)

# =========================================
# 6. PREPARE FEATURE MATRIX
# =========================================

features = [
    "stres",
    "zmeczenie",
    "kondycja",
    "ciezkosc_pracy",
    "halas",
    "drgania"
]

X = df[features].copy()

# --- explicit remapping for ranking PCA ---
X["ciezkosc_pracy"] = X["ciezkosc_pracy"].replace({
    "bardzo lekka": 0,
    "lekka": 1,
    "średnio ciężka": 2,
    "ciężka": 3,
    "bardzo ciężka": 4
})

X["halas"] = X["halas"].replace({
    "tak": 1,
    "nie": 0
})

X["drgania"] = X["drgania"].replace({
    "tak": 1,
    "nie": 0
})

# numeric conversion
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

print("\nShape BEFORE dropna:", X.shape)
print("\nMissing per column:")
print(X.isna().sum())

X = X.dropna()

print("\nShape AFTER dropna:", X.shape)
print(X.head())

# =====================================================
# STEP 5 — DIRECTIONAL HARMONIZATION
# =====================================================

# Create ranking-analysis matrix
X_rank = X.copy()

# Reverse positively oriented variables so that:
# higher value = higher burden for all indicators
X_rank["kondycja"] = 5 - X_rank["kondycja"]


# =====================================================
# STEP 6 — BETWEEN-RESPONDENT STANDARDIZATION
# =====================================================

print("\n=== RANKING PCA DEBUG ===")
print("Shape before scaling:", X_rank.shape)

print("\nMissing values per column:")
print(X_rank.isna().sum())

print("\nUnique values:")
for col in X_rank.columns:
    print(f"\n{col}:")
    print(sorted(X_rank[col].dropna().unique()))

# Safety check
if X_rank.empty:
    raise ValueError("X_rank is empty after preprocessing.")

# Z-score standardization across respondents
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_rank)


# =====================================================
# STEP 7 — PCA-BASED BURDEN EXTRACTION
# =====================================================

pca = PCA(n_components=1)
W_raw = pca.fit_transform(X_scaled).flatten()

loadings = pd.Series(
    pca.components_[0],
    index=features
)

# Ensure positive interpretive direction:
# higher PC1 = higher burden
if loadings.mean() < 0:
    W_raw = -W_raw
    loadings = -loadings


# =====================================================
# STEP 8 — PERCENTILE TRANSFORMATION
# =====================================================

W_global = pd.Series(W_raw).rank(pct=True)

df_rank = df.loc[X.index].copy()
df_rank["W_raw"] = W_raw
df_rank["W_global"] = W_global.values


# =====================================================
# STEP 9 — SUMMARY OUTPUT
# =====================================================

print("\n=== RANKING PCA RESULTS ===")

print("\nExplained variance (PC1):")
print(round(pca.explained_variance_ratio_[0], 3))

print("\nLoadings:")
print(loadings.round(3))

print("\nRaw score summary:")
print(pd.Series(W_raw).describe())

print("\nPercentile burden index summary:")
print(df_rank["W_global"].describe())


# =====================================================
# =====================================================
# STEP 10 — FIGURE 1: PCA LOADINGS
# =====================================================

import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = True

feature_labels_en = {
    "stres": "Stress",
    "zmeczenie": "Fatigue",
    "kondycja": "Condition",
    "ciezkosc_pracy": "Workload",
    "halas": "Noise",
    "drgania": "Vibration"
}

loadings_plot = loadings.sort_values().copy()
loadings_plot.index = [
    feature_labels_en.get(idx, idx)
    for idx in loadings_plot.index
]

plt.figure(figsize=(7.0, 4.2))

loadings_plot.plot(
    kind="barh"
)

plt.title("Ranking PCA Loadings (PC1)", fontsize=12)
plt.xlabel("Loading", fontsize=10)
plt.ylabel("Variable", fontsize=10)

plt.xticks(fontsize=9)
plt.yticks(fontsize=9)

plt.tight_layout()

plt.savefig(
    "figure_ranking_loadings.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================================
# STEP 11 — FIGURE 2: RAW PC1 DISTRIBUTION
# =====================================================

plt.figure(figsize=(7, 4))

sns.histplot(
    W_raw,
    bins=20,
    kde=True
)

plt.title("Distribution of Raw PCA Burden Scores")
plt.xlabel("PC1 Burden Score")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("ranking_distribution_raw.png", dpi=300)
plt.show()


# =====================================================
# STEP 12 — OPTIONAL FIGURE 3: PERCENTILE DISTRIBUTION
# =====================================================

plt.figure(figsize=(7, 4))

sns.histplot(
    df_rank["W_global"],
    bins=20
)

plt.title("Distribution of Percentile Burden Index")
plt.xlabel("W_global")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("ranking_distribution_percentile.png", dpi=300)
plt.show()


# =====================================================
# STEP 12.5 — PAIN VARIABLE CODING
# =====================================================

pain_col = [col for col in df.columns if "dolegliwości" in col][0]
pain_level_col = "jakie:"

pain_level_map = {
    "małe": 1,
    "średnie": 2,
    "duże": 3,
    "bardzo duże": 4
}


def parse_pain(x):
    if not isinstance(x, str):
        return []

    if "," in x:
        return [p.strip() for p in x.split(",") if p.strip()]
    elif ";" in x:
        return [p.strip() for p in x.split(";") if p.strip()]
    else:
        return [x.strip()]


pain_counts = []

for val in df_rank[pain_col]:
    parts = parse_pain(val)
    pain_counts.append(len(parts))

df_rank["pain_count"] = pain_counts
df_rank["pain_level"] = df_rank[pain_level_col].map(pain_level_map)
df_rank["pain_weighted"] = (
    df_rank["pain_count"] *
    df_rank["pain_level"]
)

print("\n=== PAIN VARIABLES ADDED ===")
print(df_rank[["pain_count", "pain_level", "pain_weighted"]].describe())

# =====================================================
# STEP 13 — EXPORT RESULTS
# =====================================================

df_rank.to_excel(
    "ranking_pca_results.xlsx",
    index=False
)

loadings.to_excel(
    "ranking_loadings.xlsx",
    header=["Loading"]
)

print("\nExported:")
print("- ranking_pca_results.xlsx")
print("- ranking_loadings.xlsx")
print("- ranking_loadings.png")
print("- ranking_distribution_raw.png")
print("- ranking_distribution_percentile.png")