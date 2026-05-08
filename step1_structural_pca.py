# step 1

# =========================================
# STRUCTURAL PCA ANALYSIS
# Supporting branch for latent burden structure
# Corresponds to paper Sections 3.2 / 4.1 / 4.2
# =========================================

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from factor_analyzer.factor_analyzer import (
    calculate_kmo,
    calculate_bartlett_sphericity
)

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

# --- explicit remapping for structural PCA ---
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

print("\n=== RAW X BEFORE WITHIN NORMALIZATION ===")
print(X.head())

print("\nShape before dropna:", X.shape)

print("\nMissing values per column:")
print(X.isna().sum())

print("\nUnique values per column:")
for col in X.columns:
    print(f"\n{col}:")
    print(sorted(X[col].dropna().unique()))

print("\nRow-wise STD before filtering:")
row_stds = X.std(axis=1)
print(row_stds.describe())

print("\nNumber of zero-STD rows:")
print((row_stds == 0).sum())

# =========================================
# 7. WITHIN-RESPONDENT NORMALIZATION
# =========================================

row_means = X.mean(axis=1)
row_stds = X.std(axis=1)

# usuń obserwacje bez wewnętrznej zmienności
valid_rows = row_stds > 0

print("\nRemoved constant-profile respondents:", (~valid_rows).sum())

X_struct = X.loc[valid_rows].copy()

X_struct = X_struct.sub(
    X_struct.mean(axis=1),
    axis=0
)

X_struct = X_struct.div(
    X_struct.std(axis=1),
    axis=0
)

# =========================================
# 8. PCA SUITABILITY CHECKS
# =========================================

kmo_all, kmo_model = calculate_kmo(X_struct)
bartlett_chi2, bartlett_p = calculate_bartlett_sphericity(X_struct)

print("\n=== PCA Suitability ===")
print("KMO:", round(kmo_model, 3))
print("Bartlett chi2:", round(bartlett_chi2, 3))
print("Bartlett p:", bartlett_p)

# =========================================
# 9. STRUCTURAL PCA
# =========================================

pca = PCA()
X_pca = pca.fit_transform(X_struct)

explained = pca.explained_variance_ratio_

print("\n=== Explained Variance ===")
for i, var in enumerate(explained, 1):
    print(f"PC{i}: {var:.3f}")

# =========================================
# 10. LOADINGS MATRIX
# =========================================

loadings = pd.DataFrame(
    pca.components_.T,
    index=features,
    columns=[f"PC{i}" for i in range(1, len(features)+1)]
)

print("\n=== Loadings ===")
print(loadings.round(3))

# =========================================
# 11. SCREE PLOT
# =========================================

plt.figure(figsize=(6,4))
plt.plot(
    range(1, len(explained)+1),
    explained,
    marker='o'
)
plt.axvline(x=3, linestyle='--', color='red')
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance")
plt.title("Explained Variance of Structural PCA Components")
plt.tight_layout()
plt.savefig("figure_structural_scree.png", dpi=300)
plt.show()

# =========================================
# 12. HEATMAP OF RETAINED COMPONENTS
# =========================================

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

loadings_plot = loadings.iloc[:, :3].copy()
loadings_plot.index = [
    feature_labels_en.get(idx, idx)
    for idx in loadings_plot.index
]

annot_labels = loadings_plot.copy()

for col in annot_labels.columns:
    annot_labels[col] = annot_labels[col].map(
        lambda x: f"{x:.2f}".replace("-", "−")
    )

plt.figure(figsize=(6.2, 4.4))

sns.heatmap(
    loadings_plot,
    annot=annot_labels,
    fmt="",
    cmap="coolwarm",
    center=0,
    cbar_kws={"label": "Loading"}
)

plt.title("Structural PCA Loadings (PC1–PC3)", fontsize=12)
plt.xlabel("Principal Component", fontsize=10)
plt.ylabel("Variable", fontsize=10)

plt.xticks(fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.tight_layout()

plt.savefig(
    "figure_structural_loadings.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================================
# 13. CLUSTERING IN PCA SPACE
# =========================================

X_cluster = X_pca[:, :3]

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_cluster)

'''
# Reorder clusters from lowest to highest burden
cluster_means = pd.DataFrame({
    "cluster_raw": clusters_raw,
    "PC1": X_pca[:, 0]
}).groupby("cluster_raw")["PC1"].mean().sort_values()

cluster_order = {
    old_label: new_label
    for new_label, old_label in enumerate(cluster_means.index)
}
clusters = pd.Series(clusters_raw).map(cluster_order).values
'''


sil_score = silhouette_score(X_cluster, clusters)

print("\n=== Clustering ===")
print("Silhouette score:", round(sil_score, 3))

# =========================================
# 14. CLUSTER VISUALIZATION
# =========================================

plt.figure(figsize=(6,5))
sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=clusters,
    palette="Set2"
)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Structural Burden Profiles in PCA Space")
plt.tight_layout()
plt.savefig("figure_structural_clusters.png", dpi=300)
plt.show()

profile_labels = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Very High"
}

# =========================================
# 15. EXPORT RESULTS
# =========================================

df_results = df.loc[X_struct.index].copy()
df_results["cluster"] = clusters
df_results["cluster_label"] = pd.Series(clusters).map(profile_labels)

for i in range(3):
    df_results[f"PC{i+1}"] = X_pca[:, i]

df_results.to_excel("structural_pca_results.xlsx", index=False)

print("\nExported: structural_pca_results.xlsx")