# step 3

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, kruskal


# =====================================================
# 1. LOAD RESULTS
# =====================================================

df_struct = pd.read_excel("structural_pca_results.xlsx")
df_rank = pd.read_excel("ranking_pca_results.xlsx")

print("\n=== RANK DF COLUMNS ===")
print(df_rank.columns.tolist())

# =====================================================
# 2. MERGE STRUCTURAL + RANKING RESULTS
# =====================================================

merge_cols = [
    "W_global",
    "W_raw",
    "pain_count",
    "pain_level",
    "pain_weighted"
]

merge_cols = [
    col for col in merge_cols
    if col in df_rank.columns
]

df = df_struct.join(
    df_rank[merge_cols],
    how="inner"
)

print("\nMerged shape:", df.shape)


# =====================================================
# 3. IDENTIFY PAIN VARIABLES
# =====================================================

pain_candidates = [
    col for col in df.columns
    if "bol" in col.lower() or "pain" in col.lower()
]

print("\nDetected pain-related variables:")
print(pain_candidates)


# =====================================================
# 4. SPEARMAN CORRELATIONS WITH PAIN INDICATORS
# =====================================================

print("\n=== PAIN ASSOCIATION CHECKS ===")

pain_results = []

for col in pain_candidates:

    valid = df[[col, "W_global"]].dropna()

    if len(valid) < 10:
        continue

    rho, p = spearmanr(valid["W_global"], valid[col])

    pain_results.append({
        "Variable": col,
        "Spearman_rho": rho,
        "p_value": p
    })

pain_results_df = pd.DataFrame(pain_results)

print(pain_results_df.round(4))


# =====================================================
# 5. CLUSTER CONSISTENCY CHECK
# =====================================================

print("\n=== CLUSTER CONSISTENCY CHECK ===")

if "cluster" not in df.columns:
    raise ValueError("Cluster column not found in structural PCA results.")

groups = [
    df[df["cluster"] == c]["W_global"].dropna()
    for c in sorted(df["cluster"].unique())
]

H, p_kw = kruskal(*groups)

print(f"Kruskal-Wallis H = {H:.3f}")
print(f"p-value = {p_kw:.6f}")

# =====================================================
# 5B. POST-HOC PAIRWISE COMPARISONS (MANN-WHITNEY + SIDAK)
# =====================================================

from scipy.stats import mannwhitneyu
from itertools import combinations
import numpy as np

print("\n=== POST-HOC PAIRWISE COMPARISONS ===")

pairwise_results = []

cluster_ids = sorted(df["cluster"].unique())
n_comparisons = len(list(combinations(cluster_ids, 2)))

for c1, c2 in combinations(cluster_ids, 2):

    g1 = df[df["cluster"] == c1]["W_global"].dropna()
    g2 = df[df["cluster"] == c2]["W_global"].dropna()

    U, p_raw = mannwhitneyu(g1, g2, alternative="two-sided")

    # Sidak correction
    p_sidak = 1 - (1 - p_raw) ** n_comparisons

    pairwise_results.append({
        "Comparison": f"{c1} vs {c2}",
        "U_statistic": U,
        "p_raw": p_raw,
        "p_sidak": min(p_sidak, 1.0)
    })

pairwise_df = pd.DataFrame(pairwise_results)

print(pairwise_df.round(4))

# =====================================================
# 6. CLUSTER MEANS
# =====================================================

cluster_summary = df.groupby("cluster")["W_global"].agg([
    "mean",
    "median",
    "std",
    "count"
])

print("\nCluster burden summary:")
print(cluster_summary.round(3))


# =====================================================
# 7. FIGURE — W_global ACROSS STRUCTURAL CLUSTERS
# =====================================================

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="cluster",
    y="W_global"
)

sns.pointplot(
    data=df,
    x="cluster",
    y="W_global",
    estimator="mean",
    errorbar=None,
    color="black",
    markers="D"
)

plt.title("Global Burden Index Across Structural Burden Profiles")
plt.xlabel("Structural Burden Profile")
plt.ylabel("W_global")

plt.tight_layout()
plt.savefig("cluster_vs_wglobal.png", dpi=300)
plt.show()


# =====================================================
# 8. EXPORT RESULTS
# =====================================================

pain_results_df.to_excel(
    "pain_association_results.xlsx",
    index=False
)

cluster_summary.to_excel(
    "cluster_consistency_results.xlsx"
)

pairwise_df.to_excel(
    "pairwise_cluster_comparisons.xlsx",
    index=False
)

print("\nExported:")
print("- pain_association_results.xlsx")
print("- cluster_consistency_results.xlsx")
print("- pairwise_cluster_comparisons.xlsx")
print("- cluster_vs_wglobal.png")