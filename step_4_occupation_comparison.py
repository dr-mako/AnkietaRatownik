import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import kruskal, mannwhitneyu
from itertools import combinations


# =====================================================
# 1. LOAD RANKING RESULTS
# =====================================================

df = pd.read_excel("ranking_pca_results.xlsx")


# =====================================================
# 2. DETECT OCCUPATION COLUMN
# =====================================================

occupation_col = "4. na jakim stanowisku pracy pani/pan pracuje?"

if occupation_col not in df.columns:
    raise ValueError(f"Occupation column not found: {occupation_col}")


# =====================================================
# 3. BASIC SUMMARY
# =====================================================

print("\n=== OCCUPATION COUNTS ===")
print(df[occupation_col].value_counts())


# =====================================================
# 4. OCCUPATION BURDEN SUMMARY
# =====================================================

occupation_summary = df.groupby(occupation_col)["W_global"].agg([
    "mean",
    "median",
    "std",
    "count"
])

print("\n=== OCCUPATION BURDEN SUMMARY ===")
print(occupation_summary.round(3))


# =====================================================
# 5. KRUSKAL-WALLIS TEST
# =====================================================

groups = [
    grp["W_global"].dropna().values
    for _, grp in df.groupby(occupation_col)
]

H, p_kw = kruskal(*groups)

print("\n=== OCCUPATION DIFFERENCE TEST ===")
print(f"Kruskal-Wallis H = {H:.3f}")
print(f"p-value = {p_kw:.6f}")


# =====================================================
# 6. POST-HOC PAIRWISE COMPARISONS
# =====================================================

pairwise_results = []

occupations = df[occupation_col].dropna().unique()

for occ1, occ2 in combinations(occupations, 2):

    g1 = df[df[occupation_col] == occ1]["W_global"].dropna()
    g2 = df[df[occupation_col] == occ2]["W_global"].dropna()

    U, p_raw = mannwhitneyu(g1, g2, alternative="two-sided")

    pairwise_results.append({
        "Comparison": f"{occ1} vs {occ2}",
        "U_statistic": U,
        "p_raw": p_raw
    })

pairwise_df = pd.DataFrame(pairwise_results)

# Šidák correction
m = len(pairwise_df)
pairwise_df["p_sidak"] = 1 - (1 - pairwise_df["p_raw"])**m
pairwise_df["p_sidak"] = pairwise_df["p_sidak"].clip(upper=1)

print("\n=== POST-HOC PAIRWISE OCCUPATION COMPARISONS ===")
print(pairwise_df.round(4))


# =====================================================
# 7. FIGURE — OCCUPATION VS W_global
# =====================================================

import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = True

occupation_labels_en = {
    "kierowca": "Driver",
    "ratownik medyczny": "Paramedic",
    "maszynista": "Train Operator"
}

df_plot = df.copy()
df_plot[occupation_col] = df_plot[occupation_col].map(occupation_labels_en)

plt.figure(figsize=(7.2, 5.0))

sns.boxplot(
    data=df_plot,
    x=occupation_col,
    y="W_global"
)

sns.pointplot(
    data=df_plot,
    x=occupation_col,
    y="W_global",
    estimator="mean",
    errorbar=None,
    color="black",
    markers="D"
)

plt.title("Global Burden Index Across Occupational Groups", fontsize=12)
plt.xlabel("Occupation", fontsize=10)
plt.ylabel("W_global", fontsize=10)

plt.xticks(rotation=20, fontsize=9)
plt.yticks(fontsize=9)

plt.tight_layout()

plt.savefig(
    "figure_occupation_vs_wglobal.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =====================================================
# 8. EXPORT
# =====================================================

occupation_summary.to_excel("occupation_burden_summary.xlsx")
pairwise_df.to_excel("occupation_pairwise_comparisons.xlsx", index=False)

print("\nExported:")
print("- occupation_burden_summary.xlsx")
print("- occupation_pairwise_comparisons.xlsx")
print("- occupation_vs_wglobal.png")