# step1_load_clean.py

import pandas as pd
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import kruskal
import scikit_posthocs as sp

# ETAP 1 — Wczytanie i wstępna normalizacja

df = pd.read_excel("EXept/dane.xlsx")

# Podgląd
print(df.head())
print(df.columns)

# ETAP 2 — Normalizacja tekstu

def normalize_text(x):
    if isinstance(x, str):
        return x.strip().lower()
    return x

# poprawka dla pandas 2.x
df = df.apply(lambda col: col.map(normalize_text))

# ETAP 3.1 — Mapowanie podstawowych kategorii

# TAK/NIE/MOŻE
binary_map = {
    "tak": 1,
    "nie": 0,
    "może": 0.5
}

df = df.replace(binary_map)

# Skale porządkowe
scale_map = {
    "mały": 1,
    "średni": 2,
    "duży": 3,
    "bardzo duży": 4
}

df = df.replace(scale_map)

# Kondycja psychofizyczna
condition_map = {
    "zła": 1,
    "nie mam zdania": 2,
    "dobra": 3,
    "bardzo dobra": 4
}

df = df.replace(condition_map)

# ETAP 3.2 — czyszczenie nazw kolumn

def clean_column_name(col):
    col = col.lower()
    col = col.strip()
    col = col.replace("\t", "")
    col = col.replace("  ", " ")
    return col

df.columns = [clean_column_name(col) for col in df.columns]

print(df.columns)

# ETAP 3.3 — mapowanie na sensowne nazwy

column_map = {
    "1.wiek": "wiek",
    "2. płeć": "plec",
    "3. wzrost": "wzrost",
    "6. staż pracy na danym stanowisku pracy w transporcie": "staz_pracy",
    "7. ile godzin dziennie pracuje pani/pan na danym stanowisku pracy?": "godziny_dziennie",
    "8. jak ocenia pani/pan poziom odczuwalnego stresu?": "stres",
    "9. jak ocenia pani/pan swoją kondycję psychofizyczną?": "kondycja",
    "10. jaki poziom zmęczenia odczuwa pani/pan pod koniec pracy ?": "zmeczenie",
    "jakie:": "dolegliwosci",
    "jeśli tak, to jakie?": "niedogodnosci",
    "37.jak oceniasz ciężkość wykonywanej pracy?": "ciezkosc_pracy",
    "43. czy hałas na stanowisku pracy jest uciążliwy?": "halas",
    "42. czy na stanowisku pracy są odczuwalne drgania?": "drgania",
    "4. na jakim stanowisku pracy pani/pan pracuje?": "zawod"
}

df = df.rename(columns=column_map)

print(df.columns)

def normalize_zawod(x):
    if not isinstance(x, str):
        return np.nan
    
    x = x.strip()
    
    if "ratownik" in x:
        return "ratownik"
    elif "kierowca" in x:
        return "kierowca"
    elif "maszynista" in x:
        return "maszynista"
    else:
        return "inne"

df["zawod"] = df["zawod"].apply(normalize_zawod)

# ETAP 4 — Parsowanie liczb

def extract_years(x):
    if isinstance(x, str):
        match = re.search(r"\d+[.,]?\d*", x)
        if match:
            return float(match.group(0).replace(",", "."))
        else:
            return np.nan
    return x

df["staz_pracy"] = df["staz_pracy"].apply(extract_years)
df["staz_pracy"] = pd.to_numeric(df["staz_pracy"], errors="coerce")

print("Braki w staz_pracy:", df["staz_pracy"].isna().sum())

# ETAP 5 — Tekst (Tylko przygotowanie)

text_columns = ["dolegliwosci", "niedogodnosci"]

for col in text_columns:
    df[col] = df[col].fillna("")

# ETAP 6 — Szybka kontrola jakości

print(df.info())
print(df.describe())

# ETAP 7 — mapowanie ciezkosc_pracy

ciezkosc_map = {
    "lekka": 1,
    "średnio ciężka": 2,
    "ciężka": 3,
    "bardzo ciężka": 4
}

df["ciezkosc_pracy"] = df["ciezkosc_pracy"].replace(ciezkosc_map)
df["ciezkosc_pracy"] = pd.to_numeric(df["ciezkosc_pracy"], errors="coerce")

# ETAP 8 — hałas i drgania

df["halas"] = pd.to_numeric(df["halas"], errors="coerce")
df["drgania"] = pd.to_numeric(df["drgania"], errors="coerce")

# ETAP 9 — PCA

features = ["stres", "zmeczenie", "kondycja", "ciezkosc_pracy", "halas", "drgania"]

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

X = df[features].copy()

# KLUCZOWY KROK — standaryzacja w obrębie respondenta
X = X.sub(X.mean(axis=1), axis=0)
X = X.div(X.std(axis=1), axis=0)

X = X.dropna()

# (opcjonalnie — możesz zostawić lub usunąć)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

X_scaled = X.values

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

# ETAP 9.1 — SCREE PLOT (WYKRES OSYPISKA)

explained_var = pca.explained_variance_ratio_

plt.figure(figsize=(6,4))

plt.plot(range(1, len(explained_var)+1), explained_var, marker='o')
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance")
plt.title("Explained Variance of Structural PCA Components")

plt.xticks(range(1, len(explained_var)+1))

# opcjonalnie: linia pomocnicza (np. 3 komponenty)
plt.axvline(x=3, linestyle='--')

plt.grid(alpha=0.3)
sns.set_style("white")

plt.tight_layout()
plt.show()

# ETAP 10 — ładunki PCA

loadings = pd.DataFrame(
    pca.components_,
    columns=features,
    index=[f"PC{i+1}" for i in range(len(features))]
)

print(loadings)

# =========================================================
# PCA LOADINGS HEATMAP (MDPI-STYLE)
# =========================================================

import matplotlib as mpl
mpl.rcParams['axes.unicode_minus'] = True

# English labels for variables
feature_labels_en = {
    "stres": "Stress",
    "zmeczenie": "Fatigue",
    "kondycja": "Condition",
    "ciezkosc_pracy": "Workload",
    "halas": "Noise",
    "drgania": "Vibration"
}

loadings_plot = loadings.copy()
loadings_plot.columns = [feature_labels_en.get(col, col) for col in loadings_plot.columns]

# Unicode minus in annotations
annot_labels = loadings_plot.copy()

for col in annot_labels.columns:
    annot_labels[col] = annot_labels[col].map(
        lambda x: f"{x:.2f}".replace("-", "−") if pd.notnull(x) else ""
    )

plt.figure(figsize=(7.2, 5.2))

sns.heatmap(
    loadings_plot,
    annot=annot_labels,
    fmt="",
    cmap="coolwarm",
    center=0,
    cbar_kws={"label": "Loading"}
)

plt.title("PCA loading matrix", fontsize=12)
plt.xlabel("Variable", fontsize=11)
plt.ylabel("Principal component", fontsize=11)

plt.xticks(rotation=30, ha="right", fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()

plt.savefig(
    "figure_pca_loadings.jpg",
    format="jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

X_cluster = X_pca[:, :3]

# ETAP 10.5 — ELBOW METHOD (wybór liczby klastrów)

inertia = []
k_range = range(1, 10)

for k in k_range:
    kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_test.fit(X_cluster)
    inertia.append(kmeans_test.inertia_)

plt.figure(figsize=(6,4))
plt.plot(k_range, inertia, marker='o')

plt.xlabel("Number of clusters (k)")
plt.ylabel("Within-cluster sum of squares (Inertia)")
plt.title("Elbow Plot for Cluster Selection")

plt.xticks(k_range)
plt.grid(alpha=0.3)

# opcjonalnie: zaznaczenie k=4
plt.axvline(x=4, linestyle='--')

plt.tight_layout()
plt.savefig("elbow_plot.png", dpi=300)
plt.show()

# ETAP 11 — Klasteryzacja

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_cluster)

# WALIDACJA KLASTRÓW (silhouette)
from sklearn.metrics import silhouette_score

score = silhouette_score(X_cluster, clusters)
print("\nSilhouette score:", round(score, 3))

# KLUCZOWE — spójna ramka
df_clean = df.loc[X.index].copy()
df_clean["cluster"] = clusters

print(df_clean["cluster"].value_counts())

# =========================================
# PREP FOR FUZZY (NIC NIE ZMIENIA)
# =========================================

df_clean["stres_raw"] = df_clean["stres"]
df_clean["zmeczenie_raw"] = df_clean["zmeczenie"]
df_clean["kondycja_raw"] = df_clean["kondycja"]
df_clean["ciezkosc_raw"] = df_clean["ciezkosc_pracy"]
df_clean["halas_raw"] = df_clean["halas"]
df_clean["drgania_raw"] = df_clean["drgania"]

# ETAP 12 — Boxploty (zbiorcze)

features_plot = ["stres", "zmeczenie", "kondycja", "ciezkosc_pracy", "halas", "drgania"]

plt.figure(figsize=(12,8))

for i, col in enumerate(features_plot, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x="cluster", y=col, data=df_clean)
    plt.title(col)

plt.tight_layout()
#plt.show()

# ETAP 12.5 — TEST KRUSKALA-WALLISA

print("\n=== TEST KRUSKALA-WALLISA ===")

features_test = ["stres", "zmeczenie", "kondycja", "ciezkosc_pracy", "halas", "drgania"]

for col in features_test:
    
    groups = []
    
    for k in sorted(df_clean["cluster"].unique()):
        g = df_clean[df_clean["cluster"] == k][col]
        
        g = pd.to_numeric(g, errors="coerce").dropna()
        
        if len(g) > 1:
            groups.append(g.values)
    
    if len(groups) < 2:
        print(f"\n{col} — za mało danych")
        continue
    
    stat, p = kruskal(*groups)
    
    N = sum(len(g) for g in groups)
    eta2 = stat / (N - 1)
    
    print(f"\n{col}")
    print(f"H = {stat:.3f}, p = {p:.5f}, eta² = {eta2:.3f}")

# post hoc test Dunn’a:

print("\n=== POST-HOC (Dunn test) ===")

for col in features_test:
    
    data = df_clean[[col, "cluster"]].dropna()
    
    dunn = sp.posthoc_dunn(
        data,
        val_col=col,
        group_col="cluster",
        p_adjust="bonferroni"
    )
    
    print(f"\nDunn test dla: {col}")
    print(dunn)

    N = len(df_clean)
    #eta2 = stat / (N - 1)
    #print(f"eta^2 ≈ {eta2:.3f}")

# ETAP 12.6 — FIGURA PUBLIKACYJNA (A + B)

import matplotlib.pyplot as plt
import seaborn as sns

features_plot = ["stres", "zmeczenie", "kondycja", "ciezkosc_pracy", "halas", "drgania"]

fig, axes = plt.subplots(1, 2, figsize=(14,6))

# =========================================================
# FIGURE PUBLICATION-READY (MDPI STYLE)
# =========================================================

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

# ======================
# PANEL A — BOXPLOT + POINTS
# ======================

df_melt = df_clean.melt(
    id_vars="cluster",
    value_vars=features_plot,
    var_name="variable",
    value_name="value"
)

df_melt["variable"] = df_melt["variable"].map(feature_labels_en)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(
    x="variable",
    y="value",
    hue="cluster",
    data=df_melt,
    ax=axes[0]
)

sns.stripplot(
    x="variable",
    y="value",
    hue="cluster",
    data=df_melt,
    dodge=True,
    alpha=0.3,
    ax=axes[0]
)

axes[0].set_title("A. Variable distributions across clusters", fontsize=11)
axes[0].set_xlabel("Variable", fontsize=10)
axes[0].set_ylabel("Response", fontsize=10)
axes[0].tick_params(axis='x', rotation=35, labelsize=9)

handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles[:4], labels[:4], title="Cluster", fontsize=9)

# ======================
# PANEL B — CLUSTER PROFILE HEATMAP
# ======================

cluster_means = (
    df_clean[features_plot]
    .apply(pd.to_numeric, errors="coerce")
    .assign(cluster=df_clean["cluster"])
    .groupby("cluster")
    .mean()
)

cluster_means.columns = [feature_labels_en.get(col, col) for col in cluster_means.columns]

annot_labels = cluster_means.copy()

for col in annot_labels.columns:
    annot_labels[col] = annot_labels[col].map(
        lambda x: f"{x:.2f}".replace("-", "−") if pd.notnull(x) else ""
    )

sns.heatmap(
    cluster_means,
    annot=annot_labels,
    fmt="",
    cmap="coolwarm",
    ax=axes[1]
)

axes[1].set_title("B. Cluster profiles (mean values)", fontsize=11)
axes[1].set_xlabel("Variable", fontsize=10)
axes[1].set_ylabel("Cluster", fontsize=10)

plt.tight_layout()

plt.savefig(
    "figure_cluster_profiles_combined.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ETAP 13 — ANALIZA ZAWODÓW

zawod_cluster = pd.crosstab(
    df_clean["zawod"],
    df_clean["cluster"],
    normalize="columns"
)

print("\nStruktura zawodów w klastrach:")
print(zawod_cluster)

zawod_cluster.T.plot(kind="bar", stacked=True, figsize=(8,5))
plt.title("Struktura zawodów w klastrach")
plt.ylabel("udział")
plt.tight_layout()
#plt.show()

# =========================================================
# ETAP 13 — OCCUPATIONAL STRUCTURE BY CLUSTER (MDPI STYLE)
# =========================================================

zawod_cluster = pd.crosstab(
    df_clean["zawod"],
    df_clean["cluster"],
    normalize="columns"
)

print("\nOccupational structure across clusters:")
print(zawod_cluster)

occupation_labels_en = {
    "kierowca": "Driver",
    "maszynista": "Train operator",
    "ratownik": "Paramedic"
}

zawod_cluster.index = [
    occupation_labels_en.get(idx, idx)
    for idx in zawod_cluster.index
]

ax = zawod_cluster.T.plot(
    kind="bar",
    stacked=True,
    figsize=(6.2, 4.2)
)

plt.title("Occupational structure across clusters", fontsize=11)
plt.xlabel("Cluster", fontsize=10)
plt.ylabel("Share", fontsize=10)

plt.xticks(rotation=0, fontsize=9)
plt.yticks(fontsize=9)

plt.legend(
    title="Occupation",
    fontsize=9,
    title_fontsize=9,
    frameon=True
)

plt.tight_layout()

plt.savefig(
    "figure_occupation_structure.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ETAP 13.5 — czyszczenie płci

def normalize_gender(x):
    if not isinstance(x, str):
        return np.nan
    
    x = x.strip().lower()
    
    # usunięcie polskich znaków (kluczowe!)
    x = x.replace("ę", "e").replace("ą", "a").replace("ś", "s") \
         .replace("ł", "l").replace("ż", "z").replace("ź", "z") \
         .replace("ć", "c").replace("ń", "n").replace("ó", "o")
    
    if "kob" in x:
        return "kobieta"
    if "mez" in x or "mesk" in x:
        return "mężczyzna"
    
    return np.nan  # wszystko inne traktujemy jako brak

df_clean["plec"] = df_clean["plec"].apply(normalize_gender)

print("\nUnikalne wartości płci po czyszczeniu:")
print(df_clean["plec"].value_counts())

# ETAP 14 — ANALIZA PŁCI

plec_cluster = pd.crosstab(
    df_clean["plec"],
    df_clean["cluster"],
    normalize="columns"
)

print("\nStruktura płci w klastrach:")
print(plec_cluster)

plec_cluster.T.plot(kind="bar", stacked=True, figsize=(6,4))
plt.title("Struktura płci w klastrach")
plt.ylabel("udział")
plt.tight_layout()
#plt.show()


# =========================================================
# ETAP 14 — GENDER STRUCTURE BY CLUSTER (MDPI STYLE)
# =========================================================

plec_cluster = pd.crosstab(
    df_clean["plec"],
    df_clean["cluster"],
    normalize="columns"
)

print("\nGender structure across clusters:")
print(plec_cluster)

gender_labels_en = {
    "kobieta": "Female",
    "mężczyzna": "Male"
}

plec_cluster.index = [
    gender_labels_en.get(idx, idx)
    for idx in plec_cluster.index
]

ax = plec_cluster.T.plot(
    kind="bar",
    stacked=True,
    figsize=(5.8, 4.0)
)

plt.title("Gender structure across clusters", fontsize=11)
plt.xlabel("Cluster", fontsize=10)
plt.ylabel("Share", fontsize=10)

plt.xticks(rotation=0, fontsize=9)
plt.yticks(fontsize=9)

plt.legend(
    title="Gender",
    fontsize=9,
    title_fontsize=9,
    frameon=True
)

plt.tight_layout()

plt.savefig(
    "figure_gender_structure.jpg",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# =========================
# ROZGAŁĘZIENIE ZAWODÓW
# =========================

df_clean["zawod"] = df.loc[df_clean.index, "zawod"]

df_ratownik = df_clean[df_clean["zawod"] == "ratownik"].copy()
df_kierowca = df_clean[df_clean["zawod"] == "kierowca"].copy()
df_maszynista = df_clean[df_clean["zawod"] == "maszynista"].copy()

print("\nLiczebności:")
print("ratownik:", len(df_ratownik))
print("kierowca:", len(df_kierowca))
print("maszynista:", len(df_maszynista))