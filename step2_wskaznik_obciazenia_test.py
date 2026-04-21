# =========================================
# IMPORTY
# =========================================

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr



# =========================================
# STEP 0 — PRZYGOTOWANIE DANYCH (WSPÓLNE)
# =========================================

df = pd.read_excel("Exept\dane.xlsx")

# --- normalizacja tekstu ---
def normalize_text(x):
    if isinstance(x, str):
        return x.strip().lower()
    return x

df = df.apply(lambda col: col.map(normalize_text))

# --- mapowania ---
binary_map = {"tak": 1, "nie": 0, "może": 0.5}
scale_map = {"mały": 1, "średni": 2, "duży": 3, "bardzo duży": 4}
condition_map = {"zła": 1, "nie mam zdania": 2, "dobra": 3, "bardzo dobra": 4}
ciezkosc_map = {"lekka": 1, "średnio ciężka": 2, "ciężka": 3, "bardzo ciężka": 4}

df = df.replace(binary_map)
df = df.replace(scale_map)
df = df.replace(condition_map)

# --- czyszczenie nazw kolumn ---
def clean_column_name(col):
    return col.lower().strip().replace("\t", "").replace("  ", " ")

df.columns = [clean_column_name(col) for col in df.columns]

# 🔍 DEBUG — lista kolumn
print("\n--- LISTA KOLUMN ---")
for col in df.columns:
    print(f"'{col}'")

# =========================================
# WYBÓR KOLUMNY DOLEGLIWOŚCI (NA SZTYWNO)
# =========================================

pain_col = [col for col in df.columns if "dolegliwości" in col][0]

print("\nUŻYWANA KOLUMNA:", pain_col)
print(df[pain_col].head())

# =========================================
# MAPOWANIE NAZW
# =========================================

column_map = {
    "1.wiek": "wiek",
    "2. płeć": "plec",
    "3. wzrost": "wzrost",
    "6. staż pracy na danym stanowisku pracy w transporcie": "staz_pracy",
    "7. ile godzin dziennie pracuje pani/pan na danym stanowisku pracy?": "godziny_dziennie",
    "8. jak ocenia pani/pan poziom odczuwalnego stresu?": "stres",
    "9. jak ocenia pani/pan swoją kondycję psychofizyczną?": "kondycja",
    "10. jaki poziom zmęczenia odczuwa pani/pan pod koniec pracy ?": "zmeczenie",
    "37.jak oceniasz ciężkość wykonywanej pracy?": "ciezkosc_pracy",
    "43. czy hałas na stanowisku pracy jest uciążliwy?": "halas",
    "42. czy na stanowisku pracy są odczuwalne drgania?": "drgania",
    "4. na jakim stanowisku pracy pani/pan pracuje?": "zawod"
}

df = df.rename(columns=column_map)

# --- normalizacja zawodu ---
def normalize_zawod(x):
    if not isinstance(x, str):
        return np.nan
    x = x.strip().lower()
    if "ratownik" in x:
        return "ratownik"
    elif "kierowca" in x:
        return "kierowca"
    elif "maszynista" in x:
        return "maszynista"
    else:
        return "inne"

df["zawod"] = df["zawod"].apply(normalize_zawod)

# --- liczby ---
def extract_years(x):
    if isinstance(x, str):
        match = re.search(r"\d+[.,]?\d*", x)
        return float(match.group(0).replace(",", ".")) if match else np.nan
    return x

df["staz_pracy"] = df["staz_pracy"].apply(extract_years)
df["staz_pracy"] = pd.to_numeric(df["staz_pracy"], errors="coerce")

# --- konwersje ---
df["ciezkosc_pracy"] = df["ciezkosc_pracy"].replace(ciezkosc_map)
df["ciezkosc_pracy"] = pd.to_numeric(df["ciezkosc_pracy"], errors="coerce")

df["halas"] = pd.to_numeric(df["halas"], errors="coerce")
df["drgania"] = pd.to_numeric(df["drgania"], errors="coerce")

MODE = "ratownik"

# =========================================
# STEP 0.5 — DOLEGLIWOŚCI (KODOWANIE)
# =========================================

def parse_pain(x):
    if not isinstance(x, str):
        return []

    x = x.lower()

    # obsługa różnych separatorów
    if "," in x:
        return [p.strip() for p in x.split(",") if p.strip()]
    elif ";" in x:
        return [p.strip() for p in x.split(";") if p.strip()]
    else:
        return [x.strip()]


def classify_pain(parts):
    plecy = 0
    konczyny = 0

    for p in parts:

        # KRĘGOSŁUP / TUŁÓW
        if any(k in p for k in [
            "szyi",
            "górnej części pleców",
            "środkowej części pleców",
            "dolnej części pleców",
            "plec"
        ]):
            plecy += 1

        # KOŃCZYNY
        elif any(k in p for k in [
            "rąk",
            "ramion",
            "bark",
            "nóg",
            "kolan"
        ]):
            konczyny += 1

    return plecy, konczyny


plecy_list = []
konczyny_list = []
liczba_list = []

for val in df[pain_col]:
    parts = parse_pain(val)

    # DEBUG — pierwsze 5
    if len(plecy_list) < 5:
        print("\nRAW:", val)
        print("PARSED:", parts)

    plecy, konczyny = classify_pain(parts)

    plecy_list.append(plecy)
    konczyny_list.append(konczyny)
    liczba_list.append(len(parts))


df["bol_plecy"] = plecy_list
df["bol_konczyny"] = konczyny_list
df["bol_liczba"] = liczba_list

print("\nPodsumowanie dolegliwości:")
print(df[["bol_plecy", "bol_konczyny", "bol_liczba"]].describe())

print("\nKorelacja bóle plecy vs kończyny:")
print(df[["bol_plecy", "bol_konczyny"]].corr())

# =========================================
# STEP 0.6 — NASILENIE DOLEGLIWOŚCI
# =========================================

pain_level_map = {
    "małe": 1,
    "średnie": 2,
    "duże": 3,
    "bardzo duże": 4
}

pain_level_col = "jakie:"

df["bol_poziom"] = df[pain_level_col].map(pain_level_map)

# WSKAŹNIKI WAŻONE
df["bol_plecy_w"] = df["bol_plecy"] * df["bol_poziom"]
df["bol_konczyny_w"] = df["bol_konczyny"] * df["bol_poziom"]
df["bol_total_w"] = df["bol_liczba"] * df["bol_poziom"]

print("\nPodsumowanie bólu ważonego:")
print(df[["bol_plecy_w", "bol_konczyny_w", "bol_total_w"]].describe())

# =========================================
# STEP 1 — STRUKTURA (PCA + KLASTRY)
# =========================================

#features = ["stres", "zmeczenie", "kondycja", "ciezkosc_pracy", "halas", "drgania"]
#features = ["stres", "zmeczenie", "kondycja", "ciezkosc_pracy", "halas"]


# =========================================
# CLEAN X do KMO (KRYTYCZNE)
# =========================================

features = ["stres", "zmeczenie", "kondycja", "ciezkosc_pracy", "halas", "drgania"]
X = df[features].copy()

# normalizacja w obrębie respondenta
#X = X.sub(X.mean(axis=1), axis=0)
#X = X.div(X.std(axis=1), axis=0)
#X = X.dropna()

X = df[features].copy()

# 1. tylko liczby
X = X.apply(pd.to_numeric, errors="coerce")

# 2. usuń NaN
X = X.dropna()

# 3. usuń stałe kolumny (zero wariancji)
X = X.loc[:, X.std() > 0]

print("\nKształt X do KMO:", X.shape)
print("STD:\n", X.std())

# =========================================
# KMO + BARTLETT (walidacja PCA)
# =========================================

kmo_all, kmo_model = calculate_kmo(X)
bartlett_chi2, bartlett_p = calculate_bartlett_sphericity(X)

print("\nKMO:", round(kmo_model, 3))
print("Bartlett chi2:", round(bartlett_chi2, 3))
print("Bartlett p-value:", bartlett_p)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X.values)

# DODAJ TO:
df_struct = df.loc[X.index].copy()
df_struct["PC1"] = X_pca[:, 0]

print("\nSTEP 1 — Explained variance:")
print(pca.explained_variance_ratio_)

# klasteryzacja
X_cluster = X_pca[:, :3]
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_cluster)

df_struct["cluster"] = clusters

print("\nRozkład klastrów:")
print(df_struct["cluster"].value_counts())

# =========================================
# STEP 2 — POZIOM (PCA + WSKAŹNIK)
# =========================================

X_level = df[features].copy()

# kierunek: wszystko = im więcej tym gorzej
X_level["kondycja"] = 5 - X_level["kondycja"]

# numeric + czyszczenie
for col in X_level.columns:
    X_level[col] = pd.to_numeric(X_level[col], errors="coerce")

X_level = X_level.dropna()

# standaryzacja
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_level)

# PCA → 1 komponent
pca_level = PCA(n_components=1)
W_raw = pca_level.fit_transform(X_scaled)

print("\nSTEP 2 — Explained variance:")
print(pca_level.explained_variance_ratio_)

# kierunek komponentu
loadings = pd.Series(pca_level.components_[0], index=features)
print("\nŁadunki STEP 2:")
print(loadings)

if loadings.mean() < 0:
    W_raw = -W_raw

W_raw = W_raw.flatten()

# percentyle
W_percentile = pd.Series(W_raw).rank(pct=True)

df_level = df.loc[X_level.index].copy()
df_level["W_percentile"] = W_percentile


# klasy poziomu
def interpret_level(x):
    if x < 0.25:
        return "niskie"
    elif x < 0.5:
        return "umiarkowane"
    elif x < 0.75:
        return "podwyższone"
    else:
        return "wysokie"

df_level["poziom"] = df_level["W_percentile"].apply(interpret_level)

print("\nRozkład poziomów:")
print(df_level["poziom"].value_counts())

# =========================
# STEP 2 B AUTO-WAGI (PCA + korelacje)
# =========================

# PCA loadings (bezwzględne)
weights_pca = loadings.abs()

# korelacje z PC1
corr = df_level[features + ["W_percentile"]].corr()["W_percentile"].drop("W_percentile").abs()

# połączenie (średnia)
weights = (weights_pca + corr) / 2

# normalizacja do sumy = 1
weights = weights / weights.sum()

print("\n=== AUTO WAGI ===")
print(weights.sort_values(ascending=False))


# =========================================
# STEP 2C — FUZZY MODEL
# =========================================

print("\n==============================")
print("=== STEP 2C — FUZZY MODEL ===")
print("==============================")

# =========================
# FUNKCJE PRZYNALEŻNOŚCI
# =========================

def trapmf(x, a, b, c, d):
    return np.maximum(0, np.minimum(
        np.minimum((x - a) / (b - a + 1e-6), 1),
        (d - x) / (d - c + 1e-6)
    ))

def trimf(x, a, b, c):
    return np.maximum(0, np.minimum(
        (x - a) / (b - a + 1e-6),
        (c - x) / (c - b + 1e-6)
    ))

# =========================
# NORMALIZACJA (ROBUST)
# =========================

X_fuzzy = X_level.copy()

for col in X_fuzzy.columns:
    q1 = X_fuzzy[col].quantile(0.05)
    q9 = X_fuzzy[col].quantile(0.95)
    X_fuzzy[col] = (X_fuzzy[col] - q1) / (q9 - q1)
    X_fuzzy[col] = X_fuzzy[col].clip(0, 1)

print("\n=== ROZKŁAD X_fuzzy ===")
print(X_fuzzy.describe())

# =========================
# MEMBERSHIP FUNCTIONS
# =========================

fuzzy_inputs = {}

for col in X_fuzzy.columns:
    x = X_fuzzy[col]

    fuzzy_inputs[col] = {
        "low": trapmf(x, 0.0, 0.0, 0.2, 0.4),
        "medium": trimf(x, 0.3, 0.5, 0.7),
        "high": trapmf(x, 0.4, 0.6, 1.0, 1.0)
    }

print("\nDEBUG membership:")
print("stres high mean:", fuzzy_inputs["stres"]["high"].mean())
print("zmeczenie high mean:", fuzzy_inputs["zmeczenie"]["high"].mean())
print("kondycja low mean:", fuzzy_inputs["kondycja"]["low"].mean())

# wzmocnienie osi psychicznej
fuzzy_inputs["stres"]["high"] = np.clip(fuzzy_inputs["stres"]["high"] * 1.5, 0, 1)
fuzzy_inputs["zmeczenie"]["high"] = np.clip(fuzzy_inputs["zmeczenie"]["high"] * 1.5, 0, 1)

def compute_W_fuzzy(weights, alpha):

    W = []

    for i in range(len(X_fuzzy)):

        # =========================
        # 🔵 BLOK MENTALNY (KLUCZ!)
        # =========================

        stres_h = fuzzy_inputs["stres"]["high"].iloc[i]
        zmeczenie_h = fuzzy_inputs["zmeczenie"]["high"].iloc[i]
        kondycja_l = fuzzy_inputs["kondycja"]["low"].iloc[i]

        # 🔴 SYNERGIA (KLUCZ!)
        synergy = stres_h * zmeczenie_h

        mental = (
            weights["stres"] * stres_h +
            weights["zmeczenie"] * zmeczenie_h +
            weights["kondycja"] * kondycja_l +
            2.0 * synergy
        )

        # =========================
        # 🟠 BLOK FIZYCZNY (słabszy)
        # =========================

        physical = (
            0.7 * fuzzy_inputs["ciezkosc_pracy"]["high"].iloc[i] +
            0.3 * fuzzy_inputs["halas"]["high"].iloc[i]
        )

        # =========================
        # 🔴 FINALNA KOMBINACJA
        # =========================

        val = np.power(0.9 * mental + 0.1 * physical, 1.8)

        if synergy > 0.5:
            val += 0.3

        W.append(val)

    return np.array(W)


alphas = np.linspace(0.1, 2.0, 20)

results = []
weights_dict = weights.to_dict()

for a in alphas:
    W_tmp = compute_W_fuzzy(weights_dict, a)

    # dopasuj indeksy!
    df_tmp = df_level.copy()
    df_tmp["W_tmp"] = W_tmp

    valid = df_tmp[["W_tmp", "W_percentile"]].dropna()

    if len(valid) > 0:
        rho, _ = spearmanr(valid["W_tmp"], valid["W_percentile"])
    else:
        rho = 0

    results.append((a, rho))

results_df = pd.DataFrame(results, columns=["alpha", "rho"])

print("\n=== TUNING ALPHA ===")
print(results_df.sort_values("rho", ascending=False).head())

best_alpha = results_df.sort_values("rho", ascending=False).iloc[0]["alpha"]

print("\nBEST ALPHA:", best_alpha)

print("\n=== KOLUMNY df_level ===")
for col in df_level.columns:
    print(col)

best_rho = -1
best_w = None

for _ in range(500):

    w = np.random.rand(len(features))
    w = w / w.sum()

    weights_tmp = dict(zip(features, w))

    W_tmp = compute_W_fuzzy(weights_tmp, best_alpha)

    df_tmp = df_level.copy()
    df_tmp["W_tmp"] = W_tmp

    valid = df_tmp[["W_tmp", "W_percentile"]].dropna()

    rho, _ = spearmanr(valid["W_tmp"], valid["W_percentile"])

    if rho > best_rho:
        best_rho = rho
        best_w = weights_tmp

print("\n=== BEST FUZZY (vs PCA) ===")
print("rho:", best_rho)
print("weights:", best_w)

# finalny model

W_fuzzy_final = compute_W_fuzzy(best_w, best_alpha)
df_level["W_fuzzy"] = W_fuzzy_final
df_level["W_fuzzy_pct"] = pd.Series(W_fuzzy_final).rank(pct=True)


# =========================================
# KONTROLA — PCA vs FUZZY
# =========================================

print("\n=== PCA vs FUZZY ===")
print(df_level[["W_percentile", "W_fuzzy_pct"]].corr())

print("\nW_fuzzy vs klastry:")
print(df_struct.join(df_level["W_fuzzy_pct"]).groupby("cluster")["W_fuzzy_pct"].mean())

# =========================================
# KONTROLA STRUKTURY KLASTRÓW (TU!)
# =========================================

df_check = df_struct.join(
    df_level[["W_percentile","W_fuzzy_pct"]]
)

print("\n=== KONTROLA KLASTRÓW ===")

print("\nStres / zmęczenie / kondycja:")
print(df_check.groupby("cluster")[["stres","zmeczenie","kondycja"]].mean())

print("\nPCA vs fuzzy w klastrach:")
print(df_check.groupby("cluster")[["W_percentile","W_fuzzy_pct"]].mean())

print("\nRóżnica fuzzy - PCA:")
df_check["diff"] = df_check["W_fuzzy_pct"] - df_check["W_percentile"]
print(df_check.groupby("cluster")["diff"].mean())

# =========================================
# SPÓJNOŚĆ SKALI (CRONBACH ALPHA)
# =========================================

def cronbach_alpha(df):
    df = df.dropna()
    k = df.shape[1]
    variances = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - variances.sum() / total_var)

features = ["stres", "zmeczenie", "kondycja", "ciezkosc_pracy", "halas", "drgania"]
alpha = cronbach_alpha(df[features])

print("\nCronbach alpha:", round(alpha, 3))

# =========================================
# STEP 3 — BÓL vs OBCIĄŻENIE
# =========================================

print("\n=== BÓL vs OBCIĄŻENIE ===")

print("\nbol_total_w vs PC1:")
print(df_struct[["bol_total_w", "PC1"]].corr())

print("\nbol_plecy_w vs PC1:")
print(df_struct[["bol_plecy_w", "PC1"]].corr())

print("\nbol_konczyny_w vs PC1:")
print(df_struct[["bol_konczyny_w", "PC1"]].corr())

# =========================
# KONTROLA: korelacje z PC1
# =========================

pc1 = W_raw  # główny komponent

df_check = X_level.copy()
df_check["PC1"] = pc1

print("\nKorelacje zmiennych z PC1 (czy to jest 'obciążenie'):")
print(df_check.corr()["PC1"].sort_values(ascending=False))

# =========================================
# STEP 2B — PCA PER ZAWÓD (NOWE)
# =========================================

df_level["zawod"] = df.loc[X_level.index, "zawod"]

df_level_zawod = []

for zawod in ["ratownik", "kierowca", "maszynista"]:
    
    df_z = df_level[df_level["zawod"] == zawod].copy()
    
    if len(df_z) < 10:
        print(f"\n⚠️ Za mało danych dla: {zawod}")
        continue
    
    X_z = X_level.loc[df_z.index].copy()
    
    # standaryzacja w obrębie zawodu
    scaler_z = StandardScaler()
    X_scaled_z = scaler_z.fit_transform(X_z)
    
    # PCA
    pca_z = PCA(n_components=1)
    W_raw_z = pca_z.fit_transform(X_scaled_z).flatten()
    
    # kierunek
    loadings_z = pca_z.components_[0]
    if loadings_z.mean() < 0:
        W_raw_z = -W_raw_z
    
    # percentyle
    W_percentile_z = pd.Series(W_raw_z, index=df_z.index).rank(pct=True)
    
    df_z["W_zawod"] = W_percentile_z
    
    df_level_zawod.append(df_z[["W_zawod"]])    
    
    print(f"\n{zawod} — PCA variance:")
    print(pca_z.explained_variance_ratio_)
    
# =========================
# SCALENIE WYNIKÓW
# =========================

df_zawod_scores = pd.concat(df_level_zawod)

df_level = df_level.join(df_zawod_scores, how="left")

print("\nCzy coś jest w df_level_zawod?")
print(len(df_level_zawod))

print("\nKolumny df_level:")
print(df_level.columns)

print("\nPorównanie global vs zawod:")
print(df_level.groupby("zawod")[["W_percentile", "W_zawod"]].mean())

# =========================================
# WIZUALIZACJA — GLOBAL vs ZAWÓD (PUBLIKACYJNA)
# =========================================

# mapowanie nazw zawodów na angielski
occupation_map = {
    "ratownik": "Paramedic",
    "kierowca": "Driver",
    "maszynista": "Train driver"
}

df_plot = df_level[["zawod", "W_percentile", "W_zawod"]].dropna().copy()
df_plot["zawod"] = df_plot["zawod"].map(occupation_map)

# long format
df_plot = df_plot.melt(
    id_vars="zawod",
    value_vars=["W_percentile", "W_zawod"],
    var_name="typ_wskaznika",
    value_name="wartosc"
)

# kolejność zawodów (od najwyższego obciążenia globalnego)
order = ["Paramedic", "Driver", "Train driver"]

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df_plot,
    x="zawod",
    y="wartosc",
    hue="typ_wskaznika",
    order=order,
    palette=["#4C72B0", "#DD8452"]
)

# średnie jako romby (publication-quality)
sns.pointplot(
    data=df_plot,
    x="zawod",
    y="wartosc",
    hue="typ_wskaznika",
    order=order,
    dodge=0.4,
    linestyle='none',
    markers="d",
    palette='dark:black'
)

# usunięcie podwójnej legendy
handles, labels = plt.gca().get_legend_handles_labels()
labels = ["Global (W_percentile)", "Occupation-specific (W_zawod)"]
plt.legend(handles[:2], labels, title="Index type")

# opisy
plt.title("Comparison of global and occupation-specific burden indices across professions")
plt.ylabel("Burden index (0–1)")
plt.xlabel("Occupation")

plt.tight_layout()
plt.show()



# =========================================
# INTEGRACJA STEP 1 + STEP 2
# =========================================

df_final = df_struct.join(
    df_level[[
        "W_percentile",
        "W_zawod",
        "W_fuzzy_pct"
    ]],
    how="inner"
)

# =========================================
# KONTROLA — FUZZY vs KLASTRY
# =========================================

print("\nW_fuzzy_pct vs klastry:")
print(df_final.groupby("cluster")["W_fuzzy_pct"].mean())

print(df_final["W_percentile"].head())
df_final["poziom"] = df_final["W_zawod"].apply(interpret_level)

print("\nŚredni poziom obciążenia w klastrach:")
print(df_final.groupby("cluster")["W_percentile"].mean())

# =========================================
# WYBÓR TRYBU (NOWE)
# =========================================

df_final["zawod"] = df.loc[df_final.index, "zawod"]

df_model = df_final[df_final["zawod"] == MODE].copy()

print(f"\nTryb analizy: {MODE}")
print("Liczba obserwacji:", len(df_model))

df_level = df_level[[
    "zawod",
    "stres",
    "zmeczenie",
    "kondycja",
    "ciezkosc_pracy",
    "halas",
    "drgania",
    "bol_total_w",
    "W_percentile",
    "W_fuzzy",
    "W_fuzzy_pct",
    "W_zawod"
]]

# =========================================
# STEP 2.5 — WALIDACJA WSKAŹNIKA W
# =========================================

from scipy.stats import kruskal

print("\n=== WALIDACJA W (Spearman + Kruskal) ===")

# --- korelacje z komponentami ---
print("\nKorelacje Spearmana (W vs zmienne):")

print("\nKorelacje Spearmana (W vs zmienne):")

for col in ["stres", "zmeczenie", "kondycja", "ciezkosc_pracy", "halas", "drgania"]:
    
    x = pd.to_numeric(df_final["W_zawod"], errors="coerce")
    y = pd.to_numeric(df_final[col], errors="coerce")
    
    # wspólne dane bez NaN
    valid = x.notna() & y.notna()
    
    if valid.sum() < 5:
        print(f"{col}: za mało danych")
        continue
    
    rho, p = spearmanr(x[valid], y[valid])
    
    print(f"{col}: rho={rho:.3f}, p={p:.5f}")

# --- różnice między klastrami ---
groups = [
    df_final[df_final["cluster"] == k]["W_zawod"].dropna()
    for k in sorted(df_final["cluster"].unique())
]

stat, p = kruskal(*groups)

print("\nKruskal-Wallis dla W:")
print(f"H={stat:.3f}, p={p:.5f}")


# =========================================
# WIZUALIZACJA
# =========================================

# =========================================
# FIGURE 8 — CLUSTERS vs BURDEN INDEX
# =========================================

df_plot = df_final[["cluster", "W_percentile", "W_zawod"]].dropna().copy()

df_plot = df_plot.melt(
    id_vars="cluster",
    value_vars=["W_percentile", "W_zawod"],
    var_name="index_type",
    value_name="value"
)

label_map = {
    "W_percentile": "Global (W_percentile)",
    "W_zawod": "Occupation-specific (W_zawod)"
}
df_plot["index_type"] = df_plot["index_type"].map(label_map)

plt.figure(figsize=(7,5))

sns.boxplot(
    data=df_plot,
    x="cluster",
    y="value",
    hue="index_type",
    palette=["#4C72B0", "#DD8452"]
)

sns.pointplot(
    data=df_plot,
    x="cluster",
    y="value",
    hue="index_type",
    dodge=0.4,
    linestyle='none',
    markers="d",
    palette='dark:black'
)

# naprawa legendy
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:2], labels[:2], title="Index type")

plt.title("Distribution of burden indices across clusterss")
plt.xlabel("Cluster")
plt.ylabel("Burden index (relative scale))")

plt.tight_layout()
plt.savefig("figure_8.png", dpi=300)
plt.show()

# =========================================
# STEP 3 — INTEGRACJA FILARÓW (CSV)
# =========================================

import pandas as pd
import numpy as np

# -------------------------
# ETAP 1 — wczytanie danych
# -------------------------

emg = pd.read_csv("EXept/emg_key_metrics.csv")
mio = pd.read_csv("EXept/myo_key_metrics.csv")

print("\n=== EMG ===")
print(emg.head())

print("\n=== MYOMOTION ===")
print(mio.head())


# -------------------------
# ETAP 2 — EMG → wskaźnik
# -------------------------

# używamy tylko wartości liczbowych
emg_values = pd.to_numeric(emg["value"], errors="coerce").dropna()

print("\nEMG stats:")
print(emg_values.describe())

# wskaźnik surowy
W_emg_raw = emg_values.mean()

# normalizacja 0–1 (wewnątrz EMG)
W_emg = (W_emg_raw - emg_values.min()) / (emg_values.max() - emg_values.min())

print("\nW_emg_raw:", W_emg_raw)
print("W_emg (0-1):", W_emg)


# =========================================
# STEP 3 — INTEGRACJA FILARÓW (A i B)
# =========================================

print("\n==============================")
print("=== STEP 3 — INTEGRACJA ===")
print("==============================")

# -------------------------
# ETAP 2 — EMG → wskaźnik
# -------------------------

emg_values = pd.to_numeric(emg["value"], errors="coerce").dropna()

W_emg_raw = emg_values.mean()
W_emg = (W_emg_raw - emg_values.min()) / (emg_values.max() - emg_values.min())

print("\nEMG:")
print("W_emg_raw:", W_emg_raw)
print("W_emg (0–1):", W_emg)


# -------------------------
# ETAP 3 — MYO → wskaźnik
# -------------------------

mio_values = pd.to_numeric(mio["value"], errors="coerce").dropna()

W_myo_raw = mio_values.mean()
W_myo = (W_myo_raw - mio_values.min()) / (mio_values.max() - mio_values.min())

print("\nMYO:")
print("W_myo_raw:", W_myo_raw)
print("W_myo (0–1):", W_myo)


# -------------------------
# ETAP 4 — dołączenie
# -------------------------

df_model["W_emg"] = W_emg
df_model["W_myo"] = W_myo


# =========================================
# ŚCIEŻKA A — KONTEKSTOWA (REKOMENDOWANA)
# =========================================

print("\n==============================")
print("=== STEP 3A — MODEL KONTEKSTOWY ===")
print("==============================")

# EMG/MYO jako korekta poziomu (nie dominują)

df_model["W_total_A"] = (
    df_model["W_zawod"] * 0.8 +
    W_emg * 0.1 +
    W_myo * 0.1
)

print("\nStatystyki W_total_A:")
print(df_model["W_total_A"].describe())


# poziomy

def interpret(x):
    if x < 0.25:
        return "niskie"
    elif x < 0.5:
        return "umiarkowane"
    elif x < 0.75:
        return "podwyższone"
    else:
        return "wysokie"

df_model["poziom_A"] = df_model["W_total_A"].apply(interpret)

print("\nRozkład poziomów (A):")
print(df_model["poziom_A"].value_counts())


print("\nŚredni poziom w klastrach (A):")
print(df_model.groupby("cluster")["W_total_A"].mean())


# =========================================
# ŚCIEŻKA B — CENTROWANA (ALTERNATYWA)
# =========================================

print("\n==============================")
print("=== STEP 3B — MODEL CENTROWANY ===")
print("==============================")

# centrowanie względem 0.5

W_emg_c = W_emg - 0.5
W_myo_c = W_myo - 0.5

df_model["W_total_B"] = (
    df_model["W_zawod"] +
    W_emg_c +
    W_myo_c
)

# normalizacja do 0–1

W_B = df_model["W_total_B"]
df_model["W_total_B_norm"] = (W_B - W_B.min()) / (W_B.max() - W_B.min())

print("\nStatystyki W_total_B_norm:")
print(df_model["W_total_B_norm"].describe())


df_model["poziom_B"] = df_model["W_total_B_norm"].apply(interpret)

print("\nRozkład poziomów (B):")
print(df_model["poziom_B"].value_counts())


print("\nŚredni poziom w klastrach (B):")
print(df_model.groupby("cluster")["W_total_B_norm"].mean())


# =========================================
# WIZUALIZACJA
# =========================================

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.boxplot(x="cluster", y="W_total_A", data=df_model)
plt.title("STEP 3A")

plt.subplot(1,2,2)
sns.boxplot(x="cluster", y="W_total_B_norm", data=df_model)
plt.title("STEP 3B")

plt.tight_layout()
plt.show()

# =========================================
# STEP 3.5 — WALIDACJA INTEGRACJI
# =========================================

print("\n=== WALIDACJA INTEGRACJI (Spearman) ===")

# lista wskaźników
cols = ["W_percentile", "W_zawod", "W_emg", "W_myo", "W_total_A", "W_total_B_norm"]

corr = pd.DataFrame(index=cols, columns=cols)

for i in cols:
    for j in cols:
        rho, _ = spearmanr(df_model[i], df_model[j], nan_policy="omit")
        corr.loc[i, j] = rho

print("\nMacierz korelacji:")
print(corr)

# heatmapa (opcjonalnie — polecam!)
sns.heatmap(corr.astype(float), annot=True, cmap="coolwarm")
plt.title("Korelacje wskaźników")
plt.tight_layout()
plt.show()