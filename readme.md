# Wielowymiarowa analiza obciążenia pracą

## Opis projektu

Projekt służy do analizy obciążenia pracą na podstawie trzech źródeł danych:

* danych ankietowych (subiektywnych),
* danych kinematycznych (MyoMotion),
* danych elektromiograficznych (EMG).

Analiza opiera się na rozdzieleniu dwóch kluczowych aspektów:

* **struktury obciążenia** (profil odpowiedzi),
* **poziomu obciążenia** (ranking względem populacji).

Dane pomiarowe (MyoMotion, EMG) pełnią rolę **obiektywizującą** względem danych ankietowych.

---

## Data processing (przygotowanie danych)
Ankieta
-- normalizacja tekstu
-- mapowanie odpowiedzi do wartości liczbowych
-- obsługa braków danych
-- normalizacja w obrębie respondenta
MyoMotion
-- parsowanie plików .slk
-- czyszczenie danych
-- synchronizacja czasu
-- selekcja zmiennych kinematycznych
EMG
-- parsowanie .slk
-- wybór kanałów EMG
-- czyszczenie sygnału
-- przygotowanie do ekstrakcji cech

---

# Analiza danych — etap testowy

Obecny zestaw skryptów obejmuje:

* analizę struktury odpowiedzi ankietowych,
* konstrukcję testowego wskaźnika obciążenia (ankieta),
* przetwarzanie i analizę danych MyoMotion,
* wyznaczenie wskaźników EMG,
* audyt techniczny danych pomiarowych.

Na tym etapie **nie jest jeszcze budowany finalny zintegrowany wskaźnik globalny**.

---

# Zawartość

## 1. `step1_load_clean.py`

**Rola:** analiza strukturalna odpowiedzi ankietowych.

Skrypt realizuje:

* czyszczenie danych,
* mapowanie odpowiedzi na wartości liczbowe,
* normalizację **w obrębie respondenta**,
* PCA eksploracyjne,
* klasteryzację respondentów.

### Znaczenie metodologiczne

Analiza dotyczy:

* struktury odpowiedzi,
* profilu obciążenia,
* współzależności zmiennych.

Nie opisuje poziomu obciążenia.

---

## 2. `podglad_slk.py`

**Rola:** szybki podgląd pliku `.slk`.

Skrypt:

* wyświetla pierwsze linie pliku,
* umożliwia ręczną kontrolę struktury danych.

### Znaczenie

Narzędzie pomocnicze do diagnostyki formatu danych.

---

## 3. `step2_wskaznik_obciazenia_test.py`

**Rola:** konstrukcja testowego wskaźnika obciążenia (ankieta).

Skrypt realizuje **dwa odrębne etapy PCA**:

### 🔹 ETAP A — PCA poziomu

* standaryzacja między respondentami,
* ujednolicenie kierunku zmiennych („więcej = gorzej”),
* PCA → identyfikacja głównego wymiaru zmienności.

### 🔹 ETAP B — konstrukcja wskaźnika syntetycznego

* wybór pierwszego komponentu (PC1),
* korekta znaku komponentu,
* przekształcenie do skali percentylowej.

### Wynik

* wskaźnik **relatywnego poziomu obciążenia**,
* interpretacja rankingowa (pozycja w populacji).

### Znaczenie metodologiczne

* PCA pełni funkcję **narzędzia agregacji**,
* wskaźnik ma charakter:

  * testowy,
  * oparty wyłącznie na ankiecie,
  * nie jest wskaźnikiem końcowym.

---

## 4. `oczysc_myo_motion.py`

**Rola:** przygotowanie danych MyoMotion.

Skrypt realizuje:

* parsowanie pliku `.slk`,
* odtworzenie struktury danych,
* czyszczenie wartości,
* usunięcie rekordów technicznych,
* uporządkowanie osi czasu.

---

## 5. `analiza_myo_motion.py`

**Rola:** analiza eksploracyjna MyoMotion.

Skrypt realizuje:

* statystyki opisowe,
* analizę ekspozycji powyżej progów,
* analizę regionalną (tułów, barki itd.).

### Wynik

* raporty CSV:

  * summary,
  * threshold,
  * regional.

---

## 6. `myo_wskazniki_kluczowe.py`

**Rola:** konstrukcja wskaźników MyoMotion.

Etapy:

1. **Feature extraction**

   * percentyl 95,
   * mediany,
   * % czasu powyżej progów

2. **Agregacja regionalna**

   * TRUNK_LOAD
   * SHOULDER_LOAD

3. **Redukcja**

   * zestaw wskaźników opisujących obciążenie posturalne

---

## 7. `audyt_emg_myo_bez_xlrd.py`

**Rola:** audyt techniczny danych EMG i MyoMotion.

Skrypt realizuje:

* porównanie plików,
* analizę osi czasu,
* sprawdzenie częstotliwości,
* identyfikację braków danych,
* analizę metadanych.

### Znaczenie

Etap kontrolny — zapewnia poprawność danych przed analizą.

---

## 8. `emg_wskazniki_kluczowe.py`

**Rola:** konstrukcja wskaźników EMG.

Etapy:

1. **Feature extraction**

   * mean abs,
   * median abs,
   * p95,
   * RMS

2. **Ekspozycja czasowa**

   * % czasu powyżej progów (20, 50, 100 µV)

3. **Agregacja globalna**

   * średnia między kanałami:

     * EMG_GLOBAL_MEAN_ABS
     * EMG_GLOBAL_P95_ABS
     * EMG_GLOBAL_RMS

### Znaczenie metodologiczne

Redukcja sygnału EMG do syntetycznych wskaźników obciążenia mięśniowego.

---

# Logika całego etapu

Analiza przebiega w trzech filarach:

### 🔵 Ankieta

* struktura (PCA wewnętrzne)
* poziom (PCA międzyosobowe)

### 🔵 MyoMotion

* sygnał → cechy → wskaźniki

### 🔵 EMG

* sygnał → cechy → agregaty globalne

---

# Dane wejściowe

### Ankieta

* `dane.xlsx`

### MyoMotion

* `.slk`
* `*_clean.csv`

### EMG

* `.slk`

---

# Kolejność uruchamiania

### Ankieta

```bash
python step1_load_clean.py
python step2_wskaznik_obciazenia_test.py
```

### MyoMotion

```bash
python oczysc_myo_motion.py "plik.slk"
python analiza_myo_motion.py "plik_clean.csv"
python myo_wskazniki_kluczowe.py "plik_clean.csv"
```

### EMG

```bash
python audyt_emg_myo_bez_xlrd.py "emg.slk" "myo.slk"
python emg_wskazniki_kluczowe.py "emg.slk"
```

---

# Uwagi końcowe

* W analizie świadomie rozdzielono:

  * strukturę odpowiedzi,
  * poziom obciążenia,
  * dane pomiarowe.

* PCA jest wykorzystywane w dwóch różnych rolach:

  * eksploracyjnej,
  * agregacyjnej.

* Obecny etap stanowi przygotowanie do integracji wszystkich filarów.

---

# Status projektu

Etap pośredni:

* gotowe wskaźniki cząstkowe,
* brak finalnej integracji.

---
