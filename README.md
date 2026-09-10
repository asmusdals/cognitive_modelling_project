# Cognitive Modelling Project

Gruppeprojekt til DTU-kurset **02458 Artificial Intelligence, Human Cognition and Cognitive Modeling**. Projektet undersøger vurdering og lineær encoding af ansigtstræk. I Experiment 1 vurderer deltagere ansigters attraktivitet på en skala fra 1 til 5.

## Datasættet

Projektet bruger den aligned og beskårne version af UTKFace. Filnavnene følger formatet:

```text
[age]_[gender]_[race]_[date-and-time].jpg.chip.jpg
```

Det aktuelle udvalg ligger i `data/raw/2627yearmale` og indeholder 328 manuelt udvalgte billeder:

- 149 billeder med alder 26
- 71 billeder med alder 27
- 108 billeder med alder 28
- gender-kode `0` og race-kode `0`

Mappen hedder af historiske grunde `2627yearmale`, selvom udvalget også indeholder 28-årige.

Hvert billede vises to gange i Experiment 1. En fuld gennemførsel består derfor af 656 trials.

## Repositoriets struktur

```text
.
├── data/
│   ├── raw/
│   │   ├── 2627yearmale/       # De udvalgte originale farvebilleder
│   │   └── utkcropped/         # Hele UTKFace-datasættet, ikke inkluderet i Git
│   ├── processed/
│   │   └── 2627yearmale/       # De 328 processerede gråtonebilleder
│   ├── experiment1/
│   │   └── 2627yearmale/       # Deltagernes CSV-filer, ikke inkluderet i Git
│   ├── rating_analysis/
│   │   └── 2627yearmale/       # Histogrammer og eventuelle normaliserede CSV-filer
│   ├── pca_analysis/
│   │   └── 2627yearmale/       # PC-visualiseringer, forklaret varians og PC-scores
│   ├── regression_model/
│   │   └── 2627yearmale/       # Den lineære encoding-model og dens figurer
│   └── UTKFace_Info.xlsx       # Oversigt over UTKFace
├── week1/
│   ├── preprocess.py           # Konverterer udvalgte billeder til gråtoner
│   ├── experiment1.py          # Starter rating-eksperimentet
│   ├── analyze_ratings.py      # Analyserer ratings og laver histogrammer
│   ├── pca_analysis.py         # PCA, PC-visualiseringer og forklaret varians
│   ├── regression_model.py     # Forward selection og den lineære encoding-model
│   ├── task3_report.md         # Rapportklar besvarelse af opgave 3
│   └── sec251.md               # Projektbeskrivelsen
├── requirements.txt            # Python-afhængigheder
└── README.md
```

Mapperne `data/raw/2627yearmale` og `data/processed/2627yearmale` er inkluderet i GitHub-repositoriet. Det fulde UTKFace-datasæt og deltagernes CSV-filer er ikke inkluderet.

## Kloning af projektet

```bash
git clone https://github.com/asmusdals/cognitive_modelling_project.git
cd cognitive_modelling_project
```

Projektet skal køres i et lokalt Python-miljø. Hver bruger opretter sit eget miljø; `.venv`-mappen deles ikke via Git.

## Mulighed 1: almindeligt Python `venv`

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

På Windows aktiveres miljøet i stedet med:

```powershell
.venv\Scripts\Activate.ps1
```

## Mulighed 2: `uv`

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Mulighed 3: Conda

```bash
conda create -n cognitive-modelling python=3.11 pillow matplotlib tk
conda activate cognitive-modelling
```

Tkinter bruges til eksperimentets grafiske vindue. Installationen kan testes med:

```bash
python -m tkinter
```

Hvis et lille vindue åbner, er Tkinter klar.

## Kørsel af scripts

Kør kommandoerne fra repositoriets rodmappe. Aktivér først det valgte Python-miljø.

### 1. Preprocess billederne

De processerede billeder er allerede inkluderet i repositoriet, så dette trin er ikke nødvendigt efter en almindelig kloning. Kør det igen, hvis udvalget i `data/raw/2627yearmale` ændres:

```bash
python week1/preprocess.py
```

Scriptet:

1. finder alle `.jpg`-filer i `data/raw/2627yearmale`;
2. konverterer billederne til gråtoner;
3. gemmer dem i `data/processed/2627yearmale` med deres oprindelige filnavne.

### 2. Start Experiment 1

```bash
python week1/experiment1.py
```

Programmet beder først om deltagerens studienummer. Derefter:

- tryk `MELLEMRUM` for at starte;
- brug tasterne `1`, `2`, `3`, `4` og `5` til at rate attraktivitet;
- tryk `Escape` for at afbryde forsøget.

Billederne præsenteres to gange hver i tilfældig rækkefølge. Programmet gemmer efter hver rating, så en delvis CSV bevares, hvis forsøget afbrydes.

Resultatet gemmes som:

```text
data/experiment1/2627yearmale/[studienummer].csv
```

Programmet overskriver ikke en eksisterende deltagerfil. Hvis den samme deltager skal begynde helt forfra, skal den eksisterende CSV først omdøbes eller flyttes.

### 3. Analysér ratings

Når en eller flere deltagere har gennemført eksperimentet, køres:

```bash
python week1/analyze_ratings.py
```

Scriptet læser alle fuldførte CSV-filer i `data/experiment1/2627yearmale` og:

- laver et histogram for hver deltager;
- kontrollerer, hvilke værdier på skalaen 1–5 deltageren har brugt;
- min-max-normaliserer ratings til 1–5, hvis hele skalaen ikke er blevet brugt;
- springer ufuldstændige deltagerfiler over.

Resultaterne gemmes i:

```text
data/rating_analysis/2627yearmale/
```

### 4. Kør PCA-analysen

```bash
python week1/pca_analysis.py
```

Scriptet centrerer billederne uden at standardisere pixelværdierne, beregner PCA,
visualiserer de første fem PC'er, laver et plot af forklaret varians for alle PC'er
og gemmer PC-scores til den efterfølgende encoding-model. Resultaterne gemmes i
`data/pca_analysis/2627yearmale/`.

### 5. Byg den lineære encoding-model

```bash
python week1/regression_model.py
```

Scriptet bruger deltagernes (eventuelt normaliserede) ratings som afhængig variabel
og PC-scorerne fra opgave 3 som kandidat-predictors. De relevante PC'er vælges med
forward selection, hvor hver kandidatmodel evalueres med 10-fold krydsvalidering i
stedet for goodness-of-fit. Modellen fittes til sidst på alle data med kun de valgte
PC'er. Resultaterne gemmes i `data/regression_model/2627yearmale/`:

- `forward_selection.png` — krydsvalideringsfejlen som funktion af antal PC'er;
- `selected_pcs.png` — de valgte PC'er visualiseret som i opgave 3;
- `predicted_vs_observed.png` — modellens forudsigelser mod de faktiske ratings;
- `predicted_ratings.csv` — observeret og forudsagt rating for hvert billede;
- `regression_model.npz` — vægte, intercept og vægtvektoren i pixelrum til opgave 5.

## Typisk arbejdsgang

```bash
source .venv/bin/activate
python week1/experiment1.py
python week1/analyze_ratings.py
deactivate
```

Preprocessing skal kun køres først, hvis billedudvalget er blevet ændret.

## Det lokale `.venv`-miljø

`.venv` indeholder en lokal Python-installation og projektets installerede biblioteker. Miljøet er maskin- og systemspecifikt og skal derfor ikke pushes til GitHub. Det kan altid genskabes fra `requirements.txt`.

Hvis `python` ikke kan finde Pillow eller Matplotlib, er den mest sandsynlige årsag, at miljøet ikke er aktiveret. Aktivér det med:

```bash
source .venv/bin/activate
```
