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
│   └── UTKFace_Info.xlsx       # Oversigt over UTKFace
├── week1/
│   ├── preprocess.py           # Konverterer udvalgte billeder til gråtoner
│   ├── experiment1.py          # Starter rating-eksperimentet
│   ├── analyze_ratings.py      # Analyserer ratings og laver histogrammer
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
