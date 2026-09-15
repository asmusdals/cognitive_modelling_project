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
│   ├── synthetic_faces/
│   │   └── 2627yearmale/       # Syntetiske ansigter og stimuli til Experiment 2
│   ├── experiment2/
│   │   └── 2627yearmale/       # Deltagernes CSV-filer fra Experiment 2, ikke inkluderet i Git
│   ├── experiment2_analysis/
│   │   └── 2627yearmale/       # Box plots og Spearmans rho for Experiment 2
│   ├── experiment3/
│   │   └── 2627yearmale/       # Deltagernes CSV-filer fra Experiment 3, ikke inkluderet i Git
│   ├── experiment3_analysis/
│   │   └── 2627yearmale/       # Efterbillede-effekten opdelt på adaptationsbetingelse
│   └── UTKFace_Info.xlsx       # Oversigt over UTKFace
├── week1/
│   ├── preprocess.py           # Konverterer udvalgte billeder til gråtoner
│   ├── experiment1.py          # Starter rating-eksperimentet
│   ├── analyze_ratings.py      # Analyserer ratings og laver histogrammer
│   ├── pca_analysis.py         # PCA, PC-visualiseringer og forklaret varians
│   ├── regression_model.py     # Forward selection og den lineære encoding-model
│   ├── synthetic_faces.py      # Syntetiske ansigter og kontrol af ratingintervallet
│   ├── experiment2.py          # Starter valideringseksperimentet med de syntetiske ansigter
│   ├── analyze_experiment2.py  # Box plots og Spearmans rho for Experiment 2
│   ├── experiment3.py          # Starter adaptationseksperimentet
│   ├── analyze_experiment3.py  # Efterbillede-effekten for de tre testansigter
│   ├── task3_report.md         # Rapportklar besvarelse af opgave 3
│   ├── task5_6_report.md       # Rapportklar besvarelse af opgave 5 og 6
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
conda create -n cognitive-modelling python=3.11 pillow matplotlib numpy scipy tk
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

### 6. Generér de syntetiske ansigter

```bash
python week1/synthetic_faces.py
```

Scriptet vender encoding-modellen om og konstruerer for hver ønsket rating det
entydige billede, hvis lavdimensionelle repræsentation er parallel med
vægtvektoren. Derefter kontrolleres det, om de ønskede ratings overhovedet ligger
inden for det interval af ratings, som modellen forudsiger for de trænede
billeder. Hvis ikke, genereres et ekstra sæt ansigter inde i det interval.
Resultaterne gemmes i `data/synthetic_faces/2627yearmale/`:

- `synthetic_faces_requested.png` — de 11 ønskede ratings 0,5–5,5 (opgave 5);
- `predicted_rating_range.png` — de forudsagte ratings mod de ønskede (opgave 6);
- `synthetic_faces_in_range.png` — 11 ansigter inden for det forudsagte interval;
- `synthetic_faces_comparison.png` — de to kontinua over for hinanden;
- `stimuli_requested/` og `stimuli_in_range/` — ét PNG pr. ansigt, klar til Experiment 2;
- `experiment3_stimuli.png` — adaptere og testansigter til opgave 8;
- `stimuli_experiment3_adapt/` og `stimuli_experiment3_test/` — stimuli til Experiment 3;
- `synthetic_faces.csv` — målrating, alpha og diagnostik for hvert billede;
- `synthetic_faces.npz` — billederne som arrays.

Til Experiment 3 bruges continuummets to endepunkter som adaptere, mens de tre
testansigter ligger tæt på det neutrale ansigt. Det neutrale ansigt er
gennemsnitsansigtet: ved ratingen `delta` er `alpha` nul, så der lægges ingenting
til gennemsnittet.

Afstanden mellem testansigterne styres af `TEST_STIMULUS_OFFSET` og er et
kompromis. Et ratingtrin svarer kun til omkring otte gråtoneniveauer, så ligger
testansigterne for tæt, kan deltageren slet ikke se forskel på dem; ligger de for
langt fra hinanden, er de ikke længere tæt på det neutrale ansigt. Scriptet
printer, hvor mange gråtoneniveauer der faktisk skiller testansigterne, så
værdien kan justeres på et oplyst grundlag.

Stimuli til Experiment 2 tages fra `stimuli_in_range/`, da flere af de ønskede
ratings i opgave 5 ligger uden for det interval, modellen dækker.

### 7. Start Experiment 2

```bash
python week1/experiment2.py
```

Programmet fungerer som Experiment 1 og bruger den samme 1–5-skala, men viser de
11 syntetiske ansigter fra `stimuli_in_range/` i stedet for rigtige fotos. Som
opgave 7 kræver, vises hvert ansigt mindst 10 gange i tilfældig rækkefølge, i alt
110 trials. Antallet kan ændres med `REPEATS` i toppen af scriptet, og
`STIMULUS_SET` kan sættes til `"requested"`, hvis sættet fra opgave 5 skal bruges.

Resultatet gemmes som:

```text
data/experiment2/2627yearmale/[studienummer].csv
```

Filen har én række pr. ansigt med filnavn, den forudsagte rating billedet er
genereret til, og én kolonne pr. gentagelse. Som i Experiment 1 gemmes der efter
hver rating, og en eksisterende deltagerfil bliver ikke overskrevet.

### 8. Analysér Experiment 2

```bash
python week1/analyze_experiment2.py
```

Scriptet laver det box plot, opgave 7 beder om — ratings som funktion af den
forudsagte rating, for hver enkelt deltager — og beregner Spearmans
rangkorrelationskoefficient `rho`. Rho bruges frem for Pearsons r, fordi ratings
er ordinale: en deltager, der rangordner ansigterne rigtigt, men bruger skalaen
ujævnt, tæller stadig som enig med modellen. Af samme grund bruges de rå ratings
hele vejen igennem; rho afhænger kun af rangene og ville være uændret af
min-max-normaliseringen fra opgave 2. Resultaterne gemmes i
`data/experiment2_analysis/2627yearmale/`:

- `experiment2_boxplots.png` — ét box plot pr. deltager med deres rho;
- `experiment2_group_boxplot.png` — det samme med alle deltagere samlet;
- `experiment2_stimulus_means.csv` — gennemsnitsrating pr. ansigt pr. deltager;
- `experiment2_summary.csv` — rho, p-værdi og split-half-reliabilitet.

Rho forventes at ligge tæt på 1. Hvor tæt den realistisk kan komme, begrænses af,
hvor konsistent deltageren vurderer det samme ansigt to gange, så
split-half-reliabiliteten rapporteres ved siden af. Den korrelerer deltagerens
ulige og lige gentagelser af hvert ansigt: en deltager, der ikke kan gentage sine
egne vurderinger, kan heller ikke stemme overens med modellen.

### 9. Start Experiment 3

```bash
python week1/experiment3.py
```

Adaptationseksperimentet fra opgave 8. Hver trial viser et adapterende ansigt fra
den ene ende af continuummet i 20 sekunder og derefter et af de tre
near-neutrale testansigter i 0,75 sekunder, hvorefter deltageren vurderer
testansigtet på den samme 1–5-skala. Begge endepunkter bruges som adaptere i hver
sin betingelse, hvilket med tre testansigter giver de seks betingelser, opgaven
beder om. Der tegnes et fikseringspunkt midt i alle stimuli, så adapter og
testansigt rammer den samme del af nethinden.

Trials er blokket efter adapter i stedet for at være blandet: adaptation hænger
ved fra trial til trial, og skiftevis adaptation til modsatte endepunkter ville
lade de to betingelser udligne hinanden. Rækkefølgen af de to blokke trækkes
tilfældigt pr. deltager. Tiderne og antallet af gentagelser styres af
`ADAPT_SECONDS`, `TEST_SECONDS` og `REPEATS_PER_CONDITION` i toppen af scriptet;
standardværdierne giver 18 trials på omkring otte minutter.

Det er altså med vilje, at det er det samme adapterende ansigt, der vises i alle
ni trials i en blok. Mellem to trials vises et tomt felt med kun
fikseringspunktet, så deltageren kan se, hvor et trial slutter og det næste
begynder. Det tomme felt ligger før adaptationen og aldrig mellem adapteren og
testansigtet, hvor pausen ville svække efterbillede-effekten.

Resultatet gemmes som:

```text
data/experiment3/2627yearmale/[studienummer].csv
```

Filen har én række pr. trial med blok, adaptationsbetingelse, testansigt og den
givne rating.

### 10. Analysér Experiment 3

```bash
python week1/analyze_experiment3.py
```

Scriptet plotter ratings for de tre testansigter opdelt på de to
adaptationsbetingelser og afgør for hvert testansigt, om forskellen peger den vej,
efterbillede-effekten forudsiger. Forudsigelsen er, at testansigtet opfattes
forskudt væk fra adapteren, så

```text
forskel = middelrating efter adaptation til lav ende − efter adaptation til høj ende
```

er positiv. Forskellen testes med en Mann-Whitney U-test, da ratings er ordinale
og de to betingelser er adskilte sæt af trials. Med få deltagere vejer retningen
af forskellen tungere end p-værdien, så begge dele rapporteres. Resultaterne
gemmes i `data/experiment3_analysis/2627yearmale/`:

- `experiment3_aftereffect.png` — de seks betingelser med alle deltagere samlet;
- `experiment3_per_participant.png` — det samme opdelt pr. deltager;
- `experiment3_summary.csv` — middelværdier, forskel, retning og p-værdi.

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
