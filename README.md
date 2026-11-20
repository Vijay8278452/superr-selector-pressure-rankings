# SupeRR Selector Season 2: Identifying the Most Reliable Players Under Pressure

This repository contains the complete analytical pipeline and reproducible code base for Round 1 of SupeRR Selector Season 2. The objective is to evaluate how batters and bowlers perform in high pressure match situations relative to their usual standards, identifying the players who deliver most reliably when it matters most.

---

## Headline Findings: Official Final Rankings

The system rates players using a regularised multi-objective model combining Pressure Rating (P), Clutch Gap (G) and Consistency (C), validated through 400 bootstrap consensus iterations.

### Top 10 Batters Under Pressure
1. **David Miller** (ID: 1077, Gujarat Titans) — Consensus Score: 1.594, Top 10 Prob: 81.8%
2. **Tim David** (ID: 6396, Mumbai Indians) — Consensus Score: 0.987, Top 10 Prob: 72.0%
3. **Phil Salt** (ID: 5553, Royal Challengers Bengaluru) — Consensus Score: 1.367, Top 10 Prob: 63.0%
4. **Travis Head** (ID: 2367, Sunrisers Hyderabad) — Consensus Score: 0.150, Top 10 Prob: 56.2%
5. **Shai Hope** (ID: 4369, Delhi Capitals) — Consensus Score: 0.814, Top 10 Prob: 51.7%
6. **Virat Kohli** (ID: 47, Royal Challengers Bengaluru) — Consensus Score: 0.587, Top 10 Prob: 39.8%
7. **Shreyas Iyer** (ID: 3647, Kolkata Knight Riders) — Consensus Score: 0.878, Top 10 Prob: 41.2%
8. **Namboori Tilak Varma** (ID: 7897, Mumbai Indians) — Consensus Score: 0.092, Top 10 Prob: 32.8%
9. **Prabhsimran Singh** (ID: 7216, Punjab Kings) — Consensus Score: 0.547, Top 10 Prob: 38.0%
10. **Rinku Singh** (ID: 4708, Kolkata Knight Riders) — Consensus Score: 0.422, Top 10 Prob: 30.2%

### Top 10 Bowlers Under Pressure
1. **Mohsin Khan** (ID: 6669, Lucknow Super Giants) — Consensus Score: 0.639, Top 10 Prob: 67.5%
2. **Mohammed Shami** (ID: 1566, Gujarat Titans) — Consensus Score: 1.161, Top 10 Prob: 61.5%
3. **Wanindu Hasaranga** (ID: 4900, Royal Challengers Bengaluru) — Consensus Score: 1.724, Top 10 Prob: 47.2%
4. **Tim Southee** (ID: 278, Kolkata Knight Riders) — Consensus Score: 1.227, Top 10 Prob: 44.2%
5. **Bhuvneshwar Kumar** (ID: 360, Sunrisers Hyderabad) — Consensus Score: 0.987, Top 10 Prob: 51.2%
6. **Mustafizur Rahman** (ID: 4373, Delhi Capitals) — Consensus Score: 0.987, Top 10 Prob: 35.0%
7. **Axar Patel** (ID: 2876, Delhi Capitals) — Consensus Score: 0.502, Top 10 Prob: 44.2%
8. **Tushar Deshpande** (ID: 6806, Chennai Super Kings) — Consensus Score: 1.598, Top 10 Prob: 30.5%
9. **Matt Henry** (ID: 3174, Chennai Super Kings) — Consensus Score: 0.973, Top 10 Prob: 34.8%
10. **Richard Ngarava** (ID: 4874, Zimbabwe) — Consensus Score: 0.611, Top 10 Prob: 28.7%

---

## How the Method Works

The analytical framework operates through five sequential layers:

1. **Leverage Index (LI) Pressure Definition**:
   We train calibrated out-of-fold win probability models for both innings. Before every ball, we calculate the expected swing in win probability across all possible outcomes. Dividing this swing by the sample average yields the Leverage Index. Deliveries in the top 25 percent of Leverage Index represent the high pressure regime.

2. **Performance Measurement Against Context**:
   We evaluate every delivery by comparing its actual win probability change against the expectation of an average player in that exact match situation (Win Probability Added Above Average, WPAA). Dividing by the Leverage Index produces the neutral value per ball (V = WPAA / LI), separating execution skill from situational frequency.

3. **Joint Regularised Opposition Adjustment**:
   We fit simultaneous Ridge regression models on deliveries to evaluate batters and bowlers together. This adjusts every batter for the quality of bowlers faced and every bowler for the batters confronted. Fitting separately on pressure balls and non-pressure balls yields a Pressure Rating (P), a Normal Rating (N) and a Clutch Gap (G = P - N).

4. **Consistency Scoring Across Matches**:
   We measure stability across distinct match appearances. Each player receives a shrunk hit rate (proportion of pressure appearances with positive impact) and a shrunk variance metric, combined into a Consistency score (C).

5. **Consensus Ranking**:
   Eligible players are scored using S = 0.50 z(P) + 0.20 z(G) + 0.30 z(C). We re-run this entire ranking under 400 bootstrap perturbations that resample matches, perturb pressure thresholds (top 15 to 30 percent) and vary component weights. The final order is the consensus rank across all perturbations.

---

## Repository Structure

```
superr-round1-pressure-rankings/
├── README.md                          # Project documentation and summary
├── requirements.txt                   # Pinned library dependencies
├── .gitignore                         # Version control exclusions
├── data/
│   └── README.md                      # Instructions on placing raw CSV files
├── src/
│   ├── __init__.py
│   ├── config.py                      # Global settings, thresholds and random seed 42
│   ├── data_loading.py                # Raw CSV data loading
│   ├── cleaning.py                    # Data audit, exclusions and sanitisation
│   ├── features.py                    # Pre-ball match state, par and Elo features
│   ├── win_probability.py             # Calibrated out-of-fold win probability models
│   ├── outcome_model.py               # Out-of-fold multiclass outcome model
│   ├── pressure.py                    # Leverage index, tiers and rule-based cross check
│   ├── performance.py                 # WPAA and neutral value per ball
│   ├── ratings.py                     # Simultaneous Ridge regression and bootstrap intervals
│   ├── consistency.py                 # Shrunk match hit rates and reliability checks
│   ├── ranking.py                     # Composite score and 400 bootstrap consensus replicates
│   ├── robustness.py                  # Sensitivity analysis and run-based checks
│   ├── figures.py                     # High resolution figure generation (F1 to F10)
│   └── tables.py                      # Publication table exports (T1 to T8)
├── run_all.py                         # Single command pipeline runner
├── outputs/
│   ├── tables/                        # Output CSV and LaTeX tables (T1 to T8)
│   ├── figures/                       # High resolution publication figures (F1 to F10)
│   ├── logs/                          # Execution logs and audit trail
│   └── final_rankings.csv             # Official master player ranking CSV
├── dashboard/
│   └── index.html                     # Self-contained interactive web dashboard
├── presentation/
│   └── RR_Pressure_Analysis_Presentation.pptx # Executive scouting presentation (18 slides)
└── report/
    ├── main.tex                       # LaTeX source of 18-page methodology paper
    ├── main.pdf                       # Compiled 18-page methodology research paper
    ├── RR_Pressure_Analysis_Report.tex# LaTeX source of 22-page extended report
    ├── RR_Pressure_Analysis_Report.pdf# Compiled 22-page report with RR squad fit
    └── RR_Pressure_Analysis_Report.docx# Executive Word report (30+ pages)
│   ├── figures/                       # Output PNG and PDF figures (F1 to F10)
│   ├── logs/                          # Comprehensive data audit logs
│   ├── models/                        # Serialised model artifacts
│   └── final_rankings.csv             # Dropdown submission rankings
└── report/
    ├── main.tex                       # Complete LaTeX report source
    ├── Makefile                       # Tectonic / pdflatex build script
    ├── compile.sh                     # Bash compilation wrapper
    └── main.pdf                       # Compiled methodology report PDF
```

---

## Installation and Execution

### 1. Requirements
Python 3.10 or newer is required. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

### 2. Data Setup
Ensure the unzipped CSV data files from the Data Pack (`ball_by_ball.csv`, `matches.csv`, etc.) are placed in the `data/` directory or in the repository parent directory.

### 3. Running the Pipeline
Execute the complete workflow from the repository root:

```bash
python run_all.py
```

The script runs all auditing, modelling, bootstrap perturbations, table exports and figure generations. The entire process completes in under 5 minutes on a standard laptop.

### 4. Compiling the Report
Compile the LaTeX methodology document to PDF using tectonic or pdflatex:

```bash
cd report
bash compile.sh
```
