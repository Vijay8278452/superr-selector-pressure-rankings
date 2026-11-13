# SupeRR Selector Season 2: Identifying the Most Reliable Players Under Pressure

This repository contains the complete analytical pipeline and reproducible code base for Round 1 of SupeRR Selector Season 2. The objective is to evaluate how batters and bowlers perform in high pressure match situations relative to their usual standards, identifying the players who deliver most reliably when it matters most.

---

## Headline Findings: Official Final Rankings

The system rates players using a regularised multi-objective model combining Pressure Rating (P), Clutch Gap (G) and Consistency (C), validated through 400 bootstrap consensus iterations.

### Top 10 Batters Under Pressure
1. **David Miller** (ID: 3042, Gujarat Titans)
2. **Phil Salt** (ID: 6867, Kolkata Knight Riders)
3. **Tim David** (ID: 6868, Mumbai Indians)
4. **Shreyas Iyer** (ID: 2901, Kolkata Knight Riders)
5. **Abhishek Sharma** (ID: 5543, Sunrisers Hyderabad)
6. **Shai Hope** (ID: 3374, Delhi Capitals)
7. **Rishabh Pant** (ID: 3511, Delhi Capitals)
8. **Sai Sudharsan** (ID: 8645, Gujarat Titans)
9. **Nicholas Pooran** (ID: 3303, Lucknow Super Giants)
10. **Virat Kohli** (ID: 47, Royal Challengers Bengaluru)

### Top 10 Bowlers Under Pressure
1. **Wanindu Hasaranga** (ID: 5493, Sunrisers Hyderabad)
2. **Tushar Deshpande** (ID: 5472, Chennai Super Kings)
3. **Kagiso Rabada** (ID: 3334, Punjab Kings)
4. **Tim Southee** (ID: 307, Kolkata Knight Riders)
5. **Mohammed Shami** (ID: 2899, Gujarat Titans)
6. **Kuldeep Yadav** (ID: 3105, Delhi Capitals)
7. **Mustafizur Rahman** (ID: 3350, Chennai Super Kings)
8. **Bhuvneshwar Kumar** (ID: 2888, Sunrisers Hyderabad)
9. **Matt Henry** (ID: 2977, Lucknow Super Giants)
10. **Ravi Bishnoi** (ID: 6475, Lucknow Super Giants)

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
