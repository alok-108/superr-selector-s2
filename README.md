# SupeRR Selector Season 2 — Win-Probability Leverage & Context-Adjusted Clutch Reliability

Quantitative scouting framework & IPL 2027 Goa Mini-Auction strategy built for **Rajasthan Royals (SupeRR Selector Season 2, Round 1)**.

> **Competition Compliance Note:** In accordance with Rajasthan Royals & Unstop Season 2 competition rules, the proprietary raw `data_pack/` CSV files are **excluded** via `.gitignore` and are never published or shared in this repository. Only modeling code, reproducible analytical scripts, aggregated derived summaries, visualizations, and the final submission deliverables are tracked.

---

## Methodology Highlights
- **Win-Probability Leverage Pressure Index (`PI ∈ [0, 100]`):** Monotonic `HistGradientBoosting` resource model (2026 holdout MAE = `18.54` runs, `R² = 0.819`) paired with calibrated 5-fold match-grouped Out-of-Fold Logistic Regression (`AUC = 0.901`, Brier = `0.128` in Innings 2) and counterfactual delivery transitions.
- **Two-Stage Context + Player-vs-Player Ridge RAPM (`α = 1200`):** Controls for competition tier (`IPL` +15.7% era inflation vs Full-Member T20Is), venue leave-one-match-out environment par, pace/spin, phase, wickets remaining, exact `batting_order`, and joint out-of-fold opponent quality.
- **Empirical-Bayes Shrinkage (`k = 500`) & Regression-Adjusted Clutch Residual:** Blends top-quartile high-pressure impact (`50%`) with continuous `(PI/100)²` weighted impact (`50%`) across **200,999 deliveries** (**182,038** Top-Flight T20 balls across **784** matches; `n = 73` qualified batters, `n = 79` qualified bowlers).
- **Out-of-Time Scientific Validation (`2022–24 → 2025–26`):** Proves naive `HP − LP` deltas fail out-of-sample (`r = −0.019` batters, `r = −0.055` bowlers), while overall class + Empirical-Bayes RAPM recovers genuine predictive signal.

---

## Final Ordered Top-10 Rankings (`PRS`, Top-Flight 2022–2026)

| Rank | Batters Under Pressure (`PRS`) | Bowlers Under Pressure (`PRS`) |
| :---: | :--- | :--- |
| **#1** | **Suryakumar Yadav** (`85.9`) | **Krunal Pandya** (`79.9`) |
| **#2** | **Tim David** (`83.2`) | **Axar Patel** (`78.4`) |
| **#3** | **Namboori Tilak Varma** (`77.5`) | **Bhuvneshwar Kumar** (`73.4`) |
| **#4** | **Shreyas Iyer** (`73.9`) | **Sunil Narine** (`73.0`) |
| **#5** | **Shashank Singh** (`69.0`) | **Josh Hazlewood** (`72.2`) |
| **#6** | **David Miller** (`67.2`) | **Jasprit Bumrah** (`71.7`) |
| **#7** | **Dinesh Karthik** (`63.1`) | **Nathan Ellis** (`71.7`) |
| **#8** | **Heinrich Klaasen** (`62.1`) | **Rashid Khan** (`67.6`) |
| **#9** | **Dasun Shanaka** (`62.1`) | **Kuldeep Yadav** (`67.3`) |
| **#10** | **Ishan Kishan** (`61.3`) | **Wanindu Hasaranga** (`67.1`) |

---

## Key Repository Layout
- `scripts/` — End-to-end modeling pipeline (`15_master_rapm_pipeline.py`, `17_scientific_deep_dive.py`, `18_dual_allrounder_audit.py`, `12_generate_charts.py`, `13_build_report_and_dashboard.py`)
- `outputs/` — Derived player leaderboards, temporal OOS validation, Jaipur/Guwahati venue splits, LHB/RHB splits, Nov 2026 trade audits, and Moneyball radars
- `report/` — Compiled 11-page PDF report (`SupeRR_Selector_S2_Methodology_Report.pdf`), standalone interactive dashboard HTML (`SupeRR_Selector_S2_Dashboard.html`), and 10 high-DPI figures (`report/figures/`)
- `docs/index.html` — Self-contained interactive War-Room Dashboard served for GitHub Pages (zero raw dataset files exposed)
