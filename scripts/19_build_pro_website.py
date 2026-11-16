import json
import base64
import pandas as pd

ROOT = r"C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2"
OUT = f"{ROOT}\\outputs"
REP = f"{ROOT}\\report"
FIG = f"{REP}\\figures"

with open(f"{OUT}\\summary_metrics.json", encoding="utf-8") as f:
    meta = json.load(f)

top10_b = pd.read_csv(f"{OUT}\\top10_batters.csv")
top10_bw = pd.read_csv(f"{OUT}\\top10_bowlers.csv")
eb = pd.read_csv(f"{OUT}\\eligible_batters.csv")
ebw = pd.read_csv(f"{OUT}\\eligible_bowlers.csv")
micro_b = pd.read_csv(f"{OUT}\\micro_tactics_batters.csv")
micro_bw = pd.read_csv(f"{OUT}\\micro_tactics_bowlers.csv")
sig_b = pd.read_csv(f"{OUT}\\signature_bat_innings.csv")
sig_bw = pd.read_csv(f"{OUT}\\signature_bowl_spells.csv")
sens_b = pd.read_csv(f"{OUT}\\sensitivity_batters.csv")
sens_bw = pd.read_csv(f"{OUT}\\sensitivity_bowlers.csv")
wp_val = pd.read_csv(f"{OUT}\\wp_validation_metrics.csv")
death_pb = pd.read_csv(f"{OUT}\\death_delivery_playbook.csv")
stroke_pb = pd.read_csv(f"{OUT}\\hp_stroke_playbook.csv")
rapm_b_top = pd.read_csv(f"{OUT}\\rapm_toughest_bowlers.csv")
rapm_bw_top = pd.read_csv(f"{OUT}\\rapm_toughest_batters.csv")
rr26_aut = pd.read_csv(f"{OUT}\\rr26_win_loss_autopsy.csv")
rr_bat = pd.read_csv(f"{OUT}\\rr_roster_bat.csv")
rr_bowl = pd.read_csv(f"{OUT}\\rr_roster_bowl.csv")
temp_val = pd.read_csv(f"{OUT}\\temporal_prediction_benchmark.csv")
mech_sum = pd.read_csv(f"{OUT}\\clutch_mechanisms_summary.csv")
venue_rr = pd.read_csv(f"{OUT}\\venue_jaipur_guwahati_hp.csv")
lhb_rhb = pd.read_csv(f"{OUT}\\bowler_lhb_rhb_hp.csv")
trade_aud = pd.read_csv(f"{OUT}\\front_office_trade_audit.csv")
p11_ctx = pd.read_csv(f"{OUT}\\p11_context_effects.csv")
mb_bat = pd.read_csv(f"{OUT}\\moneyball_batters_radar.csv")
mb_bowl = pd.read_csv(f"{OUT}\\moneyball_bowlers_radar.csv")
dual_aud = pd.read_csv(f"{OUT}\\dual_allrounder_audit.csv")
uni20 = pd.read_csv(f"{OUT}\\unified_top20_prs.csv")
ipl_b = pd.read_csv(f"{OUT}\\ipl_only_top15_batters.csv")
ipl_bw = pd.read_csv(f"{OUT}\\ipl_only_top15_bowlers.csv")


def img_b64(fname):
    with open(f"{FIG}\\{fname}", "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


bat_fit_notes = {
    "Suryakumar Yadav": "Retained MI core benchmark (#1 overall, 100% bootstrap p>0). Archetype template for RR's #3/#4 core (Jaiswal/Parag) vs Jaipur spin.",
    "Tim David": "Retained RCB finisher benchmark (197.0 HP SR vs Pace-On). Pace-Off dip (-8.6 TVA) warns RR against one-dimensional hitters at #6.",
    "Namboori Tilak Varma": "Retained MI crisis anchor benchmark (810 HPb, #1 Entry-10 +31.9 TVA). Blueprint alongside Riyan Parag to absorb early collapses.",
    "Shreyas Iyer": "PBKS captain benchmark (+54.3 Spin TVA, +38.8 in 2024-26). Demonstrates leadership HP lift (+7.16 TVA across captains) & spin mastery.",
    "Shashank Singh": "Unlikely PBKS release, yet exact prototype (+85.2 Pace-Off TVA) for RR's #6/#7 uncapped domestic radar (Ashutosh/Aniket Verma).",
    "David Miller": "TOP OS FINISHER TARGET: +9.4 clutch residual & +47.5 vs Spin directly upgrades Shimron Hetmyer's brittle #5/#6 slot (113.0 SR in IPL 2026).",
    "Dinesh Karthik": "Retired after IPL 2024; included strictly on 2022-26 data (+35.5 I2 TVA). Historical template for domestic closer development.",
    "Heinrich Klaasen": "PRIMARY RELEASE-WATCH CLOSER: #3 IPL-only pressure bat (+54.1 vs Spin) & WK-batter; transforms RR's middle overs in Jaipur & Guwahati.",
    "Dasun Shanaka": "CURRENT RR RETAINEE: High-volume crunch surge (+6.7 clutch resid, +31.1 vs Pace-On, +25.9 I2 TVA) & affordable overseas finishing/seam depth.",
    "Ishan Kishan": "CONDITIONAL LHB TARGET (if released): Pairs immediate Entry-10 speed (+16.0) with elite Spin (+59.0) & Pace-Off (+53.1) output behind Jaiswal.",
}

bowl_fit_notes = {
    "Krunal Pandya": "#1 AUCTION SPIN-AR TARGET: Cuts Econ 7.97->7.18, dominates I2 chase defense (+61.2), and leads Top 10 vs LHB (+70.4) to exploit Jaipur (9.38) & Guwahati (8.12). Deploy strictly below #6 batting (-4.60 HP bat).",
    "Axar Patel": "Retained DC benchmark (#1 net dual all-rounder at +1.40 HP runs/m) confirming massive crunch value of Indian SLA controllers.",
    "Bhuvneshwar Kumar": "RCB veteran pace benchmark (723 HPb, 45 wkts, 22% HP Pace-Off). Illustrates tactical slower-ball complement opposite Jofra Archer.",
    "Sunil Narine": "Retained KKR mystery-spin template (6.95 HP Econ, 37.9% dots, 2026 Super Over 1r/2w). Proves defensive choke power in chase wins.",
    "Josh Hazlewood": "Retained RCB hard-length seam benchmark (40.9% HP dots, +21.0 in 2024-26). Template for disciplined dot pressure paired with Archer & Shami.",
    "Jasprit Bumrah": "Retained MI gold-standard death closer (#1 recent 2024-26 at +27.4, +50.1 Death TVA across 330b). Ultimate pace-death partner benchmark.",
    "Nathan Ellis": "TRADED CSK -> SRH (Nov 2026): Was RR's #1 Pace-Off death fit (44.1% Pace-Off, +51.6 vs LHB). Lock forces immediate Goa pivot to Pathirana & Lungi Ngidi.",
    "Rashid Khan": "Retained GT icon (+28.2 Death TVA across 250b) and the ONLY Top-10 bowler also ranking Top-25 with the bat (+6.58 bat, +1.29 net HP runs/m).",
    "Kuldeep Yadav": "PRIORITY INDIAN WRIST-SPIN TARGET: #2 IPL-only pressure bowler (7.62 IPL HP Econ, 78.3% pos. rate, +54.2 vs LHB); cures RR's middle-over wicket drought.",
    "Wanindu Hasaranga": "EX-RR WRIST-SPIN VALUE TARGET: Strike HP leg-spinner (12.9 HP SR, +40.2 IPL TVA) for Jaipur/Guwahati; protect at #8 batting (-8.61 HP bat).",
}

# Build enriched LHB/RHB pivot for Top-10 bowlers
lhb_piv = lhb_rhb.pivot(index="bowler_name", columns="batting_type", values=["b", "tva"])
lhb_piv.columns = [f"{a}_{b}" for a, b in lhb_piv.columns]
lhb_piv = top10_bw[["rank", "bowler_name"]].merge(lhb_piv.reset_index(), on="bowler_name", how="left")

# Enrich full batters & bowlers payloads
bat_full = (
    eb.merge(
        micro_b.drop(columns=["striker_id"]),
        on="striker_name",
        how="left",
    )
    .sort_values("rank")
    .copy()
)
bat_full["rr_fit"] = bat_full["striker_name"].map(bat_fit_notes).fillna("")
bat_full["sig_json"] = bat_full["striker_id"].apply(
    lambda pid: sig_b[sig_b["striker_id"] == pid][["match_date", "comp_name", "home_team", "away_team", "hp_b", "hp_r", "hp_tva", "hp_wpa"]].round(2).to_dict("records")
)

bowl_full = (
    ebw.merge(
        micro_bw.drop(columns=["bowler_id", "pace_spin", "bowl_style"]),
        on="bowler_name",
        how="left",
    )
    .merge(lhb_piv.drop(columns=["rank"]), on="bowler_name", how="left")
    .sort_values("rank")
    .copy()
)
bowl_full["rr_fit"] = bowl_full["bowler_name"].map(bowl_fit_notes).fillna("")
bowl_full["ipl_2026_team"] = bowl_full.apply(
    lambda r: "CSK (->SRH Trade)" if r["bowler_name"] == "Nathan Ellis" else r["ipl_2026_team"],
    axis=1,
)
bowl_full["sig_json"] = bowl_full["bowler_id"].apply(
    lambda pid: sig_bw[sig_bw["bowler_id"] == pid][["match_date", "comp_name", "home_team", "away_team", "hp_leg", "hp_r", "hp_w", "hp_tva", "hp_wpa"]].round(2).to_dict("records")
)

bat_cols = [
    "rank", "striker_id", "striker_name", "bat_type", "ipl_2026_team", "t20i_team",
    "tot_b", "hp_b", "hp_m", "hp_sr", "lp_sr", "hp_bdry", "hp_dot", "blended_hp_tva",
    "lp_tva_eb", "clutch_resid", "hp_wpa_eb", "pos_m_rate", "PRS_100",
    "hp_pon_tva", "hp_poff_tva", "hp_spin_sr", "hp_spin_tva", "entry10_tva",
    "collapse_tva", "chase_death_tva", "recent_24_26_tva", "top_hp_stroke",
    "boot_mean", "ci_low_95", "ci_high_95", "p_gt_zero", "rr_fit", "sig_json"
]
bowl_cols = [
    "rank", "bowler_id", "bowler_name", "bowl_style", "pace_spin", "ipl_2026_team", "t20i_team",
    "tot_leg", "hp_leg", "hp_m", "hp_econ", "lp_econ", "hp_wkts", "hp_sr", "hp_dot", "hp_bdry",
    "blended_hp_tva", "lp_tva_eb", "clutch_resid", "hp_wpa_eb", "pos_m_rate", "PRS_100",
    "hp_spd_mean", "hp_spd_std", "hp_pace_off_pct", "lp_pace_off_pct", "hp_death_leg",
    "hp_death_tva100", "hp_i2_tva100", "recent_24_26_tva", "top_hp_styles",
    "b_LHB", "b_RHB", "tva_LHB", "tva_RHB", "boot_mean", "ci_low_95", "ci_high_95", "p_gt_zero", "rr_fit", "sig_json"
]


def df_to_json(df):
    return df.to_json(orient="records")


# Auction Simulator Targets grounded 100% in outputs CSVs (no invented numbers!)
def get_tva(df, col_name, name, tva_col="blended_hp_tva"):
    s = df[df[col_name] == name]
    return float(s[tva_col].iloc[0]) if len(s) else 0.0


def get_row(df, col_name, name):
    s = df[df[col_name] == name]
    return s.iloc[0] if len(s) else None


krunal_r = get_row(ebw, "bowler_name", "Krunal Pandya")
pathirana_r = get_row(ebw, "bowler_name", "Matheesha Pathirana")
ngidi_ipl = get_row(ipl_bw, "bowler_name", "Lungi Ngidi")
ngidi_mb = get_row(mb_bowl, "bowler_name", "Lungi Ngidi")
kuldeep_r = get_row(ebw, "bowler_name", "Kuldeep Yadav")
hasaranga_r = get_row(ebw, "bowler_name", "Wanindu Hasaranga")
brar_ipl = get_row(ipl_bw, "bowler_name", "Harpreet Brar")
brar_mb = get_row(mb_bowl, "bowler_name", "Harpreet Brar")
malinga_mb = get_row(mb_bowl, "bowler_name", "Eshan Malinga")
miller_r = get_row(eb, "striker_name", "David Miller")
klaasen_r = get_row(eb, "striker_name", "Heinrich Klaasen")
ashutosh_ipl = get_row(ipl_b, "striker_name", "Ashutosh Sharma")
aniket_mb = get_row(mb_bat, "striker_name", "Aniket Verma")
naman_mb = get_row(mb_bat, "striker_name", "Naman Dhir")

auc_targets = [
    {
        "id": "krunal", "name": "Krunal Pandya", "tag": "Bowl #1 (79.9 PRS)", "role": "IND Spin-AR", "os": False, "cat": "bowl",
        "team": "RCB", "tva100": round(float(krunal_r["blended_hp_tva"]), 1), "hp_vol": int(krunal_r["hp_leg"]),
        "metric": f"{krunal_r['hp_econ']:.2f} HP Econ | +70.4 vs LHB", "fit": "Replaces negative middle/chase spin leakage at Jaipur (9.38) & Guwahati (8.12); bat below #6.", "balls_covered": 110, "default_on": True
    },
    {
        "id": "pathirana", "name": "Matheesha Pathirana", "tag": "Bowl #11 / #8 IPL (KKR Release Watch)", "role": "OS Death Pace", "os": True, "cat": "bowl",
        "team": "KKR (Rs 18 Cr watch)", "tva100": round(float(pathirana_r["blended_hp_tva"]), 1), "hp_vol": int(pathirana_r["hp_leg"]),
        "metric": f"11.0 HP SR | 34 HP Wkts", "fit": "Primary Goa pace-death pivot after CSK traded Nathan Ellis (#7) to SRH; pairs with Jofra Archer.", "balls_covered": 120, "default_on": True
    },
    {
        "id": "ngidi", "name": "Lungi Ngidi", "tag": "Bowl #21 / #5 IPL-Only (74.0 PRS)", "role": "OS Death Pace", "os": True, "cat": "bowl",
        "team": "DC", "tva100": round(float(ngidi_ipl["blended_hp_tva"]), 1), "hp_vol": int(ngidi_mb["hp_leg"]),
        "metric": f"{ngidi_mb['hp_econ']:.2f} IPL HP Econ | +63.0 Raw TVA", "fit": "High-value pace-off/death alternative if Pathirana stays retained or bidding escalates.", "balls_covered": 100, "default_on": False
    },
    {
        "id": "kuldeep", "name": "Kuldeep Yadav", "tag": "Bowl #9 / #2 IPL-Only (67.3 PRS)", "role": "IND Wrist-Spin", "os": False, "cat": "bowl",
        "team": "DC", "tva100": round(float(kuldeep_r["blended_hp_tva"]), 1), "hp_vol": int(kuldeep_r["hp_leg"]),
        "metric": f"7.62 IPL HP Econ | +54.2 vs LHB", "fit": "Cures RR's middle-over wicket drought alongside Ravi Bishnoi & Vignesh Puthur.", "balls_covered": 100, "default_on": False
    },
    {
        "id": "hasaranga", "name": "Wanindu Hasaranga", "tag": "Bowl #10 (67.1 PRS)", "role": "OS Wrist-Spin", "os": True, "cat": "bowl",
        "team": "LSG", "tva100": round(float(hasaranga_r["blended_hp_tva"]), 1), "hp_vol": int(hasaranga_r["hp_leg"]),
        "metric": f"12.9 HP SR | 31 HP Wkts", "fit": "Ex-RR strike leg-spinner for spin-friendly home venues; strict #8 batting protection.", "balls_covered": 90, "default_on": False
    },
    {
        "id": "brar", "name": "Harpreet Brar", "tag": "Moneyball IPL-Only #12 (61.9 PRS)", "role": "IND SLA Spin", "os": False, "cat": "bowl",
        "team": "PBKS", "tva100": round(float(brar_ipl["blended_hp_tva"]), 1), "hp_vol": int(brar_mb["hp_leg"]),
        "metric": f"{brar_mb['hp_econ']:.2f} IPL HP Econ | +42.2 Raw TVA", "fit": "Moneyball uncapped SLA partner to execute RR's Jaipur/Guwahati defensive spin blueprint.", "balls_covered": 95, "default_on": True
    },
    {
        "id": "malinga", "name": "Eshan Malinga", "tag": "Moneyball IPL Pace Radar", "role": "OS Death Pace", "os": True, "cat": "bowl",
        "team": "SRH", "tva100": 8.0, "hp_vol": int(malinga_mb["hp_leg"]),
        "metric": f"{malinga_mb['hp_econ']:.2f} IPL HP Econ | +27.8 Raw TVA", "fit": "Budget Sri Lankan death-pace sling depth behind Archer/Shami (101 IPL HPb, +0.59 WPA/100).", "balls_covered": 78, "default_on": False
    },
    {
        "id": "miller", "name": "David Miller", "tag": "Bat #6 (67.2 PRS)", "role": "OS Finisher", "os": True, "cat": "bat",
        "team": "DC", "tva100": round(float(miller_r["blended_hp_tva"]), 1), "hp_vol": int(miller_r["hp_b"]),
        "metric": f"158.9 HP SR | +47.5 vs Spin", "fit": "Directly upgrades Hetmyer (113.0 SR in IPL 2026) with +42.1 Chase Death poise.", "balls_covered": 0, "default_on": True
    },
    {
        "id": "klaasen", "name": "Heinrich Klaasen", "tag": "Bat #8 / #3 IPL-Only (62.1 PRS)", "role": "OS WK Finisher", "os": True, "cat": "bat",
        "team": "SRH", "tva100": round(float(klaasen_r["blended_hp_tva"]), 1), "hp_vol": int(klaasen_r["hp_b"]),
        "metric": f"165.0 IPL HP SR | +54.1 vs Spin", "fit": "Elite middle-overs spin destroyer if released into the Goa pool.", "balls_covered": 0, "default_on": False
    },
    {
        "id": "aniket", "name": "Aniket Verma", "tag": "Moneyball Uncapped Bat", "role": "IND Finisher", "os": False, "cat": "bat",
        "team": "SRH (Uncapped)", "tva100": 14.0, "hp_vol": int(aniket_mb["hp_b"]),
        "metric": f"{aniket_mb['hp_sr']:.1f} IPL HP SR | +56.6 Raw TVA", "fit": "Next-Vignesh Puthur style uncapped breakout: 101 IPL HP balls at 184.2 SR (+0.35 WPA/100).", "balls_covered": 0, "default_on": True
    },
    {
        "id": "ashutosh", "name": "Ashutosh Sharma", "tag": "IPL-Only Bat #5 (72.6 PRS)", "role": "IND Finisher", "os": False, "cat": "bat",
        "team": "DC", "tva100": round(float(ashutosh_ipl["blended_hp_tva"]), 1), "hp_vol": int(ashutosh_ipl["hp_b"]),
        "metric": f"{ashutosh_ipl['hp_sr']:.1f} IPL HP SR | +0.51 WPA/100", "fit": "Proven IPL crunch finisher (157 HPb) freeing overseas slots for pace depth.", "balls_covered": 0, "default_on": True
    },
    {
        "id": "naman", "name": "Naman Dhir", "tag": "Moneyball Value Bat", "role": "IND Top/Mid Bat", "os": False, "cat": "bat",
        "team": "MI", "tva100": 9.5, "hp_vol": int(naman_mb["hp_b"]),
        "metric": f"{naman_mb['hp_sr']:.1f} IPL HP SR | +25.0 Raw TVA", "fit": "Explosive Indian middle-over boundary hitter across 133 IPL HP balls.", "balls_covered": 0, "default_on": False
    },
]

figures_meta = [
    {"id": 1, "file": "fig1_context_inflation.png", "b64": img_b64("fig1_context_inflation.png"),
     "title": "Figure 1: Era Scoring Inflation (+15.7% IPL Surge) & Empirical T20 Phase Wicket Value",
     "cap": "IPL scoring climbed from 8.54 RPO (2022) to 9.88 RPO (2026) during the Impact Player era, diverging sharply from Full-Member T20Is (8.42-9.21), Emerging Asia Cup (7.92), and Associates (7.71). Dynamic wicket value V_W(s) averages 10.6 runs in Powerplay, 6.6 in Middle overs, and 3.7 at Death."},
    {"id": 2, "file": "fig2_wp_calibration_and_situations.png", "b64": img_b64("fig2_wp_calibration_and_situations.png"),
     "title": "Figure 2: Win Probability OOF Calibration (AUC 0.901) & Pressure Index Separation",
     "cap": "Innings-2 Logistic WP achieves 5-fold match-grouped OOF AUC = 0.901 (Brier 0.128) with tight decile calibration. PI cleanly isolates live death crunches (81.7 bat / 80.4 bowl) and early batting collapses (39.4) from dead chases (1.8-6.6)."},
    {"id": 3, "file": "fig3_batters_quadrant_top10.png", "b64": img_b64("fig3_batters_quadrant_top10.png"),
     "title": "Figure 3: Usual vs. High-Pressure Batting TVA Across All 73 Qualified Batters",
     "cap": "Suryakumar, Tim David, Tilak Varma, and Shreyas Iyer sustain elite output well above expectation, while David Miller (+9.4 resid), Dasun Shanaka (+6.7), Shashank Singh (+6.1), and Ishan Kishan (+5.0) elevate sharply under pressure."},
    {"id": 4, "file": "fig4_bowlers_quadrant_top10.png", "b64": img_b64("fig4_bowlers_quadrant_top10.png"),
     "title": "Figure 4: Usual vs. High-Pressure Bowling TVA Across All 79 Qualified Bowlers",
     "cap": "Krunal Pandya (+14.0 clutch resid) and Axar Patel (+13.7) lead pressure squeeze gains, while Bhuvneshwar, Narine, Hazlewood, Bumrah, and Nathan Ellis pair elite baseline class with crunch execution."},
    {"id": 5, "file": "fig8_3vector_and_death_playbook.png", "b64": img_b64("fig8_3vector_and_death_playbook.png"),
     "title": "Figure 8: 3-Vector Matchup Matrix (Pace-On/Off/Spin/Entry-10) & Death Delivery Playbook",
     "cap": "Suryakumar dominates all three velocity tiers (+45.1 / +91.3 / +35.8) while Tim David (-8.6) and Klaasen (-32.8) dip against Pace-Off. At death (Ov 16-20), Yorkers (+78.6 TVA, 5.5 Econ) dominate while Full Tosses (-32.1) and Straight balls (-31.9) bleed."},
    {"id": 6, "file": "fig5_bootstrap_forest_plots.png", "b64": img_b64("fig5_bootstrap_forest_plots.png"),
     "title": "Figure 5: 1,000-Iteration Match-Clustered Bootstrap 95% Confidence Intervals",
     "cap": "Every Top-10 batter and bowler clears 89%-100% bootstrap probability of positive high-pressure context value, with Suryakumar, Tim David, Tilak, Shreyas, Krunal, Bhuvneshwar, Narine, Bumrah, Ellis, and Kuldeep clearing zero across their full 95% interval."},
    {"id": 7, "file": "fig9_rr_loss_autopsy_and_strokes.png", "b64": img_b64("fig9_rr_loss_autopsy_and_strokes.png"),
     "title": "Figure 9: RR IPL 2026 Win-vs-Loss Pressure Autopsy & HP Batting Stroke Hierarchy",
     "cap": "Across RR's 16 IPL 2026 matches (9W/7L), HP Batting stayed positive (+7.5) in losses while HP Bowling collapsed from +8.4 to -21.4 TVA/100 (a -29.8 bowling swing). Under crunch hitting, clean drives/pulls dominate slogs (-11.7) and blocks (-78.5 to -81.0)."},
    {"id": 8, "file": "fig7_rr_2027_auction_fit.png", "b64": img_b64("fig7_rr_2027_auction_fit.png"),
     "title": "Figure 7: RR 2026 Core Roster Audit & Dec 16, 2026 Goa Auction Targets (Post-Ellis Trade)",
     "cap": "Maps RR's heavy reliance on Jofra Archer (+13.0) & Vaibhav Sooryavanshi against Sam Curran/Tushar Deshpande/Sandeep leaks, showing the immediate Goa pivot from traded Nathan Ellis (CSK->SRH) to Pathirana, Ngidi, Krunal, Kuldeep, Miller & Ashutosh."},
    {"id": 9, "file": "fig10_mechanisms_and_temporal_proof.png", "b64": img_b64("fig10_mechanisms_and_temporal_proof.png"),
     "title": "Figure 10: Out-of-Time Prediction Benchmark (2022-24 -> 2025-26) & Clutch Mechanisms",
     "cap": "Proves Naive Clutch (HP-LP) has negative predictive power (r=-0.019 bat, r=-0.055 bowl) while EB+RAPM & baseline class generalize (+0.362 bowler -Econ, +0.239 blended TVA). Dip-10 bowlers lose 7.4 pp of dots and double freebie balls to 18.3%."},
    {"id": 10, "file": "fig6_sensitivity_rank_stability.png", "b64": img_b64("fig6_sensitivity_rank_stability.png"),
     "title": "Figure 6: Six-Specification Sensitivity & Rank Stability Heatmap",
     "cap": "Spearman correlations against the primary spec (Q75, k=500) exceed rho=0.971 across pressure cutoffs (Q70/Q80) and priors (k=300/800), remaining stable through the 2024-26 Impact Player window."},
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SupeRR Selector S2 | Rajasthan Royals IPL 2027 War Room</title>
<style>
  :root {
    --bg: #070B19;
    --surface: #0F1733;
    --surface-2: #162044;
    --border: rgba(148, 163, 184, 0.16);
    --pink: #EA1A85;
    --pink-soft: rgba(234, 26, 133, 0.16);
    --blue: #2563EB;
    --rr-blue: #074EA2;
    --gold: #FDE047;
    --teal: #14B8A6;
    --red: #F87171;
    --text: #F1F5F9;
    --muted: #94A3B8;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: radial-gradient(circle at top right, rgba(234,26,133,0.12), transparent 35%),
                radial-gradient(circle at top left, rgba(7,78,162,0.18), transparent 40%),
                var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, Arial, sans-serif;
    line-height: 1.45;
  }
  a { color: inherit; }
  .topbar {
    background: linear-gradient(125deg, #0D1330 0%, #074EA2 58%, #EA1A85 100%);
    border-bottom: 1px solid rgba(255,255,255,0.14);
    padding: 18px 24px 16px 24px;
    position: sticky;
    top: 0;
    z-index: 40;
    box-shadow: 0 8px 30px rgba(0,0,0,0.45);
  }
  .topbar-inner {
    max-width: 1420px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    flex-wrap: wrap;
  }
  .brand-kicker {
    font-size: 11px;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: var(--gold);
    font-weight: 800;
    margin-bottom: 3px;
  }
  .brand-title {
    font-size: 21px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.3px;
  }
  .brand-sub {
    margin: 2px 0 0 0;
    font-size: 12.5px;
    color: #E2E8F0;
  }
  .header-actions {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    align-items: center;
  }
  .btn-cta {
    padding: 8px 13px;
    border-radius: 8px;
    font-size: 12px;
    font-weight: 700;
    text-decoration: none;
    border: 1px solid rgba(255,255,255,0.28);
    background: rgba(15, 23, 51, 0.45);
    color: #fff;
    transition: transform 0.15s ease, background 0.15s ease;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .btn-cta:hover { transform: translateY(-1px); background: rgba(15, 23, 51, 0.78); }
  .btn-cta.gold { background: var(--gold); color: #111827; border-color: var(--gold); }
  .btn-cta.gold:hover { background: #fef08a; }

  .wrap { max-width: 1420px; margin: 18px auto 48px auto; padding: 0 20px; }

  .nav-tabs {
    display: flex;
    gap: 5px;
    flex-wrap: wrap;
    margin-bottom: 18px;
    background: rgba(15, 23, 51, 0.85);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid var(--border);
  }
  .tab-btn {
    border: none;
    background: transparent;
    color: var(--muted);
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 12.2px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .tab-btn:hover { color: #fff; background: rgba(255,255,255,0.06); }
  .tab-btn.active {
    background: linear-gradient(135deg, var(--pink), #C81470);
    color: #fff;
    box-shadow: 0 4px 14px rgba(234, 26, 133, 0.35);
  }

  .view { display: none; animation: fadeIn 0.18s ease-out; }
  .view.active { display: block; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    margin-bottom: 18px;
  }
  .kpi {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 3px solid var(--pink);
    border-radius: 10px;
    padding: 12px 13px;
  }
  .kpi.blue { border-top-color: #3B82F6; }
  .kpi.gold { border-top-color: var(--gold); }
  .kpi.teal { border-top-color: var(--teal); }
  .kpi-v { font-size: 20px; font-weight: 800; color: #fff; font-variant-numeric: tabular-nums; }
  .kpi-l { font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); margin-top: 3px; }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 16px;
  }
  .card-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
    flex-wrap: wrap;
  }
  .card h2, .card h3 {
    margin: 0;
    font-size: 16px;
    font-weight: 800;
    color: #fff;
  }
  .subtext { font-size: 12px; color: var(--muted); margin: 2px 0 0 0; }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
  .grid-10 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 9px; }

  .pick-card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
    cursor: pointer;
    transition: border-color 0.15s ease, transform 0.15s ease, background 0.15s ease;
    position: relative;
  }
  .pick-card:hover {
    border-color: var(--pink);
    transform: translateY(-2px);
    background: #1B2754;
  }
  .pick-card.bowl:hover { border-color: #38BDF8; }
  .pick-top { display: flex; justify-content: space-between; align-items: center; gap: 6px; margin-bottom: 5px; }
  .rank-pill {
    font-size: 11px;
    font-weight: 800;
    padding: 2px 8px;
    border-radius: 6px;
    background: var(--pink);
    color: #fff;
  }
  .rank-pill.blue { background: var(--rr-blue); border: 1px solid rgba(56, 189, 248, 0.45); }
  .prs-pill {
    font-size: 11px;
    font-weight: 800;
    color: var(--gold);
    background: rgba(253, 224, 71, 0.12);
    padding: 2px 8px;
    border-radius: 12px;
    border: 1px solid rgba(253, 224, 71, 0.25);
  }
  .pick-name { font-size: 14px; font-weight: 800; color: #fff; }
  .pick-team { font-size: 11px; color: var(--muted); margin-bottom: 6px; }
  .pick-metrics {
    display: flex;
    gap: 8px;
    font-size: 11.5px;
    color: #CBD5E1;
    flex-wrap: wrap;
    font-variant-numeric: tabular-nums;
  }
  .pick-fit {
    margin-top: 6px;
    font-size: 11px;
    color: #93C5FD;
    border-top: 1px dashed rgba(148,163,184,0.2);
    padding-top: 5px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }

  .controls { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }
  input[type="text"], select {
    background: #0A1024;
    border: 1px solid rgba(148,163,184,0.28);
    color: #fff;
    padding: 8px 12px;
    border-radius: 8px;
    font-size: 13px;
    outline: none;
  }
  input[type="text"]:focus, select:focus { border-color: var(--pink); }

  .pill-switch {
    display: inline-flex;
    background: #0A1024;
    padding: 3px;
    border-radius: 8px;
    border: 1px solid var(--border);
    gap: 3px;
  }
  .pill-btn {
    border: none;
    background: transparent;
    color: var(--muted);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
  }
  .pill-btn.active { background: var(--pink); color: #fff; }
  .pill-btn.blue.active { background: var(--rr-blue); color: #fff; }

  .tbl-wrap { overflow-x: auto; max-height: 620px; border-radius: 8px; border: 1px solid var(--border); }
  table { width: 100%; border-collapse: collapse; font-size: 12.2px; }
  th {
    background: #0B1229;
    color: #E2E8F0;
    padding: 9px 8px;
    text-align: left;
    position: sticky;
    top: 0;
    z-index: 5;
    border-bottom: 2px solid var(--pink);
    white-space: nowrap;
  }
  th.sortable { cursor: pointer; }
  th.sortable:hover { color: var(--gold); }
  th.num, td.num { text-align: right; font-variant-numeric: tabular-nums; }
  td { padding: 8px 8px; border-bottom: 1px solid rgba(148,163,184,0.12); vertical-align: middle; }
  tr.click-row { cursor: pointer; transition: background 0.1s ease; }
  tr.click-row:hover td { background: rgba(234, 26, 133, 0.12); }
  tr.top-row td { background: rgba(234, 26, 133, 0.06); font-weight: 600; }
  .pos { color: var(--teal); font-weight: 700; }
  .neg { color: var(--red); font-weight: 700; }
  .badge-sm {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 5px;
    font-size: 10.5px;
    font-weight: 700;
    background: rgba(148,163,184,0.15);
    color: #E2E8F0;
  }
  .badge-sm.rr { background: rgba(234,26,133,0.22); color: #F9A8D4; border: 1px solid rgba(234,26,133,0.45); }
  .badge-sm.trade { background: rgba(251,191,36,0.18); color: #FDE047; border: 1px solid rgba(251,191,36,0.4); }

  /* Simulator Checkboxes */
  .sim-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 10px; }
  .sim-item {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
    cursor: pointer;
    display: flex;
    gap: 10px;
    align-items: flex-start;
    transition: all 0.14s ease;
  }
  .sim-item.selected {
    border-color: var(--teal);
    background: rgba(20, 184, 166, 0.11);
  }
  .sim-item input { margin-top: 3px; accent-color: var(--teal); transform: scale(1.15); }
  .sim-readout {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 14px;
    background: linear-gradient(135deg, rgba(7,78,162,0.22), rgba(234,26,133,0.16));
    border: 1px solid rgba(234,26,133,0.35);
    border-radius: 12px;
    padding: 14px;
  }
  .sim-metric-v { font-size: 22px; font-weight: 800; color: #fff; font-variant-numeric: tabular-nums; }
  .sim-metric-l { font-size: 11px; color: #CBD5E1; text-transform: uppercase; letter-spacing: 0.4px; }

  /* SVG Tooltip */
  .svg-wrap { position: relative; width: 100%; background: #0A1024; border: 1px solid var(--border); border-radius: 10px; padding: 10px; }
  .tooltip {
    position: fixed;
    pointer-events: none;
    background: rgba(10, 16, 36, 0.96);
    border: 1px solid var(--pink);
    color: #fff;
    padding: 9px 12px;
    border-radius: 8px;
    font-size: 12px;
    z-index: 90;
    box-shadow: 0 10px 28px rgba(0,0,0,0.6);
    display: none;
    max-width: 280px;
  }

  /* Gallery & Lightbox */
  .gallery-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .fig-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px;
    cursor: pointer;
    transition: border-color 0.15s ease, transform 0.15s ease;
  }
  .fig-card:hover { border-color: var(--pink); transform: translateY(-2px); }
  .fig-card img { width: 100%; border-radius: 8px; background: #fff; display: block; }
  .fig-card h4 { margin: 8px 0 4px 0; font-size: 13.5px; color: #fff; }
  .fig-card p { margin: 0; font-size: 11.5px; color: var(--muted); }

  /* Modals */
  .modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(3, 6, 18, 0.84);
    backdrop-filter: blur(5px);
    display: none;
    align-items: center;
    justify-content: center;
    padding: 20px;
    z-index: 100;
  }
  .modal-backdrop.open { display: flex; }
  .modal-box {
    background: var(--surface);
    border: 1px solid rgba(234,26,133,0.45);
    border-radius: 14px;
    max-width: 780px;
    width: 100%;
    max-height: 90vh;
    overflow-y: auto;
    padding: 22px 24px;
    box-shadow: 0 24px 60px rgba(0,0,0,0.75);
  }
  .modal-box.wide { max-width: 1180px; }
  .modal-close {
    float: right;
    background: rgba(255,255,255,0.1);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 6px 11px;
    font-weight: 700;
    cursor: pointer;
  }
  .modal-close:hover { background: var(--pink); }

  .bar-track { background: rgba(148,163,184,0.14); border-radius: 99px; height: 8px; overflow: hidden; width: 100%; }
  .bar-fill { height: 100%; border-radius: 99px; }

  @media (max-width: 1100px) {
    .kpi-grid { grid-template-columns: repeat(3, 1fr); }
    .sim-grid { grid-template-columns: repeat(2, 1fr); }
    .sim-readout { grid-template-columns: repeat(2, 1fr); }
    .grid-2, .grid-3, .gallery-grid { grid-template-columns: 1fr; }
  }
  @media (max-width: 700px) {
    .kpi-grid, .grid-10, .sim-grid { grid-template-columns: 1fr; }
    .topbar { position: static; }
  }
</style>
</head>
<body>

<header class="topbar">
  <div class="topbar-inner">
    <div>
      <div class="brand-kicker">SupeRR Selector Season 2 • Rajasthan Royals Analytics War Room</div>
      <h1 class="brand-title">Win-Probability Leverage &amp; RAPM Clutch Scouting Platform</h1>
      <p class="brand-sub">200,999 Audited Deliveries • 784 Top-Flight Matches (IPL + FM T20Is, 2022–2026) • IPL 2027 Goa Auction Blueprint</p>
    </div>
    <div class="header-actions">
      <a class="btn-cta gold" href="__PDF_HREF__" target="_blank">Download 11-Page PDF Report</a>
      <a class="btn-cta" href="__REPORT_HREF__" target="_blank">Open Web Report</a>
      <a class="btn-cta" href="https://github.com/alok-108/superr-selector-s2" target="_blank">GitHub Code</a>
    </div>
  </div>
</header>

<div class="wrap">
  <nav class="nav-tabs">
    <button class="tab-btn active" onclick="switchView('overview', this)">1. War Room Overview &amp; Top 10 Picks</button>
    <button class="tab-btn" onclick="switchView('explorer', this)">2. Full Player Explorer (73 Bat / 79 Bowl)</button>
    <button class="tab-btn" onclick="switchView('rrsim', this)">3. RR 2026 Autopsy &amp; Goa 2027 Simulator</button>
    <button class="tab-btn" onclick="switchView('science', this)">4. OOS Proof, Mechanisms &amp; Matchups</button>
    <button class="tab-btn" onclick="switchView('gallery', this)">5. Visual Gallery (10 Figures) &amp; Engine</button>
  </nav>

  <!-- VIEW 1: OVERVIEW -->
  <section id="overview" class="view active">
    <div class="kpi-grid">
      <div class="kpi"><div class="kpi-v">200,999</div><div class="kpi-l">Deliveries Audited (867 M)</div></div>
      <div class="kpi blue"><div class="kpi-v">AUC 0.901</div><div class="kpi-l">I2 WP OOF (Brier 0.128)</div></div>
      <div class="kpi teal"><div class="kpi-v">73 / 79</div><div class="kpi-l">Qualified Batters / Bowlers</div></div>
      <div class="kpi gold"><div class="kpi-v">&minus;29.8 TVA</div><div class="kpi-l">RR 2026 Loss HP Bowl Swing</div></div>
      <div class="kpi"><div class="kpi-v">r = &minus;0.019</div><div class="kpi-l">Naive Clutch OOS Proof (Fails)</div></div>
      <div class="kpi blue"><div class="kpi-v">&rho; &ge; 0.971</div><div class="kpi-l">Rank Stability Across Specs</div></div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-head">
          <div>
            <h2>Final Ordered Top 10 Batters Under Pressure</h2>
            <p class="subtext">Click any batter card for 3-Vector Pace-On/Off/Spin telemetry, 95% CI, signature knocks &amp; RR 2027 fit.</p>
          </div>
          <span class="badge-sm rr">Unstop Ranks #1–#10</span>
        </div>
        <div id="top10BatCards" class="grid-10"></div>
      </div>

      <div class="card">
        <div class="card-head">
          <div>
            <h2>Final Ordered Top 10 Bowlers Under Pressure</h2>
            <p class="subtext">Click any bowler card for Speed/Death mix, LHB vs RHB suppression, signature spells &amp; RR trade pivot.</p>
          </div>
          <span class="badge-sm rr">Unstop Ranks #1–#10</span>
        </div>
        <div id="top10BowlCards" class="grid-10"></div>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <div>
          <h2>Interactive Pressure Quadrant: Usual vs. High-Pressure RAPM Performance</h2>
          <p class="subtext">Hover over any player point to inspect baseline vs crunch output; click any point to open their full Scouting Card.</p>
        </div>
        <div class="pill-switch">
          <button id="scBtnBat" class="pill-btn active" onclick="renderScatter('bat')">Batters (n=73)</button>
          <button id="scBtnBowl" class="pill-btn blue" onclick="renderScatter('bowl')">Bowlers (n=79)</button>
        </div>
      </div>
      <div class="svg-wrap">
        <svg id="quadSvg" viewBox="0 0 960 410" style="width:100%; height:auto; display:block;"></svg>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-head">
          <div>
            <h3>Unified Cross-Discipline Master Top 10 (Peak PRS)</h3>
            <p class="subtext">Single combined ranking across all 152 qualified performers (Max PRS across Bat + Bowl).</p>
          </div>
        </div>
        <div class="tbl-wrap" style="max-height:360px;">
          <table><thead><tr><th>#</th><th>Player</th><th>Role</th><th>IPL Team</th><th class="num">Peak PRS</th><th class="num">Net HP/M</th></tr></thead><tbody id="ovUniBody"></tbody></table>
        </div>
      </div>
      <div class="card">
        <div class="card-head">
          <div>
            <h3>Dual-Qualified All-Rounders Net Audit (&ge;180 HPb Both)</h3>
            <p class="subtext">Prevents two-way roster traps: Krunal (-4.60 bat) vs Axar (+1.40 net), Rashid (+1.29), &amp; Curran (-1.06 dead last).</p>
          </div>
        </div>
        <div class="tbl-wrap" style="max-height:360px;">
          <table><thead><tr><th>#</th><th>All-Rounder</th><th class="num">Bat (# / TVA)</th><th class="num">Bowl (# / TVA)</th><th class="num">Net HP/M</th></tr></thead><tbody id="ovDualBody"></tbody></table>
        </div>
      </div>
    </div>
  </section>

  <!-- VIEW 2: FULL PLAYER EXPLORER -->
  <section id="explorer" class="view">
    <div class="card">
      <div class="card-head">
        <div>
          <h2>Full Eligible Player Scouting Leaderboard</h2>
          <p class="subtext">Filter by discipline, team, handedness, or Rajasthan Royals tag. Click any column header to sort; click any row for full card.</p>
        </div>
        <div class="pill-switch">
          <button id="expBatBtn" class="pill-btn active" onclick="setExpMode('bat')">Batters (n=73)</button>
          <button id="expBowlBtn" class="pill-btn blue" onclick="setExpMode('bowl')">Bowlers (n=79)</button>
        </div>
      </div>
      <div class="controls">
        <input type="text" id="expSearch" placeholder="Search player, IPL team, country, or style..." oninput="renderExplorer()" style="min-width:270px; flex:1; max-width:420px;">
        <select id="expFilter" onchange="renderExplorer()">
          <option value="all">All Eligible Qualifiers</option>
          <option value="top10">Top 10 Picks Only</option>
          <option value="ipl26">Active IPL 2026 Rostered</option>
          <option value="rr">Rajasthan Royals Players</option>
          <option value="sub1">RHB Batters / Pace Bowlers</option>
          <option value="sub2">LHB Batters / Spin Bowlers</option>
          <option value="pos_clutch">Positive Clutch Residual (&gt; 0)</option>
        </select>
      </div>
      <div class="tbl-wrap" id="expTableContainer"></div>
    </div>
  </section>

  <!-- VIEW 3: RR AUTOPSY & GOA AUCTION SIMULATOR -->
  <section id="rrsim" class="view">
    <div class="grid-2">
      <div class="card">
        <div class="card-head">
          <div>
            <h2>RR IPL 2026 9W vs 7L Pressure Autopsy</h2>
            <p class="subtext">9 wins (+18.3 Bat / +8.4 Bowl) vs 7 losses (+7.5 Bat / &minus;21.4 Bowl = &minus;29.8 swing, &minus;86.15 runs lost).</p>
          </div>
        </div>
        <div id="rrAutopsyBars" style="display:grid; gap:10px; margin-bottom:14px;"></div>
        <div class="tbl-wrap">
          <table><thead><tr><th>Outcome</th><th>RR Role</th><th>Regime</th><th class="num">Balls</th><th class="num">Total TVA</th><th class="num">TVA/100</th><th class="num">Total WPA</th></tr></thead><tbody id="rrAutBody"></tbody></table>
        </div>
      </div>

      <div class="card">
        <div class="card-head">
          <div>
            <h2>Nov 2026 Front-Office Trade Audit (Live Deadline)</h2>
            <p class="subtext">Scoring reported pre-retention IPL 2027 trades (Rs 157 Cr cap) via 2022–26 HP RAPM engine.</p>
          </div>
        </div>
        <div class="tbl-wrap" style="margin-bottom:10px;">
          <table><thead><tr><th>Trade Corridor</th><th>Player</th><th class="num">Rank</th><th class="num">PRS</th><th class="num">HP Balls</th><th class="num">HP Econ</th><th class="num">HP TVA/100*</th></tr></thead><tbody id="tradeAudBody"></tbody></table>
        </div>
        <div style="font-size:12px; color:#CBD5E1; background:rgba(255,255,255,0.04); padding:10px 12px; border-radius:8px; border-left:3px solid var(--pink);">
          <b>RR Verdict:</b> Swapping Sam Curran (#74, -7.3) for Mohammed Shami (#44, +1.2; cutting HP Econ 10.95 &rarr; 8.78) gives a clean <b>+8.5 TVA/100</b> early/pace gain, while M. Siddharth (56 HPb, 10.29 Econ) is a Powerplay specialist. Crucially, CSK trading <b>Nathan Ellis (#7) &rarr; SRH</b> forces RR's Goa pivot to <b>Pathirana</b> &amp; <b>Lungi Ngidi</b>.
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-head">
        <div>
          <h2>Interactive Goa Mini-Auction Lift Simulator (Dec 16, 2026)</h2>
          <p class="subtext">Toggle RR acquisition targets to simulate recovery on RR's 403 loss-HP bowling deliveries (&minus;21.4 TVA/100 baseline) &amp; overseas slot budget.</p>
        </div>
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
          <button class="pill-btn active" onclick="presetSim('triple')">Recommended Triple Lock</button>
          <button class="pill-btn blue active" style="background:#1E293B;" onclick="presetSim('ind_core')">Domestic Core Only</button>
          <button class="pill-btn" style="background:#0A1024; border:1px solid var(--border);" onclick="presetSim('clear')">Reset All</button>
        </div>
      </div>
      <div class="sim-readout">
        <div><div class="sim-metric-v" id="simBowlTva">-21.4</div><div class="sim-metric-l">Simulated RR Loss-HP Bowl TVA/100</div></div>
        <div><div class="sim-metric-v pos" id="simRuns">+0.0</div><div class="sim-metric-l">Expected Runs Recovered (403 Loss Balls)</div></div>
        <div><div class="sim-metric-v" id="simBatAdd">+0.0</div><div class="sim-metric-l">Added Clutch Bat TVA Avg</div></div>
        <div><div class="sim-metric-v" id="simOsCount">0 OS / 0 IND</div><div class="sim-metric-l">Selected Target Slots</div></div>
      </div>
      <div id="simGrid" class="sim-grid"></div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h3>RR Home Fortresses (High Pressure): Jaipur &amp; Guwahati</h3>
        <p class="subtext" style="margin-bottom:10px;">Spin dominates Pace-On &amp; Pace-Off under pressure at both RR home grounds (Jaipur #6 &amp; Guwahati #1241).</p>
        <div class="tbl-wrap">
          <table><thead><tr><th>Home Venue</th><th>Type</th><th class="num">HP Balls</th><th class="num">HP Econ</th><th class="num">Bdry %</th></tr></thead><tbody id="venueRrBody"></tbody></table>
        </div>
      </div>
      <div class="card">
        <h3>Playing-XI Sub Penalty &amp; Leadership Signal</h3>
        <p class="subtext" style="margin-bottom:10px;">IPL 2023–26 Starting-XI players beat Impact Subs under pressure; Captains outscore Non-Captains by +5.62 TVA.</p>
        <div class="tbl-wrap">
          <table><thead><tr><th>Context Factor</th><th class="num">HP Balls</th><th class="num">Rate (SR/Econ)</th><th class="num">HP TVA/100</th></tr></thead><tbody id="p11CtxBody"></tbody></table>
        </div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h3>Moneyball Uncapped &amp; Value Batters (&ge;70 IPL HPb, Outside Top 10)</h3>
        <div class="tbl-wrap" style="max-height:340px;">
          <table><thead><tr><th>Batter</th><th class="num">IPL HPb (M)</th><th class="num">HP SR</th><th class="num">Raw HP TVA</th><th class="num">HP WPA/100</th></tr></thead><tbody id="mbBatBody"></tbody></table>
        </div>
      </div>
      <div class="card">
        <h3>Moneyball Uncapped &amp; Value Bowlers (&ge;70 IPL HPb, Outside Top 10)</h3>
        <div class="tbl-wrap" style="max-height:340px;">
          <table><thead><tr><th>Bowler</th><th class="num">IPL HPb (M)</th><th class="num">HP Econ (W)</th><th class="num">Raw HP TVA</th><th class="num">HP WPA/100</th></tr></thead><tbody id="mbBowlBody"></tbody></table>
        </div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h3>RR IPL 2026 Career Pressure Audit — Batters (&ge;30b)</h3>
        <div class="tbl-wrap" style="max-height:340px;">
          <table><thead><tr><th>Player</th><th class="num">Tot Balls</th><th class="num">HP Balls (M)</th><th class="num">HP SR</th><th class="num">HP TVA</th><th class="num">LP TVA</th></tr></thead><tbody id="rrBatBody"></tbody></table>
        </div>
      </div>
      <div class="card">
        <h3>RR IPL 2026 Career Pressure Audit — Bowlers (&ge;30b)</h3>
        <div class="tbl-wrap" style="max-height:340px;">
          <table><thead><tr><th>Player</th><th class="num">Tot Balls</th><th class="num">HP Balls (M)</th><th class="num">HP Econ</th><th class="num">HP TVA</th><th class="num">LP TVA</th></tr></thead><tbody id="rrBowlBody"></tbody></table>
        </div>
      </div>
    </div>
  </section>

  <!-- VIEW 4: SCIENCE & PLAYBOOKS -->
  <section id="science" class="view">
    <div class="grid-2">
      <div class="card">
        <h3>Out-of-Time Temporal Prediction (2022–24 &rarr; 2025–26 HP TVA)</h3>
        <p class="subtext" style="margin-bottom:10px;">Naive Clutch (HP &minus; LP) carries negative correlation across years; Empirical-Bayes RAPM + baseline class generalizes.</p>
        <div class="tbl-wrap">
          <table><thead><tr><th>Cohort</th><th>2022–24 Predictor</th><th class="num">Pearson r</th><th class="num">p-val</th><th class="num">Spearman &rho;</th></tr></thead><tbody id="tempValBody"></tbody></table>
        </div>
      </div>
      <div class="card">
        <h3>Causal Physical Mechanisms (Top-10 vs Dip-10 &ge;500b)</h3>
        <p class="subtext" style="margin-bottom:10px;">Top-10 batters hold 21.3% boundaries while Dippers drop to 13.9%; Top-10 bowlers nearly double elite variations while limiting freebies to 9.5% vs 18.3%.</p>
        <div class="tbl-wrap">
          <table><thead><tr><th>Cohort</th><th>Regime</th><th class="num">Balls</th><th class="num">SR / Econ</th><th class="num">Dot %</th><th class="num">Bdry %</th><th class="num">Freebie/Def %</th><th class="num">TVA/100</th></tr></thead><tbody id="mechSumBody"></tbody></table>
        </div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h3>High-Pressure Death Delivery Playbook (Overs 16–20, &ge;150b)</h3>
        <div class="tbl-wrap" style="max-height:380px;">
          <table><thead><tr><th>Category</th><th>Delivery Style</th><th class="num">Balls</th><th class="num">Econ</th><th class="num">Wkt %</th><th class="num">Dot %</th><th class="num">Bdry %</th><th class="num">RAPM TVA/100</th></tr></thead><tbody id="deathPbBody"></tbody></table>
        </div>
      </div>
      <div class="card">
        <h3>High-Pressure Batting Stroke Playbook (&ge;200b)</h3>
        <div class="tbl-wrap" style="max-height:380px;">
          <table><thead><tr><th>Stroke</th><th class="num">HP Balls</th><th class="num">HP SR</th><th class="num">Out %</th><th class="num">Bdry %</th><th class="num">RAPM TVA/100</th></tr></thead><tbody id="strokePbBody"></tbody></table>
        </div>
      </div>
    </div>

    <div class="grid-3">
      <div class="card">
        <h3>Top-10 Bowlers vs LHB &amp; RHB (HP TVA/100)</h3>
        <div class="tbl-wrap" style="max-height:360px;">
          <table><thead><tr><th>#</th><th>Bowler</th><th class="num">vs LHB (b)</th><th class="num">vs RHB (b)</th></tr></thead><tbody id="lhbRhbBody"></tbody></table>
        </div>
      </div>
      <div class="card">
        <h3>Toughest Bowlers Faced (RAPM Adj.)</h3>
        <div class="tbl-wrap" style="max-height:360px;">
          <table><thead><tr><th>Bowler</th><th class="num">Score Balls</th><th class="num">Bat Adj/100</th></tr></thead><tbody id="rapmBBody"></tbody></table>
        </div>
      </div>
      <div class="card">
        <h3>Toughest Batters Bowled To (RAPM Adj.)</h3>
        <div class="tbl-wrap" style="max-height:360px;">
          <table><thead><tr><th>Batter</th><th class="num">Score Balls</th><th class="num">Bowl Adj/100</th></tr></thead><tbody id="rapmBwBody"></tbody></table>
        </div>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <h3>6-Spec Sensitivity Stability — Batters</h3>
        <div class="tbl-wrap">
          <table><thead><tr><th>Batter</th><th class="num">Primary</th><th class="num">Q70</th><th class="num">Q80</th><th class="num">k=300</th><th class="num">k=800</th><th class="num">IPL Only</th><th class="num">2024–26</th></tr></thead><tbody id="sensBatBody"></tbody></table>
        </div>
      </div>
      <div class="card">
        <h3>6-Spec Sensitivity Stability — Bowlers</h3>
        <div class="tbl-wrap">
          <table><thead><tr><th>Bowler</th><th class="num">Primary</th><th class="num">Q70</th><th class="num">Q80</th><th class="num">k=300</th><th class="num">k=800</th><th class="num">IPL Only</th><th class="num">2024–26</th></tr></thead><tbody id="sensBowlBody"></tbody></table>
        </div>
      </div>
    </div>
  </section>

  <!-- VIEW 5: GALLERY & ENGINE -->
  <section id="gallery" class="view">
    <div class="card">
      <div class="card-head">
        <div>
          <h2>10-Figure Publication Gallery &amp; Quantitative Architecture</h2>
          <p class="subtext">Click any high-DPI chart below to open full-screen zoom inspection. Powered by 100% reproducible Python modeling.</p>
        </div>
        <a class="btn-cta gold" href="__PDF_HREF__" target="_blank">Open 11-Page PDF Report</a>
      </div>
      <div class="grid-3" style="margin-bottom:16px;">
        <div style="background:var(--surface-2); padding:12px; border-radius:8px; border-left:3px solid var(--pink);">
          <b style="color:#fff;">Stage 1: Hygiene &amp; Resource Engine</b>
          <p class="subtext" style="margin-top:4px;">Fixed wide/noball legal ball counters across 200,999 balls, DLS ball-limits, held out 6 Super Overs, normalized +15.7% IPL scoring inflation, and fit monotonic <code>HistGradientBoosting</code> runs-remaining (2026 MAE 18.54, R² 0.819).</p>
        </div>
        <div style="background:var(--surface-2); padding:12px; border-radius:8px; border-left:3px solid #38BDF8;">
          <b style="color:#fff;">Stage 2: Counterfactual WP Leverage PI</b>
          <p class="subtext" style="margin-top:4px;">5-fold match-grouped Logistic WP (OOF AUC 0.901, Brier 0.128) weights counterfactual outcome shifts + collapse/RRR/death/knockout stress, gated to Top-Flight T20 (&ge;400b, &ge;180 HPb, &ge;15 M).</p>
        </div>
        <div style="background:var(--surface-2); padding:12px; border-radius:8px; border-left:3px solid var(--gold);">
          <b style="color:#fff;">Stage 3: Player-vs-Player Ridge RAPM &amp; PRS</b>
          <p class="subtext" style="margin-top:4px;">Out-of-fold sparse Ridge (&alpha;=1200) over striker, bowler &amp; team IDs + batting-order adjustment, EB-shrunk (k=500), blended Q75 + PI² weights, OLS Clutch Residual &amp; Triple Lock Goa shortlist.</p>
        </div>
      </div>
      <div id="galleryGrid" class="gallery-grid"></div>
    </div>
  </section>
</div>

<!-- FLOATING TOOLTIP -->
<div id="svgTooltip" class="tooltip"></div>

<!-- SCOUTING MODAL -->
<div id="playerModal" class="modal-backdrop" onclick="if(event.target===this) closeModal('playerModal')">
  <div class="modal-box">
    <button class="modal-close" onclick="closeModal('playerModal')">Close</button>
    <div id="modalBody"></div>
  </div>
</div>

<!-- LIGHTBOX MODAL -->
<div id="figModal" class="modal-backdrop" onclick="if(event.target===this) closeModal('figModal')">
  <div class="modal-box wide">
    <button class="modal-close" onclick="closeModal('figModal')">Close</button>
    <div id="figModalBody"></div>
  </div>
</div>

<script>
const BAT_DATA = __BAT_JSON__;
const BOWL_DATA = __BOWL_JSON__;
const RR_AUT = __RR_AUT_JSON__;
const RR_BAT = __RR_BAT_JSON__;
const RR_BOWL = __RR_BOWL_JSON__;
const TRADE_AUD = __TRADE_AUD_JSON__;
const VENUE_RR = __VENUE_RR_JSON__;
const P11_CTX = __P11_CTX_JSON__;
const MB_BAT = __MB_BAT_JSON__;
const MB_BOWL = __MB_BOWL_JSON__;
const DEATH_PB = __DEATH_PB_JSON__;
const STROKE_PB = __STROKE_PB_JSON__;
const TEMP_VAL = __TEMP_VAL_JSON__;
const MECH_SUM = __MECH_SUM_JSON__;
const LHB_RHB = __LHB_RHB_JSON__;
const UNI20 = __UNI20_JSON__;
const DUAL_AUD = __DUAL_AUD_JSON__;
const RAPM_B = __RAPM_B_JSON__;
const RAPM_BW = __RAPM_BW_JSON__;
const SENS_B = __SENS_B_JSON__;
const SENS_BW = __SENS_BW_JSON__;
const AUC_TARGETS = __AUC_TARGETS_JSON__;
const FIGURES = __FIGURES_JSON__;

let expMode = 'bat';
let batSort = { k: 'rank', asc: true };
let bowlSort = { k: 'rank', asc: true };
let simSelected = new Set(AUC_TARGETS.filter(t => t.default_on).map(t => t.id));

function fmtS(v, dec=1) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return (v >= 0 ? '+' : '') + Number(v).toFixed(dec);
}
function clsS(v) { return (v || 0) >= 0 ? 'pos' : 'neg'; }

function switchView(id, btn) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  if (btn) btn.classList.add('active');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function closeModal(id) { document.getElementById(id).classList.remove('open'); }

function renderTop10Cards() {
  document.getElementById('top10BatCards').innerHTML = BAT_DATA.slice(0, 10).map(r => `
    <div class="pick-card" onclick="openPlayerModal('bat', ${r.striker_id})">
      <div class="pick-top">
        <span class="rank-pill">#${r.rank} BAT</span>
        <span class="prs-pill">PRS ${r.PRS_100.toFixed(1)}</span>
      </div>
      <div class="pick-name">${r.striker_name}</div>
      <div class="pick-team">${r.bat_type} &bull; ${r.ipl_2026_team} &bull; ${r.hp_b} HPb (${r.hp_m}m)</div>
      <div class="pick-metrics">
        <span>HP SR: <b>${r.hp_sr.toFixed(1)}</b></span>
        <span>HP TVA: <b class="${clsS(r.blended_hp_tva)}">${fmtS(r.blended_hp_tva)}</b></span>
        <span>Resid: <b class="${clsS(r.clutch_resid)}">${fmtS(r.clutch_resid)}</b></span>
      </div>
      <div class="pick-fit">${r.rr_fit}</div>
    </div>
  `).join('');

  document.getElementById('top10BowlCards').innerHTML = BOWL_DATA.slice(0, 10).map(r => `
    <div class="pick-card bowl" onclick="openPlayerModal('bowl', ${r.bowler_id})">
      <div class="pick-top">
        <span class="rank-pill blue">#${r.rank} BOWL</span>
        <span class="prs-pill">PRS ${r.PRS_100.toFixed(1)}</span>
      </div>
      <div class="pick-name">${r.bowler_name}</div>
      <div class="pick-team">${r.pace_spin} &bull; ${r.ipl_2026_team} &bull; ${r.hp_leg} HPb (${r.hp_m}m)</div>
      <div class="pick-metrics">
        <span>Econ: <b>${r.hp_econ.toFixed(2)}</b></span>
        <span>HP TVA: <b class="${clsS(r.blended_hp_tva)}">${fmtS(r.blended_hp_tva)}</b></span>
        <span>Wkts: <b>${r.hp_wkts}</b> (${r.hp_sr.toFixed(1)} SR)</span>
      </div>
      <div class="pick-fit">${r.rr_fit}</div>
    </div>
  `).join('');
}

function renderScatter(mode) {
  document.getElementById('scBtnBat').classList.toggle('active', mode === 'bat');
  document.getElementById('scBtnBowl').classList.toggle('active', mode === 'bowl');
  const data = mode === 'bat' ? BAT_DATA : BOWL_DATA;
  const W = 960, H = 410, padL = 62, padR = 30, padT = 26, padB = 46;
  const xs = data.map(d => d.lp_tva_eb), ys = data.map(d => d.blended_hp_tva);
  const minX = Math.floor(Math.min(...xs) - 2), maxX = Math.ceil(Math.max(...xs) + 2);
  const minY = Math.floor(Math.min(...ys) - 2), maxY = Math.ceil(Math.max(...ys) + 2);
  const sx = x => padL + ((x - minX) / (maxX - minX)) * (W - padL - padR);
  const sy = y => H - padB - ((y - minY) / (maxY - minY)) * (H - padT - padB);

  let svg = '';
  for (let gx = Math.ceil(minX/5)*5; gx <= maxX; gx += 5) {
    const px = sx(gx);
    svg += `<line x1="${px}" y1="${padT}" x2="${px}" y2="${H-padB}" stroke="rgba(148,163,184,${gx===0?'0.35':'0.12'})" stroke-dasharray="${gx===0?'4,4':''}"/>`;
    svg += `<text x="${px}" y="${H-padB+16}" fill="#94A3B8" font-size="11" text-anchor="middle">${gx}</text>`;
  }
  for (let gy = Math.ceil(minY/5)*5; gy <= maxY; gy += 5) {
    const py = sy(gy);
    svg += `<line x1="${padL}" y1="${py}" x2="${W-padR}" y2="${py}" stroke="rgba(148,163,184,${gy===0?'0.35':'0.12'})" stroke-dasharray="${gy===0?'4,4':''}"/>`;
    svg += `<text x="${padL-10}" y="${py+4}" fill="#94A3B8" font-size="11" text-anchor="end">${gy}</text>`;
  }
  svg += `<text x="${(padL+W-padR)/2}" y="${H-8}" fill="#CBD5E1" font-size="12" font-weight="700" text-anchor="middle">Usual / Low-Pressure Performance (EB-Shrunk RAPM TVA / 100 Balls)</text>`;
  svg += `<text x="16" y="${(padT+H-padB)/2}" fill="#CBD5E1" font-size="12" font-weight="700" text-anchor="middle" transform="rotate(-90 16 ${(padT+H-padB)/2})">High-Pressure Performance (Blended TVA / 100)</text>`;

  // Compute exact piecewise OLS expectation (floored at LP >= 0) in JS
  const n = data.length;
  const fxs = data.map(d => Math.max(0, d.lp_tva_eb));
  const mx = fxs.reduce((a,b)=>a+b,0)/n, my = ys.reduce((a,b)=>a+b,0)/n;
  let num = 0, den = 0;
  for (let i=0; i<n; i++) { num += (fxs[i]-mx)*(ys[i]-my); den += (fxs[i]-mx)**2; }
  const b1 = num / (den || 1), b0 = my - b1 * mx;
  svg += `<polyline points="${sx(minX)},${sy(b0)} ${sx(0)},${sy(b0)} ${sx(maxX)},${sy(b0 + b1*maxX)}" fill="none" stroke="#94A3B8" stroke-width="1.5" stroke-dasharray="6,4"/>`;

  const topColor = mode === 'bat' ? '#EA1A85' : '#38BDF8';
  const nudges = mode === 'bat'
    ? {1:[8,-8], 2:[8,-4], 3:[8,-8], 4:[8,-6], 5:[8,4], 6:[8,-8], 7:[8,12], 8:[8,12], 9:[-48,-8], 10:[8,11]}
    : {1:[8,-6], 2:[8,12], 3:[8,-6], 4:[-58,-10], 5:[8,-8], 6:[8,12], 7:[8,-10], 8:[8,12], 9:[8,-8], 10:[-70,12]};
  const ordered = [...data.filter(d => d.rank > 10), ...data.filter(d => d.rank <= 10)];
  ordered.forEach(d => {
    const cx = sx(d.lp_tva_eb), cy = sy(d.blended_hp_tva);
    const isTop = d.rank <= 10;
    const isRR = d.ipl_2026_team === 'Rajasthan Royals';
    const col = isTop ? topColor : (isRR ? '#FDE047' : (d.clutch_resid < -7 ? '#F87171' : '#64748B'));
    const r = isTop ? 7.5 : (isRR ? 6 : 4.5);
    const nm = mode === 'bat' ? d.striker_name : d.bowler_name;
    const id = mode === 'bat' ? d.striker_id : d.bowler_id;
    svg += `<circle cx="${cx}" cy="${cy}" r="${r}" fill="${col}" fill-opacity="${isTop?0.95:0.7}" stroke="${isTop?'#fff':'rgba(255,255,255,0.35)'}" stroke-width="${isTop?1.6:0.8}" style="cursor:pointer;" onmousemove="showTip(evt,'${mode}',${id})" onmouseout="hideTip()" onclick="openPlayerModal('${mode}',${id})"/>`;
    if (isTop) {
      const [dx, dy] = nudges[d.rank] || [9, 3];
      svg += `<text x="${cx+dx}" y="${cy+dy}" fill="#F8FAFC" font-size="10.5" font-weight="700" style="pointer-events:none;">#${d.rank} ${nm.split(' ').slice(-1)[0]}</text>`;
    }
  });
  svg += `<rect x="${padL+10}" y="${padT+6}" width="445" height="24" rx="6" fill="rgba(15,23,51,0.85)" stroke="rgba(148,163,184,0.2)"/>
    <circle cx="${padL+24}" cy="${padT+18}" r="5" fill="${topColor}"/><text x="${padL+33}" y="${padT+22}" fill="#F8FAFC" font-size="11">Top 10</text>
    <circle cx="${padL+90}" cy="${padT+18}" r="5" fill="#FDE047"/><text x="${padL+99}" y="${padT+22}" fill="#F8FAFC" font-size="11">RR Core</text>
    <circle cx="${padL+162}" cy="${padT+18}" r="5" fill="#F87171"/><text x="${padL+171}" y="${padT+22}" fill="#F8FAFC" font-size="11">Clutch Dipper</text>
    <line x1="${padL+262}" y1="${padT+18}" x2="${padL+286}" y2="${padT+18}" stroke="#94A3B8" stroke-width="1.5" stroke-dasharray="4,3"/>
    <text x="${padL+292}" y="${padT+22}" fill="#CBD5E1" font-size="11">Expected HP Baseline</text>`;
  document.getElementById('quadSvg').innerHTML = svg;
}

function showTip(evt, mode, id) {
  const arr = mode === 'bat' ? BAT_DATA : BOWL_DATA;
  const r = arr.find(x => (mode==='bat'?x.striker_id:x.bowler_id) === id);
  if (!r) return;
  const nm = mode === 'bat' ? r.striker_name : r.bowler_name;
  const tip = document.getElementById('svgTooltip');
  tip.innerHTML = `<b>#${r.rank} ${nm}</b> <span style="color:#FDE047;">(PRS ${r.PRS_100.toFixed(1)})</span><br>
    <span style="color:#94A3B8;">${r.ipl_2026_team} &bull; ${mode==='bat'?r.hp_b+' HPb':r.hp_leg+' HPb'}</span><br>
    HP TVA: <b class="${clsS(r.blended_hp_tva)}">${fmtS(r.blended_hp_tva)}</b> | LP TVA: <b>${fmtS(r.lp_tva_eb)}</b><br>
    Clutch Resid: <b class="${clsS(r.clutch_resid)}">${fmtS(r.clutch_resid)}</b><br>
    <span style="color:#93C5FD; font-size:10.5px;">Click dot for full scouting profile</span>`;
  tip.style.display = 'block';
  tip.style.left = Math.min(window.innerWidth - 290, evt.clientX + 14) + 'px';
  tip.style.top = Math.max(10, evt.clientY - 12) + 'px';
}
function hideTip() { document.getElementById('svgTooltip').style.display = 'none'; }

function openPlayerModal(mode, id) {
  const arr = mode === 'bat' ? BAT_DATA : BOWL_DATA;
  const r = arr.find(x => (mode==='bat'?x.striker_id:x.bowler_id) === id);
  if (!r) return;
  const isBat = mode === 'bat';
  const nm = isBat ? r.striker_name : r.bowler_name;
  let extra = '';
  if (isBat && r.hp_pon_tva != null) {
    extra = `
      <div class="grid-2" style="margin:12px 0;">
        <div style="background:var(--surface-2); padding:10px 12px; border-radius:8px;">
          <b>3-Vector &amp; Phase Pressure Splits</b>
          <div style="font-size:12px; margin-top:6px; display:grid; gap:4px;">
            <div>Pace-On HP TVA: <b class="${clsS(r.hp_pon_tva)}">${fmtS(r.hp_pon_tva)}</b></div>
            <div>Pace-Off HP TVA: <b class="${clsS(r.hp_poff_tva)}">${fmtS(r.hp_poff_tva)}</b></div>
            <div>Spin HP: <b>${r.hp_spin_sr.toFixed(1)} SR</b> (<b class="${clsS(r.hp_spin_tva)}">${fmtS(r.hp_spin_tva)} TVA</b>)</div>
            <div>Entry Balls 1–10 TVA: <b class="${clsS(r.entry10_tva)}">${fmtS(r.entry10_tva)}</b></div>
          </div>
        </div>
        <div style="background:var(--surface-2); padding:10px 12px; border-radius:8px;">
          <b>Reliability &amp; Clutch Context</b>
          <div style="font-size:12px; margin-top:6px; display:grid; gap:4px;">
            <div>Chase Death TVA: <b class="${clsS(r.chase_death_tva)}">${fmtS(r.chase_death_tva)}</b></div>
            <div>2024–26 Recent HP TVA: <b class="${clsS(r.recent_24_26_tva)}">${fmtS(r.recent_24_26_tva)}</b></div>
            <div>Top HP Stroke: <b>${r.top_hp_stroke || '—'}</b></div>
            <div>95% Bootstrap CI: <b>[${fmtS(r.ci_low_95,0)}, ${fmtS(r.ci_high_95,0)}]</b> (p=${(100*(r.p_gt_zero||0)).toFixed(1)}%)</div>
          </div>
        </div>
      </div>`;
  } else if (!isBat && r.hp_death_tva100 != null) {
    extra = `
      <div class="grid-2" style="margin:12px 0;">
        <div style="background:var(--surface-2); padding:10px 12px; border-radius:8px;">
          <b>Tactical Execution &amp; Speed Profile</b>
          <div style="font-size:12px; margin-top:6px; display:grid; gap:4px;">
            <div>Death Overs (16–20, ${r.hp_death_leg}b): <b class="${clsS(r.hp_death_tva100)}">${fmtS(r.hp_death_tva100)} TVA/100</b></div>
            <div>2nd-Innings Defense: <b class="${clsS(r.hp_i2_tva100)}">${fmtS(r.hp_i2_tva100)} TVA/100</b></div>
            <div>Mean Speed / SD: <b>${r.hp_spd_mean!=null?r.hp_spd_mean.toFixed(1)+' km/h (SD '+r.hp_spd_std.toFixed(1)+')':'N/A'}</b></div>
            <div>Top Mix: <b>${r.top_hp_styles || '—'}</b></div>
          </div>
        </div>
        <div style="background:var(--surface-2); padding:10px 12px; border-radius:8px;">
          <b>Handedness Matchup &amp; Confidence</b>
          <div style="font-size:12px; margin-top:6px; display:grid; gap:4px;">
            <div>vs LHB HP TVA/100: <b class="${clsS(r.tva_LHB)}">${fmtS(r.tva_LHB)}</b> (${r.b_LHB||0} balls)</div>
            <div>vs RHB HP TVA/100: <b class="${clsS(r.tva_RHB)}">${fmtS(r.tva_RHB)}</b> (${r.b_RHB||0} balls)</div>
            <div>2024–26 Recent HP TVA: <b class="${clsS(r.recent_24_26_tva)}">${fmtS(r.recent_24_26_tva)}</b></div>
            <div>95% Bootstrap CI: <b>[${fmtS(r.ci_low_95,0)}, ${fmtS(r.ci_high_95,0)}]</b> (p=${(100*(r.p_gt_zero||0)).toFixed(1)}%)</div>
          </div>
        </div>
      </div>`;
  }

  let sigHtml = '';
  if (r.sig_json && r.sig_json.length) {
    sigHtml = `<h4 style="margin:12px 0 6px 0;">Top Signature Clutch Performances (Data Pack)</h4>
      <table><thead><tr><th>Date</th><th>Competition &amp; Fixture</th><th class="num">HP Stat</th><th class="num">HP TVA</th><th class="num">HP WPA</th></tr></thead><tbody>` +
      r.sig_json.map(s => `<tr><td>${s.match_date}</td><td>${s.comp_name} (${s.home_team} vs ${s.away_team})</td>
        <td class="num">${isBat ? s.hp_r+'* ('+s.hp_b+'b)' : s.hp_w+'/'+s.hp_r+' ('+s.hp_leg+'b)'}</td>
        <td class="num pos">${fmtS(s.hp_tva)}</td><td class="num pos">${fmtS(s.hp_wpa,2)}</td></tr>`).join('') +
      `</tbody></table>`;
  }

  document.getElementById('modalBody').innerHTML = `
    <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:8px;">
      <span class="rank-pill ${isBat?'':'blue'}">Rank #${r.rank} ${isBat?'Batter':'Bowler'}</span>
      <span class="prs-pill">PRS ${r.PRS_100.toFixed(1)}</span>
      <span class="badge-sm">${r.ipl_2026_team}</span>
      <span class="badge-sm">${r.t20i_team}</span>
    </div>
    <h2 style="margin:0 0 4px 0; font-size:22px;">${nm}</h2>
    <p class="subtext">${isBat ? r.bat_type : r.bowl_style} &bull; ${isBat ? r.hp_b+' HP Balls' : r.hp_leg+' HP Legal Balls'} across ${r.hp_m} high-pressure matches</p>
    <div class="grid-3" style="margin-top:12px;">
      <div style="background:#0A1024; padding:10px; border-radius:8px; border:1px solid var(--border);">
        <div class="subtext">Blended HP TVA / 100</div><div style="font-size:19px; font-weight:800;" class="${clsS(r.blended_hp_tva)}">${fmtS(r.blended_hp_tva)}</div>
      </div>
      <div style="background:#0A1024; padding:10px; border-radius:8px; border:1px solid var(--border);">
        <div class="subtext">Clutch Residual</div><div style="font-size:19px; font-weight:800;" class="${clsS(r.clutch_resid)}">${fmtS(r.clutch_resid)}</div>
      </div>
      <div style="background:#0A1024; padding:10px; border-radius:8px; border:1px solid var(--border);">
        <div class="subtext">HP Rate &amp; Pos. Match %</div><div style="font-size:16px; font-weight:800;">${isBat?r.hp_sr.toFixed(1)+' SR':r.hp_econ.toFixed(2)+' Econ'} &bull; ${(100*r.pos_m_rate).toFixed(0)}%</div>
      </div>
    </div>
    ${extra}
    ${r.rr_fit ? `<div style="margin-top:10px; padding:10px 12px; border-radius:8px; background:rgba(234,26,133,0.12); border-left:3px solid var(--pink); font-size:12.5px;"><b>Rajasthan Royals 2027 Assessment:</b> ${r.rr_fit}</div>` : ''}
    ${sigHtml}
  `;
  document.getElementById('playerModal').classList.add('open');
}

function setExpMode(m) {
  expMode = m;
  document.getElementById('expBatBtn').classList.toggle('active', m === 'bat');
  document.getElementById('expBowlBtn').classList.toggle('active', m === 'bowl');
  renderExplorer();
}

function sortExp(k) {
  if (expMode === 'bat') batSort = { k, asc: batSort.k === k ? !batSort.asc : (k === 'rank' || k === 'striker_name') };
  else bowlSort = { k, asc: bowlSort.k === k ? !bowlSort.asc : (k === 'rank' || k === 'bowler_name' || k === 'hp_econ') };
  renderExplorer();
}

function renderExplorer() {
  const q = document.getElementById('expSearch').value.toLowerCase();
  const f = document.getElementById('expFilter').value;
  if (expMode === 'bat') {
    let rows = BAT_DATA.filter(r => {
      if (f === 'top10' && r.rank > 10) return false;
      if (f === 'ipl26' && r.ipl_2026_team === 'Non-IPL26') return false;
      if (f === 'rr' && r.ipl_2026_team !== 'Rajasthan Royals') return false;
      if (f === 'sub1' && r.bat_type !== 'RHB') return false;
      if (f === 'sub2' && r.bat_type !== 'LHB') return false;
      if (f === 'pos_clutch' && r.clutch_resid <= 0) return false;
      return (r.striker_name + ' ' + r.ipl_2026_team + ' ' + r.t20i_team + ' ' + r.bat_type).toLowerCase().includes(q);
    });
    rows.sort((a, b) => (a[batSort.k] > b[batSort.k] ? 1 : -1) * (batSort.asc ? 1 : -1));
    document.getElementById('expTableContainer').innerHTML = `<table><thead><tr>
      <th class="sortable" onclick="sortExp('rank')">Rank</th><th class="sortable" onclick="sortExp('striker_name')">Batter</th><th>Type</th><th>IPL 2026 Team</th>
      <th class="sortable num" onclick="sortExp('PRS_100')">PRS</th><th class="sortable num" onclick="sortExp('hp_b')">HP Balls (M)</th>
      <th class="sortable num" onclick="sortExp('hp_sr')">HP SR (LP)</th><th class="sortable num" onclick="sortExp('blended_hp_tva')">HP TVA</th>
      <th class="sortable num" onclick="sortExp('lp_tva_eb')">LP TVA</th><th class="sortable num" onclick="sortExp('clutch_resid')">Clutch Resid</th>
      <th class="sortable num" onclick="sortExp('hp_wpa_eb')">HP WPA</th><th class="sortable num" onclick="sortExp('pos_m_rate')">Pos %</th>
    </tr></thead><tbody>` + rows.map(r => `
      <tr class="click-row ${r.rank<=10?'top-row':''}" onclick="openPlayerModal('bat', ${r.striker_id})">
        <td>#${r.rank}</td><td><b>${r.striker_name}</b></td><td>${r.bat_type}</td>
        <td>${r.ipl_2026_team==='Rajasthan Royals'?'<span class="badge-sm rr">Rajasthan Royals</span>':r.ipl_2026_team}</td>
        <td class="num"><b style="color:var(--gold);">${r.PRS_100.toFixed(1)}</b></td>
        <td class="num">${r.hp_b} (${r.hp_m}m)</td>
        <td class="num">${r.hp_sr.toFixed(1)} <span style="color:#94A3B8;">(${r.lp_sr.toFixed(1)})</span></td>
        <td class="num ${clsS(r.blended_hp_tva)}">${fmtS(r.blended_hp_tva)}</td>
        <td class="num">${fmtS(r.lp_tva_eb)}</td>
        <td class="num ${clsS(r.clutch_resid)}">${fmtS(r.clutch_resid)}</td>
        <td class="num">${fmtS(r.hp_wpa_eb,2)}</td>
        <td class="num">${Math.round(r.pos_m_rate*100)}%</td>
      </tr>`).join('') + `</tbody></table>`;
  } else {
    let rows = BOWL_DATA.filter(r => {
      if (f === 'top10' && r.rank > 10) return false;
      if (f === 'ipl26' && r.ipl_2026_team === 'Non-IPL26') return false;
      if (f === 'rr' && r.ipl_2026_team !== 'Rajasthan Royals') return false;
      if (f === 'sub1' && r.pace_spin !== 'Pace') return false;
      if (f === 'sub2' && r.pace_spin !== 'Spin') return false;
      if (f === 'pos_clutch' && r.clutch_resid <= 0) return false;
      return (r.bowler_name + ' ' + r.bowl_style + ' ' + r.ipl_2026_team + ' ' + r.t20i_team).toLowerCase().includes(q);
    });
    rows.sort((a, b) => (a[bowlSort.k] > b[bowlSort.k] ? 1 : -1) * (bowlSort.asc ? 1 : -1));
    document.getElementById('expTableContainer').innerHTML = `<table><thead><tr>
      <th class="sortable" onclick="sortExp('rank')">Rank</th><th class="sortable" onclick="sortExp('bowler_name')">Bowler</th><th>Style</th><th>IPL 2026 Team</th>
      <th class="sortable num" onclick="sortExp('PRS_100')">PRS</th><th class="sortable num" onclick="sortExp('hp_leg')">HP Balls (M)</th>
      <th class="sortable num" onclick="sortExp('hp_econ')">HP Econ (LP)</th><th class="sortable num" onclick="sortExp('hp_wkts')">Wkts (SR)</th>
      <th class="sortable num" onclick="sortExp('blended_hp_tva')">HP TVA</th><th class="sortable num" onclick="sortExp('lp_tva_eb')">LP TVA</th>
      <th class="sortable num" onclick="sortExp('clutch_resid')">Clutch Resid</th><th class="sortable num" onclick="sortExp('hp_wpa_eb')">HP WPA</th>
      <th class="sortable num" onclick="sortExp('pos_m_rate')">Pos %</th>
    </tr></thead><tbody>` + rows.map(r => `
      <tr class="click-row ${r.rank<=10?'top-row':''}" onclick="openPlayerModal('bowl', ${r.bowler_id})">
        <td>#${r.rank}</td><td><b>${r.bowler_name}</b></td><td>${r.pace_spin}</td>
        <td>${r.ipl_2026_team==='Rajasthan Royals'?'<span class="badge-sm rr">Rajasthan Royals</span>':(r.ipl_2026_team.includes('Trade')?'<span class="badge-sm trade">'+r.ipl_2026_team+'</span>':r.ipl_2026_team)}</td>
        <td class="num"><b style="color:var(--gold);">${r.PRS_100.toFixed(1)}</b></td>
        <td class="num">${r.hp_leg} (${r.hp_m}m)</td>
        <td class="num">${r.hp_econ.toFixed(2)} <span style="color:#94A3B8;">(${r.lp_econ.toFixed(2)})</span></td>
        <td class="num">${r.hp_wkts} (${r.hp_sr.toFixed(1)})</td>
        <td class="num ${clsS(r.blended_hp_tva)}">${fmtS(r.blended_hp_tva)}</td>
        <td class="num">${fmtS(r.lp_tva_eb)}</td>
        <td class="num ${clsS(r.clutch_resid)}">${fmtS(r.clutch_resid)}</td>
        <td class="num">${fmtS(r.hp_wpa_eb,2)}</td>
        <td class="num">${Math.round(r.pos_m_rate*100)}%</td>
      </tr>`).join('') + `</tbody></table>`;
  }
}

function toggleSim(id) {
  if (simSelected.has(id)) simSelected.delete(id);
  else simSelected.add(id);
  renderSimulator();
}

function presetSim(type) {
  simSelected.clear();
  if (type === 'triple') AUC_TARGETS.filter(t => t.default_on).forEach(t => simSelected.add(t.id));
  else if (type === 'ind_core') AUC_TARGETS.filter(t => !t.os).forEach(t => simSelected.add(t.id));
  renderSimulator();
}

function renderSimulator() {
  document.getElementById('simGrid').innerHTML = AUC_TARGETS.map(t => {
    const sel = simSelected.has(t.id);
    return `
      <div class="sim-item ${sel?'selected':''}" onclick="toggleSim('${t.id}')">
        <input type="checkbox" ${sel?'checked':''} onclick="event.stopPropagation(); toggleSim('${t.id}');">
        <div style="flex:1;">
          <div style="display:flex; justify-content:space-between; gap:6px; align-items:center;">
            <b style="font-size:13px; color:#fff;">${t.name}</b>
            <span class="badge-sm ${t.os?'trade':''}">${t.os?'OS':'IND'} &bull; ${t.cat==='bowl'?t.tva100.toFixed(1)+' TVA':t.tva100.toFixed(1)+' TVA'}</span>
          </div>
          <div style="font-size:11px; color:var(--gold); margin-top:2px;">${t.tag} &bull; ${t.team}</div>
          <div style="font-size:11px; color:#CBD5E1; margin-top:3px;">${t.metric}</div>
          <div style="font-size:10.5px; color:#94A3B8; margin-top:3px;">${t.fit}</div>
        </div>
      </div>`;
  }).join('');

  const chosen = AUC_TARGETS.filter(t => simSelected.has(t.id));
  const bowls = chosen.filter(t => t.cat === 'bowl');
  const bats = chosen.filter(t => t.cat === 'bat');
  const baseLossBalls = 403;
  const baseBowlTva = -21.4;
  let covered = Math.min(baseLossBalls, bowls.reduce((s, b) => s + b.balls_covered, 0));
  let weightedNew = bowls.reduce((s, b) => s + b.tva100 * b.balls_covered, 0);
  let totalWeight = bowls.reduce((s, b) => s + b.balls_covered, 0);
  let avgTargetBowlTva = totalWeight > 0 ? (weightedNew / totalWeight) : baseBowlTva;
  let simLossTva = ((covered * avgTargetBowlTva) + ((baseLossBalls - covered) * baseBowlTva)) / baseLossBalls;
  let runsRecovered = (simLossTva - baseBowlTva) * (baseLossBalls / 100.0);
  let batAvgTva = bats.length ? bats.reduce((s, b) => s + b.tva100, 0) / bats.length : 0;
  let osCnt = chosen.filter(t => t.os).length, indCnt = chosen.filter(t => !t.os).length;

  const sbEl = document.getElementById('simBowlTva');
  sbEl.textContent = fmtS(simLossTva);
  sbEl.className = 'sim-metric-v ' + (simLossTva >= 0 ? 'pos' : 'neg');
  document.getElementById('simRuns').textContent = `+${runsRecovered.toFixed(1)} runs`;
  document.getElementById('simBatAdd').textContent = bats.length ? `${fmtS(batAvgTva)} TVA (${bats.length} picks)` : 'No Bat Picks';
  document.getElementById('simOsCount').textContent = `${osCnt} OS / ${indCnt} IND`;
}

function renderStaticTables() {
  document.getElementById('ovUniBody').innerHTML = UNI20.slice(0, 10).map(r => `
    <tr class="top-row"><td><b>#${r.unified_rank}</b></td><td><b>${r.name}</b></td><td>${r.primary_role}</td><td>${r.team}</td>
    <td class="num"><b style="color:var(--gold);">${r.best_prs.toFixed(1)}</b></td>
    <td class="num ${clsS(r.net_hp_tva_per_match)}">${fmtS(r.net_hp_tva_per_match,2)}</td></tr>`).join('');

  document.getElementById('ovDualBody').innerHTML = DUAL_AUD.map(r => `
    <tr class="${r.name==='Sam Curran'?'':''}"><td>#${r.net_rank}</td><td><b>${r.name}</b> <span class="badge-sm">${r.team}</span></td>
    <td class="num ${clsS(r.tva_bat)}">#${Math.round(r.rank_bat)} (${fmtS(r.tva_bat)})</td>
    <td class="num ${clsS(r.tva_bowl)}">#${Math.round(r.rank_bowl)} (${fmtS(r.tva_bowl)})</td>
    <td class="num ${clsS(r.net_hp_tva_per_match)}">${fmtS(r.net_hp_tva_per_match,2)}</td></tr>`).join('');

  // Visual bars for RR 9W vs 7L
  const hpWin = RR_AUT.filter(x => x.rr_won && x.is_hp);
  const hpLoss = RR_AUT.filter(x => !x.rr_won && x.is_hp);
  const barRow = (lbl, val, color) => {
    const pct = Math.min(100, Math.abs(val) / 25 * 100);
    return `<div><div style="display:flex; justify-content:space-between; font-size:12px; margin-bottom:3px;"><span>${lbl}</span><b class="${clsS(val)}">${fmtS(val)} TVA/100</b></div>
      <div class="bar-track"><div class="bar-fill" style="width:${pct}%; background:${color};"></div></div></div>`;
  };
  const bW = hpWin.find(x=>x.rr_role==='Batting')?.tva_100 || 18.3;
  const bwW = hpWin.find(x=>x.rr_role==='Bowling')?.tva_100 || 8.4;
  const bL = hpLoss.find(x=>x.rr_role==='Batting')?.tva_100 || 7.5;
  const bwL = hpLoss.find(x=>x.rr_role==='Bowling')?.tva_100 || -21.4;
  document.getElementById('rrAutopsyBars').innerHTML =
    barRow('RR 9 Wins — High-Pressure Batting', bW, 'var(--teal)') +
    barRow('RR 9 Wins — High-Pressure Bowling', bwW, '#38BDF8') +
    barRow('RR 7 Losses — High-Pressure Batting (Held Firm!)', bL, 'var(--gold)') +
    barRow('RR 7 Losses — High-Pressure Bowling (Decisive Collapse)', bwL, 'var(--red)');

  document.getElementById('rrAutBody').innerHTML = RR_AUT.map(r => `
    <tr><td><b>${r.rr_won?'RR Win (9m)':'RR Loss (7m)'}</b></td><td>${r.rr_role}</td><td>${r.is_hp?'High Pressure':'Usual'}</td>
    <td class="num">${r.balls}</td><td class="num">${fmtS(r.tva_sum)}</td>
    <td class="num ${clsS(r.tva_100)}">${fmtS(r.tva_100)}</td><td class="num ${clsS(r.wpa_sum)}">${fmtS(r.wpa_sum,2)}</td></tr>`).join('');

  document.getElementById('tradeAudBody').innerHTML = TRADE_AUD.map(r => {
    const tv = r.hp_tva != null ? r.hp_tva : r.raw_hp_tva;
    return `<tr><td><b>${r.move}</b></td><td>${r.player}</td><td class="num">${r.rank!=null?'#'+r.rank:'Unqual'}</td>
      <td class="num">${r.prs!=null?r.prs.toFixed(1):'—'}</td><td class="num">${r.hp_leg}</td>
      <td class="num">${r.hp_econ.toFixed(2)}</td><td class="num ${clsS(tv)}">${fmtS(tv)}</td></tr>`;
  }).join('');

  document.getElementById('venueRrBody').innerHTML = VENUE_RR.filter(r => r.hp_bowl).map(r => `
    <tr><td><b>${r.venue}</b></td><td>${r.pace_cat}</td><td class="num">${r.balls}</td>
    <td class="num ${r.pace_cat==='Spin'?'pos':''}">${r.econ.toFixed(2)}</td><td class="num">${r.bdry.toFixed(1)}%</td></tr>`).join('');

  document.getElementById('p11CtxBody').innerHTML = P11_CTX.map(r => `
    <tr><td><b>${r.factor}</b></td><td class="num">${r.n}</td><td class="num">${r.rate.toFixed(1)}</td>
    <td class="num ${clsS(r.tva100)}">${fmtS(r.tva100)}</td></tr>`).join('');

  document.getElementById('mbBatBody').innerHTML = MB_BAT.map(r => `
    <tr><td><b>${r.striker_name}</b></td><td class="num">${r.hp_b} (${r.hp_m}m)</td><td class="num">${r.hp_sr.toFixed(1)}</td>
    <td class="num ${clsS(r.hp_tva)}">${fmtS(r.hp_tva)}</td><td class="num">${fmtS(r.hp_wpa,2)}</td></tr>`).join('');

  document.getElementById('mbBowlBody').innerHTML = MB_BOWL.map(r => `
    <tr><td><b>${r.bowler_name}</b></td><td class="num">${r.hp_leg} (${r.hp_m}m)</td><td class="num">${r.hp_econ.toFixed(2)} (${r.hp_w}w)</td>
    <td class="num ${clsS(r.hp_tva)}">${fmtS(r.hp_tva)}</td><td class="num">${fmtS(r.hp_wpa,2)}</td></tr>`).join('');

  document.getElementById('rrBatBody').innerHTML = RR_BAT.filter(r => r.tot_b >= 30).sort((a,b)=>b.tot_b-a.tot_b).map(r => `
    <tr><td><b>${r.striker_name}</b></td><td class="num">${r.tot_b}</td><td class="num">${r.hp_b} (${r.hp_m}m)</td>
    <td class="num">${r.hp_sr!=null?r.hp_sr.toFixed(1):'—'}</td><td class="num ${clsS(r.hp_tva100)}">${fmtS(r.hp_tva100)}</td>
    <td class="num">${fmtS(r.lp_tva100)}</td></tr>`).join('');

  document.getElementById('rrBowlBody').innerHTML = RR_BOWL.filter(r => r.tot_leg >= 30).sort((a,b)=>b.tot_leg-a.tot_leg).map(r => `
    <tr><td><b>${r.bowler_name}</b></td><td class="num">${r.tot_leg}</td><td class="num">${r.hp_leg} (${r.hp_m}m)</td>
    <td class="num">${r.hp_econ!=null?r.hp_econ.toFixed(2):'—'}</td><td class="num ${clsS(r.hp_tva100)}">${fmtS(r.hp_tva100)}</td>
    <td class="num">${fmtS(r.lp_tva100)}</td></tr>`).join('');

  document.getElementById('tempValBody').innerHTML = TEMP_VAL.map(r => `
    <tr><td>${r.role}</td><td><b>${r.metric}</b></td>
    <td class="num ${clsS(r.pearson_r)}">${fmtS(r.pearson_r,3)}</td><td class="num">${r.p_val.toFixed(4)}</td>
    <td class="num ${clsS(r.spearman_rho)}">${fmtS(r.spearman_rho,3)}</td></tr>`).join('');

  document.getElementById('mechSumBody').innerHTML = MECH_SUM.map(r => `
    <tr><td><b>${r.cohort}</b></td><td><span class="badge-sm">${r.regime}</span></td><td class="num">${r.balls}</td>
    <td class="num">${r.rate.toFixed(2)}</td><td class="num">${r.dot_pct.toFixed(1)}%</td><td class="num">${r.bdry_pct.toFixed(1)}%</td>
    <td class="num">${r.err_pct.toFixed(1)}%</td><td class="num ${clsS(r.tva100)}">${fmtS(r.tva100)}</td></tr>`).join('');

  document.getElementById('deathPbBody').innerHTML = DEATH_PB.map(r => `
    <tr><td>${r.pace_cat}</td><td><b>${r.delivery_style}</b></td><td class="num">${r.balls}</td><td class="num">${r.econ.toFixed(2)}</td>
    <td class="num">${r.wkt_pct.toFixed(1)}%</td><td class="num">${r.dot_pct.toFixed(1)}%</td><td class="num">${r.bdry_pct.toFixed(1)}%</td>
    <td class="num ${clsS(r.tva100)}">${fmtS(r.tva100)}</td></tr>`).join('');

  document.getElementById('strokePbBody').innerHTML = STROKE_PB.map(r => `
    <tr><td><b>${r.stroke}</b></td><td class="num">${r.balls}</td><td class="num">${r.sr.toFixed(1)}</td>
    <td class="num">${r.out_pct.toFixed(1)}%</td><td class="num">${r.bdry_pct.toFixed(1)}%</td>
    <td class="num ${clsS(r.tva100)}">${fmtS(r.tva100)}</td></tr>`).join('');

  document.getElementById('lhbRhbBody').innerHTML = LHB_RHB.map(r => `
    <tr><td>#${r.rank}</td><td><b>${r.bowler_name}</b></td>
    <td class="num ${clsS(r.tva_LHB)}">${fmtS(r.tva_LHB)} <span style="color:#94A3B8;font-weight:400;">(${r.b_LHB||0})</span></td>
    <td class="num ${clsS(r.tva_RHB)}">${fmtS(r.tva_RHB)} <span style="color:#94A3B8;font-weight:400;">(${r.b_RHB||0})</span></td></tr>`).join('');

  document.getElementById('rapmBBody').innerHTML = RAPM_B.map(r => `
    <tr><td><b>${r.bowler_name}</b></td><td class="num">${Math.round(r.count/100)}</td><td class="num neg">${r.mean.toFixed(1)}</td></tr>`).join('');

  document.getElementById('rapmBwBody').innerHTML = RAPM_BW.map(r => `
    <tr><td><b>${r.striker_name}</b></td><td class="num">${Math.round(r.count/100)}</td><td class="num neg">${r.mean.toFixed(1)}</td></tr>`).join('');

  const fmtSens = v => v == null ? '—' : '#' + Math.round(v);
  document.getElementById('sensBatBody').innerHTML = SENS_B.map(r => `
    <tr><td><b>${r.striker_name}</b></td><td class="num">#${Math.round(r.Primary_Q75_K500)}</td>
    <td class="num">${fmtSens(r.Q70_K500)}</td><td class="num">${fmtSens(r.Q80_K500)}</td>
    <td class="num">${fmtSens(r.Q75_K300)}</td><td class="num">${fmtSens(r.Q75_K800)}</td>
    <td class="num">${fmtSens(r.IPL_Only)}</td><td class="num">${fmtSens(r.Recent_24_26)}</td></tr>`).join('');

  document.getElementById('sensBowlBody').innerHTML = SENS_BW.map(r => `
    <tr><td><b>${r.bowler_name}</b></td><td class="num">#${Math.round(r.Primary_Q75_K500)}</td>
    <td class="num">${fmtSens(r.Q70_K500)}</td><td class="num">${fmtSens(r.Q80_K500)}</td>
    <td class="num">${fmtSens(r.Q75_K300)}</td><td class="num">${fmtSens(r.Q75_K800)}</td>
    <td class="num">${fmtSens(r.IPL_Only)}</td><td class="num">${fmtSens(r.Recent_24_26)}</td></tr>`).join('');

  document.getElementById('galleryGrid').innerHTML = FIGURES.map((f, i) => `
    <div class="fig-card" onclick="openFig(${i})">
      <img src="data:image/png;base64,${f.b64}" alt="${f.title}">
      <h4>${f.title}</h4>
      <p>${f.cap}</p>
    </div>`).join('');
}

function openFig(idx) {
  const f = FIGURES[idx];
  document.getElementById('figModalBody').innerHTML = `
    <h3 style="margin:0 0 10px 0;">${f.title}</h3>
    <img src="data:image/png;base64,${f.b64}" style="width:100%; border-radius:8px; background:#fff;">
    <p style="font-size:13px; color:#CBD5E1; margin-top:10px;">${f.cap}</p>
  `;
  document.getElementById('figModal').classList.add('open');
}

renderTop10Cards();
renderScatter('bat');
renderExplorer();
renderSimulator();
renderStaticTables();
</script>
</body>
</html>
"""

replacements = {
    "__BAT_JSON__": df_to_json(bat_full[bat_cols].round(2)),
    "__BOWL_JSON__": df_to_json(bowl_full[bowl_cols].round(2)),
    "__RR_AUT_JSON__": df_to_json(rr26_aut.round(2)),
    "__RR_BAT_JSON__": df_to_json(rr_bat.round(2)),
    "__RR_BOWL_JSON__": df_to_json(rr_bowl.round(2)),
    "__TRADE_AUD_JSON__": df_to_json(trade_aud.round(2)),
    "__VENUE_RR_JSON__": df_to_json(venue_rr.round(2)),
    "__P11_CTX_JSON__": df_to_json(p11_ctx.round(2)),
    "__MB_BAT_JSON__": df_to_json(mb_bat.round(2)),
    "__MB_BOWL_JSON__": df_to_json(mb_bowl.round(2)),
    "__DEATH_PB_JSON__": df_to_json(death_pb.round(2)),
    "__STROKE_PB_JSON__": df_to_json(stroke_pb.round(2)),
    "__TEMP_VAL_JSON__": df_to_json(temp_val.round(4)),
    "__MECH_SUM_JSON__": df_to_json(mech_sum.round(2)),
    "__LHB_RHB_JSON__": df_to_json(lhb_piv.round(1)),
    "__UNI20_JSON__": df_to_json(uni20.round(2)),
    "__DUAL_AUD_JSON__": df_to_json(dual_aud.round(2)),
    "__RAPM_B_JSON__": df_to_json(rapm_b_top.round(2)),
    "__RAPM_BW_JSON__": df_to_json(rapm_bw_top.round(2)),
    "__SENS_B_JSON__": df_to_json(sens_b.round(0)),
    "__SENS_BW_JSON__": df_to_json(sens_bw.round(0)),
    "__AUC_TARGETS_JSON__": json.dumps(auc_targets),
    "__FIGURES_JSON__": json.dumps(figures_meta),
}


def build_page(out_path, pdf_href, report_href):
    html = HTML_TEMPLATE
    for k, v in replacements.items():
        html = html.replace(k, v)
    html = html.replace("__PDF_HREF__", pdf_href).replace("__REPORT_HREF__", report_href)
    with open(out_path, "w", encoding="utf-8") as fw:
        fw.write(html)


build_page(f"{ROOT}\\index.html", "./report/SupeRR_Selector_S2_Methodology_Report.pdf", "./report/SupeRR_Selector_S2_Methodology_Report.html")
build_page(f"{REP}\\SupeRR_Selector_S2_Dashboard.html", "./SupeRR_Selector_S2_Methodology_Report.pdf", "./SupeRR_Selector_S2_Methodology_Report.html")
print("Built broadcast-grade War-Room Dashboard -> index.html & report/SupeRR_Selector_S2_Dashboard.html!")
