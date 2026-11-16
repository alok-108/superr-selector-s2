import os
import json
import base64
import pandas as pd
import numpy as np

OUT = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\outputs'
REP = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\report'
FIG = f'{REP}/figures'

with open(f'{OUT}/summary_metrics.json') as f:
    meta = json.load(f)

top10_b = pd.read_csv(f'{OUT}/top10_batters.csv')
top10_bw = pd.read_csv(f'{OUT}/top10_bowlers.csv')
ebat = pd.read_csv(f'{OUT}/eligible_batters.csv')
ebowl = pd.read_csv(f'{OUT}/eligible_bowlers.csv')
micro_b = pd.read_csv(f'{OUT}/micro_tactics_batters.csv')
micro_bw = pd.read_csv(f'{OUT}/micro_tactics_bowlers.csv')
sig_b = pd.read_csv(f'{OUT}/signature_bat_innings.csv')
sig_bw = pd.read_csv(f'{OUT}/signature_bowl_spells.csv')
sens_b = pd.read_csv(f'{OUT}/sensitivity_batters.csv')
sens_bw = pd.read_csv(f'{OUT}/sensitivity_bowlers.csv')
wp_val = pd.read_csv(f'{OUT}/wp_validation_metrics.csv')
sit_val = pd.read_csv(f'{OUT}/situation_validation.csv')
death_pb = pd.read_csv(f'{OUT}/death_delivery_playbook.csv')
stroke_pb = pd.read_csv(f'{OUT}/hp_stroke_playbook.csv')
rapm_b_top = pd.read_csv(f'{OUT}/rapm_toughest_bowlers.csv')
rapm_bw_top = pd.read_csv(f'{OUT}/rapm_toughest_batters.csv')
rr26_aut = pd.read_csv(f'{OUT}/rr26_win_loss_autopsy.csv')
rr_bat = pd.read_csv(f'{OUT}/rr_roster_bat.csv')
rr_bowl = pd.read_csv(f'{OUT}/rr_roster_bowl.csv')
dip_b = pd.read_csv(f'{OUT}/batters_biggest_dip.csv')
dip_bw = pd.read_csv(f'{OUT}/bowlers_biggest_dip.csv')
ipl_b = pd.read_csv(f'{OUT}/ipl_only_top15_batters.csv')
ipl_bw = pd.read_csv(f'{OUT}/ipl_only_top15_bowlers.csv')
tier_s = pd.read_csv(f'{OUT}/tier_season_summary.csv')
temp_val = pd.read_csv(f'{OUT}/temporal_prediction_benchmark.csv')
mech_sum = pd.read_csv(f'{OUT}/clutch_mechanisms_summary.csv')
venue_rr = pd.read_csv(f'{OUT}/venue_jaipur_guwahati_hp.csv')
lhb_rhb = pd.read_csv(f'{OUT}/bowler_lhb_rhb_hp.csv')
trade_aud = pd.read_csv(f'{OUT}/front_office_trade_audit.csv')
p11_ctx = pd.read_csv(f'{OUT}/p11_context_effects.csv')
mb_bat = pd.read_csv(f'{OUT}/moneyball_batters_radar.csv')
mb_bowl = pd.read_csv(f'{OUT}/moneyball_bowlers_radar.csv')
dual_aud = pd.read_csv(f'{OUT}/dual_allrounder_audit.csv')
uni20 = pd.read_csv(f'{OUT}/unified_top20_prs.csv')

def img_b64(fname):
    with open(f'{FIG}/{fname}', 'rb') as fh:
        return base64.b64encode(fh.read()).decode('ascii')

def disp_name(nm):
    return 'Namboori Tilak Varma (Tilak Varma)' if 'Tilak' in nm else nm

def short_name(nm):
    return 'Tilak Varma' if 'Tilak' in nm else nm

def fmt_opt(val, suffix=''):
    return f"{val:+.1f}{suffix}" if pd.notna(val) else "N/A"

# Merge micro tactics & top signature match for Top 10 Batters
tb = top10_b.merge(micro_b, on=['striker_id','striker_name'], suffixes=('', '_micro'))
sig_b_top1 = sig_b.sort_values('hp_tva', ascending=False).groupby('striker_id').first().reset_index()
tb = tb.merge(sig_b_top1[['striker_id','match_date','comp_name','home_team','away_team','hp_b','hp_r','hp_tva','hp_wpa']],
              on='striker_id', suffixes=('', '_sig'))

tbw = top10_bw.merge(micro_bw, on=['bowler_id','bowler_name'], suffixes=('', '_micro'))
sig_bw_top1 = sig_bw.sort_values('hp_tva', ascending=False).groupby('bowler_id').first().reset_index()
tbw = tbw.merge(sig_bw_top1[['bowler_id','match_date','comp_name','home_team','away_team','hp_leg','hp_r','hp_w','hp_tva','hp_wpa']],
                on='bowler_id', suffixes=('', '_sig'))

# Custom analytical scouting blurbs grounded strictly in RAPM computed data
bat_notes = {
    'Suryakumar Yadav': 'Clear #1 across combined & IPL-only models (100% bootstrap p>0, 66% pos. match rate; +45.1 Pace-On, +91.3 Pace-Off, +35.8 Spin, +29.3 Entry-10; 4* & 3* Super Overs). <b>RR Fit:</b> Retained MI archetype template for RR’s young #3/#4 core (Jaiswal/Parag) vs Jaipur spin.',
    'Tim David': 'Premier Pace-On finisher (197.0 HP SR / +51.7 TVA vs Pace-On) with dataset-high +0.40 shrunk WPA/100 (p>0=98.8%; +34.1 in 2024–26 despite -8.6 Pace-Off friction). <b>RR Fit:</b> Retained RCB power finisher benchmark; Pace-Off dip warns RR against one-dimensional hitters at #6.',
    'Namboori Tilak Varma': 'Highest HP workload among all 73 batters (810b, 65m; 43.1 HP Avg) with #1 Entry-10 impact (+31.9 TVA across 324b), +49.4 vs Pace-Off, and p>0=99.9%. <b>RR Fit:</b> Retained MI anchor benchmark alongside Riyan Parag to stabilize early top-order collapses.',
    'Shreyas Iyer': 'Middle-order spin destroyer (160.8 HP SR / +54.3 TVA vs Spin) and #1 recent-era (2024–26) batter (+38.8 TVA; +87.4 Chase Death; 75 HP runs in IPL 2025 Q2). <b>RR Fit:</b> PBKS captain benchmark showcasing +7.16 leadership HP lift & spin dominance for RR’s middle order.',
    'Shashank Singh': '#4 IPL-only breakout (162.4 HP SR overall / 166.1 IPL, 51.3 Avg, +0.30 WPA/100) obliterating Pace-Off (+85.2 TVA/100) and +25.2 in 2024–26. <b>RR Fit:</b> Unlikely PBKS release, yet ideal blueprint (+85.2 Pace-Off) for RR’s uncapped #6/#7 radar (Ashutosh/Aniket Verma).',
    'David Miller': 'Archetypal pressure lifter (+9.4 clutch resid above +2.1 LP; 158.9 HP SR across 51m), crushing Spin (+47.5 TVA) and Chase Death (+42.1 TVA, p>0=95.7%). <b>RR Fit:</b> <b>Top OS Finisher Target</b>—directly upgrades Hetmyer’s #5/#6 closing slot (113.0 SR in IPL 2026).',
    'Dinesh Karthik': 'Proven IPL 2022–24 closer (376 HPb, 40m, 160.4 SR, +35.5 I2 HP TVA; +29.0 Pace-Off). Retired after IPL 2024; included strictly on 2022–26 data (p>0=92.4%). <b>RR Fit:</b> Retired historical blueprint for domestic finisher development (Jurel/Ferreira/Ashutosh).',
    'Heinrich Klaasen': '#3 IPL-only pressure star (165.0 IPL HP SR, 70.4% pos. IPL matches) and Spin crusher (+54.1 TVA; +30.8 in 2024–26), holding +11.9 against +22.5 LP. <b>RR Fit:</b> <b>Primary Release-Watch Closer</b>—elite Spin-hitting WK transforming RR’s Jaipur/Guwahati middle overs.',
    'Dasun Shanaka': 'High-volume crunch surge (+6.7 clutch resid above +0.2 LP; 448b, 43m, p>0=91.8%), punishing Pace-On (+31.1 TVA) & chases (+25.9 I2 TVA). <b>RR Fit:</b> <b>Current RR Retainee</b>—affordable OS finisher & emergency seam option delivering proven crunch chase poise.',
    'Ishan Kishan': 'Enters Top 10 after RAPM normalization (165.7 HP SR, 60% pos. matches, p>0=92.3%), pairing +16.0 Entry-10 speed with +59.0 Spin & +53.1 Pace-Off. <b>RR Fit:</b> <b>LHB Top/Middle WK Target (if released)</b>—adds instant powerplay/middle-over acceleration behind Jaiswal.',
}

bowl_notes = {
    'Krunal Pandya': '#1 across combined & IPL-only models (98.7% p>0); cuts Econ 7.97→7.18, spikes dots to 35.4%, dominates I2 defense (+61.2 TVA/100, 70% Orthodox/24% Arm). <b>RR Fit:</b> <b>#1 Auction Spin-AR Target</b>—exploits Jaipur (9.38) & Guwahati (8.12) HP spin edge as an Indian overs 7–16 controller.',
    'Axar Patel': 'Elite middle/death squeeze (+13.7 clutch resid; 7.68 HP Econ, +21.0 Death TVA across 85b; 28 wkts, +0.26 WPA/100, p>0=97.2%). <b>RR Fit:</b> Retained DC all-round benchmark (+1.40 net HP runs/m) proving why non-OS left-arm spin-ARs win crunch chases.',
    'Bhuvneshwar Kumar': 'Dataset-high 723 HPb across 71m (45 wkts, 60.6% pos. rate, 97.5% p>0), scaling Pace-Off 15.4%→22.0% for +19.0 Death TVA (+24.5 IPL). <b>RR Fit:</b> RCB veteran benchmark illustrating how heavy HP Pace-Off deployment complements Jofra Archer’s raw speed.',
    'Sunil Narine': 'Lowest HP Econ in Top 10 (6.95 across 352b, 37.9% dots, +62.1 Death TVA across 86b) via 50/50 Off-Break/Carrom; sealed KKR’s 2026 Super Over (1r, 2w). <b>RR Fit:</b> Retained KKR mystery-spin template confirming defensive choke value (7.0 Econ) in tight chases.',
    'Josh Hazlewood': 'Precision hard-length executor (449 HPb, 45m, Top-10-best 40.9% dots, +0.31 WPA/100), #3 in 2024–26 recent bowling (+21.0; +37.6 T20I). <b>RR Fit:</b> Retained RCB pace benchmark demonstrating hard-length dot pressure opposite Archer & Mohammed Shami.',
    'Jasprit Bumrah': '#1 recent-era (2024–26) bowler (+27.4 TVA) with unmatched death mastery (330/580 HPb at Death for +50.1 TVA, 7.46 Econ, 98.8% p>0, 12% Yorker). <b>RR Fit:</b> Retained MI gold-standard closer (+50.1 Death TVA)—target tactical partner archetype for Archer.',
    'Nathan Ellis': 'Deception-driven death specialist (44.1% HP Pace-Off vs 33.3% LP, 11.1 km/h SD, 12.5 SR, 69% pos. rate, +51.6 vs LHB, 99.1% p>0). <b>RR Fit:</b> <b>Traded CSK→SRH (Nov 2026)</b>—was our #1 Pace-Off death fit; locks market & triggers Goa pivot to Pathirana/Ngidi.',
    'Rashid Khan': 'Workhorse wrist-spinner (587 HPb, 66m, 41 wkts @ 14.3 SR, 8.10 Econ) turning 50/48 Googly–Leg Break into +28.2 Death TVA across 250b. <b>RR Fit:</b> Retained GT icon—sole Top-10 bowler also Top-25 with the bat (+1.29 net HP runs/m) for two-way balance.',
    'Kuldeep Yadav': '#2 IPL-only bowler (7.62 IPL HP Econ, 78.3% pos. IPL rate, +35.6 IPL TVA), accelerating under pressure (87.9 km/h) with 59% Chinaman / 31% Googly. <b>RR Fit:</b> <b>Priority Indian Strike Wrist-Spinner</b> (+54.2 vs LHB) to cure RR’s middle-over wicket drought.',
    'Wanindu Hasaranga': 'Strike wrist-spinner (31 HP wkts @ 12.9 SR across 39m; 52% Leg-Break / 46% Googly) surging to +40.2 IPL HP TVA (+0.23 WPA, p>0=96%). <b>RR Fit:</b> <b>Ex-RR Wrist-Spin Value Target</b>—proven wicket-taker for Jaipur/Guwahati (deploy strictly below #7 batting).',
}

# Build Top 10 Batters table rows HTML
bat_rows_html = ""
for _, r in tb.iterrows():
    if r['ipl_2026_team'] != 'Non-IPL26':
        ipl_badge = r['ipl_2026_team']
    elif 'Karthik' in r['striker_name']:
        ipl_badge = f"Retired (Last IPL: {int(r['last_ipl_season'])})"
    else:
        ipl_badge = f"Non-IPL26 (Last IPL: {int(r['last_ipl_season'])})" if pd.notna(r['last_ipl_season']) else "T20I Only"
    p_str = "100%" if r['p_gt_zero'] >= 0.9995 else (f"{100*r['p_gt_zero']:.1f}%" if r['p_gt_zero'] >= 0.99 else f"{100*r['p_gt_zero']:.0f}%")
    bat_rows_html += f"""
    <tr>
      <td class="rank-cell">#{int(r['rank'])}</td>
      <td><strong>{disp_name(r['striker_name'])}</strong><br><span class="sub">{r['bat_type']} | {ipl_badge}</span></td>
      <td class="num"><strong>{r['PRS_100']:.1f}</strong></td>
      <td class="num">{int(r['hp_b'])} <span class="sub">({int(r['hp_m'])}m)</span></td>
      <td class="num">{r['hp_sr']:.1f} <span class="sub">({r['lp_sr']:.1f})</span></td>
      <td class="num pos">+{r['blended_hp_tva']:.1f}</td>
      <td class="num">{r['lp_tva_eb']:+.1f}</td>
      <td class="num pos">+{r['clutch_resid']:.1f}</td>
      <td class="num">+{r['hp_wpa_eb']:.2f}</td>
      <td class="num">{100*r['pos_m_rate']:.0f}% <span class="sub">(p={p_str})</span></td>
    </tr>"""

bowl_rows_html = ""
for _, r in tbw.iterrows():
    if r['bowler_name'] == 'Nathan Ellis':
        ipl_badge = "Chennai Super Kings (→SRH Trade)"
    else:
        ipl_badge = r['ipl_2026_team'] if r['ipl_2026_team'] != 'Non-IPL26' else "Non-IPL26"
    p_str = "100%" if r['p_gt_zero'] >= 0.9995 else (f"{100*r['p_gt_zero']:.1f}%" if r['p_gt_zero'] >= 0.99 else f"{100*r['p_gt_zero']:.0f}%")
    bowl_rows_html += f"""
    <tr>
      <td class="rank-cell blue">#{int(r['rank'])}</td>
      <td><strong>{r['bowler_name']}</strong><br><span class="sub">{r['bowl_style']} | {ipl_badge}</span></td>
      <td class="num"><strong>{r['PRS_100']:.1f}</strong></td>
      <td class="num">{int(r['hp_leg'])} <span class="sub">({int(r['hp_m'])}m)</span></td>
      <td class="num">{r['hp_econ']:.2f} <span class="sub">({r['lp_econ']:.2f})</span></td>
      <td class="num">{int(r['hp_wkts'])} <span class="sub">({r['hp_sr']:.1f} SR)</span></td>
      <td class="num pos">+{r['blended_hp_tva']:.1f}</td>
      <td class="num">{r['lp_tva_eb']:+.1f}</td>
      <td class="num pos">+{r['clutch_resid']:.1f}</td>
      <td class="num">+{r['hp_wpa_eb']:.2f}</td>
      <td class="num">{100*r['pos_m_rate']:.0f}% <span class="sub">(p={p_str})</span></td>
    </tr>"""

# Micro profiles cards HTML
bat_cards_html = ""
for _, r in tb.iterrows():
    nm = r['striker_name']
    bat_cards_html += f"""
    <div class="profile-card">
      <div class="pc-head"><span class="badge pink">Rank #{int(r['rank'])}</span> <strong>{disp_name(nm)}</strong> <span class="pc-score">PRS {r['PRS_100']:.1f}</span></div>
      <div class="pc-stats">
        <span><b>Pace-On / Off:</b> {r['hp_pon_tva']:+.1f} / {r['hp_poff_tva']:+.1f} TVA</span>
        <span><b>Spin HP:</b> {r['hp_spin_sr']:.1f} SR ({r['hp_spin_tva']:+.1f} TVA)</span>
        <span><b>Entry-10 / 24–26:</b> {r['entry10_tva']:+.1f} / {fmt_opt(r['recent_24_26_tva'])}</span>
        <span><b>Top Stroke / CI:</b> {r['top_hp_stroke']} ([{r['ci_low_95']:+.0f}, {r['ci_high_95']:+.0f}])</span>
      </div>
      <p class="pc-note">{bat_notes[nm]}</p>
    </div>"""

bowl_cards_html = ""
for _, r in tbw.iterrows():
    nm = r['bowler_name']
    spd_txt = f"{r['hp_spd_mean']:.1f} km/h (σ={r['hp_spd_std']:.1f})" if pd.notna(r['hp_spd_mean']) else "N/A"
    poff_txt = f" | POFF {r['hp_pace_off_pct']:.0f}%" if r['pace_spin'] == 'Pace' else ""
    bowl_cards_html += f"""
    <div class="profile-card">
      <div class="pc-head"><span class="badge blue">Rank #{int(r['rank'])}</span> <strong>{nm}</strong> <span class="pc-score">PRS {r['PRS_100']:.1f}</span></div>
      <div class="pc-stats">
        <span><b>HP Dot/Bdry:</b> {r['hp_dot_pct']:.1f}% / {r['hp_bdry_pct']:.1f}%{poff_txt}</span>
        <span><b>Death ({int(r['hp_death_leg'])}b) / 24–26:</b> {r['hp_death_tva100']:+.1f} / {fmt_opt(r['recent_24_26_tva'])}</span>
        <span><b>Speed:</b> {spd_txt}</span>
        <span><b>Mix:</b> {r['top_hp_styles']}</span>
      </div>
      <p class="pc-note">{bowl_notes[nm]}</p>
    </div>"""

# Build Trade Audit table rows HTML
trade_rows_html = ""
for _, r in trade_aud.iterrows():
    rk_str = f"#{int(r['rank'])}" if pd.notna(r['rank']) else "Unqualified"
    prs_str = f"{r['prs']:.1f}" if pd.notna(r['prs']) else "— (<400b)"
    tva_val = r['hp_tva'] if pd.notna(r['hp_tva']) else r['raw_hp_tva']
    cls_t = 'pos' if tva_val >= 0 else 'neg'
    trade_rows_html += f"""
    <tr>
      <td><strong>{r['move']}</strong></td>
      <td>{r['player']} <span class="sub">({rk_str})</span></td>
      <td class="num"><strong>{prs_str}</strong></td>
      <td class="num">{int(r['hp_leg'])}</td>
      <td class="num">{r['hp_econ']:.2f}</td>
      <td class="num {cls_t}">{tva_val:+.1f}</td>
    </tr>"""

# Side-by-side RAPM Toughest Opponents table rows
rapm_rows_html = ""
for i in range(8):
    rb_row = rapm_b_top.iloc[i]
    rbw_row = rapm_bw_top.iloc[i]
    bnm = 'Tilak Varma' if 'Tilak' in rbw_row['striker_name'] else rbw_row['striker_name']
    rapm_rows_html += f"""
    <tr>
      <td class="rank-cell blue">#{i+1}</td>
      <td><strong>{rb_row['bowler_name']}</strong></td>
      <td class="num">{int(round(rb_row['count']/100)):,}</td>
      <td class="num neg">{rb_row['mean']:.1f}</td>
      <td style="width:14px; background:#FFFFFF; border:none;"></td>
      <td class="rank-cell">#{i+1}</td>
      <td><strong>{bnm}</strong></td>
      <td class="num">{int(round(rbw_row['count']/100)):,}</td>
      <td class="num neg">{rbw_row['mean']:.1f}</td>
    </tr>"""

# Assemble Report HTML
report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SupeRR Selector Season 2 | Round 1 Methodology Report</title>
<style>
  @page {{
    size: A4 portrait;
    margin: 13mm 14mm 14mm 14mm;
    @bottom-left {{
      content: "SupeRR Selector Season 2 — Rajasthan Royals Analytics Challenge";
      font-family: 'Segoe UI', Arial, sans-serif;
      font-size: 7.5pt;
      color: #64748B;
    }}
    @bottom-right {{
      content: "Page " counter(page) " of " counter(pages);
      font-family: 'Segoe UI', Arial, sans-serif;
      font-size: 7.5pt;
      color: #64748B;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Arial, sans-serif;
    color: #1E293B;
    font-size: 9.2pt;
    line-height: 1.46;
    margin: 0;
    padding: 0;
    background: #FFFFFF;
  }}
  .cover-banner {{
    background: linear-gradient(135deg, #131A40 0%, #074EA2 58%, #EA1A85 100%);
    color: #FFFFFF;
    padding: 22px 24px 20px 24px;
    border-radius: 8px;
    margin-bottom: 14px;
  }}
  .cover-kicker {{
    font-size: 8pt;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    color: #FDE047;
    font-weight: 700;
    margin-bottom: 5px;
  }}
  .cover-banner h1 {{
    font-size: 19pt;
    margin: 0 0 6px 0;
    line-height: 1.18;
    letter-spacing: -0.3px;
  }}
  .cover-sub {{
    font-size: 9.6pt;
    color: #E2E8F0;
    margin: 0 0 12px 0;
  }}
  .kpi-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 14px;
  }}
  .kpi-box {{
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-top: 3px solid #EA1A85;
    border-radius: 6px;
    padding: 8px 10px;
  }}
  .kpi-box.blue {{ border-top-color: #074EA2; }}
  .kpi-val {{ font-size: 13.5pt; font-weight: 800; color: #131A40; }}
  .kpi-lbl {{ font-size: 7.5pt; color: #64748B; text-transform: uppercase; letter-spacing: 0.4px; margin-top: 2px; }}

  h2 {{
    font-size: 12.5pt;
    color: #131A40;
    border-bottom: 2px solid #EA1A85;
    padding-bottom: 3px;
    margin: 16px 0 8px 0;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 10.3pt;
    color: #074EA2;
    margin: 11px 0 5px 0;
    page-break-after: avoid;
  }}
  p {{ margin: 0 0 7px 0; }}
  ul {{ margin: 4px 0 8px 18px; padding: 0; }}
  li {{ margin-bottom: 3px; }}

  .callout {{
    background: #FDF2F8;
    border-left: 4px solid #EA1A85;
    padding: 9px 12px;
    border-radius: 0 6px 6px 0;
    margin: 8px 0 10px 0;
    break-inside: avoid;
  }}
  .callout.blue {{
    background: #EFF6FF;
    border-left-color: #074EA2;
  }}
  .formula-box {{
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    padding: 8px 12px;
    font-family: 'Cambria Math', 'Consolas', monospace;
    font-size: 8.6pt;
    color: #0F172A;
    margin: 6px 0 9px 0;
    break-inside: avoid;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    margin: 6px 0 11px 0;
    font-size: 8.1pt;
    break-inside: avoid;
  }}
  th {{
    background: #131A40;
    color: #FFFFFF;
    text-align: left;
    padding: 5.5px 6px;
    font-weight: 700;
    border-bottom: 2px solid #EA1A85;
  }}
  th.num, td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td {{
    padding: 4.5px 6px;
    border-bottom: 1px solid #E2E8F0;
    vertical-align: middle;
  }}
  tr:nth-child(even) td {{ background: #F8FAFC; }}
  .rank-cell {{
    font-weight: 800;
    color: #EA1A85;
    text-align: center;
    width: 34px;
  }}
  .rank-cell.blue {{ color: #074EA2; }}
  .sub {{ font-size: 7.2pt; color: #64748B; font-weight: 400; }}
  .pos {{ color: #0D9488; font-weight: 700; }}
  .neg {{ color: #DC2626; font-weight: 700; }}

  .fig-wrap {{
    margin: 7px 0 10px 0;
    text-align: center;
    break-inside: avoid;
  }}
  .fig-wrap img {{
    width: 96%;
    max-width: 96%;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
  }}
  .fig-wrap.compact img {{
    width: 84%;
    max-width: 84%;
  }}
  .fig-wrap.tight img {{
    width: 65%;
    max-width: 65%;
  }}
  table.compact-tbl td, table.compact-tbl th {{
    padding: 2.5px 5px;
    line-height: 1.25;
  }}
  .fig-cap {{
    font-size: 7.7pt;
    color: #475569;
    margin-top: 3px;
    text-align: left;
    font-style: italic;
  }}

  .profiles-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px 8px;
    margin-top: 5px;
  }}
  .profile-card {{
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 5.5px 8px;
    background: #FAFCFF;
    break-inside: avoid;
  }}
  .pc-head {{
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 3px;
    font-size: 8.6pt;
  }}
  .pc-score {{
    margin-left: auto;
    font-weight: 700;
    color: #0F172A;
    font-size: 8pt;
    background: #E2E8F0;
    padding: 1px 6px;
    border-radius: 10px;
  }}
  .badge {{
    color: #fff;
    font-weight: 700;
    font-size: 7.4pt;
    padding: 1px 6px;
    border-radius: 4px;
  }}
  .badge.pink {{ background: #EA1A85; }}
  .badge.blue {{ background: #074EA2; }}
  .pc-stats {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5px 6px;
    font-size: 7.2pt;
    color: #334155;
    margin-bottom: 3px;
  }}
  .pc-note {{
    font-size: 7.3pt;
    color: #475569;
    margin: 0;
    line-height: 1.28;
  }}
  .page-break {{ page-break-before: always; }}
</style>
</head>
<body>

<div class="cover-banner">
  <div class="cover-kicker">SupeRR Selector Season 2 • Round 1 Submission • Rajasthan Royals Analytics</div>
  <h1>Win-Probability Leverage &amp; Context-Adjusted Clutch Reliability</h1>
  <div class="cover-sub">A Reproducible Ball-by-Ball Framework Identifying the Top 10 Most Reliable Batters &amp; Bowlers Under Pressure (IPL &amp; T20I, 2022–2026) and Shaping RR’s IPL 2027 Auction Strategy</div>
</div>

<div class="kpi-strip">
  <div class="kpi-box"><div class="kpi-val">{meta['total_deliveries']:,}</div><div class="kpi-lbl">Deliveries Audited ({meta['matches_with_bbb']} Matches)</div></div>
  <div class="kpi-box blue"><div class="kpi-val">AUC {wp_val.iloc[1]['roc_auc']:.3f}</div><div class="kpi-lbl">Innings 2 WP OOF Model (Brier {wp_val.iloc[1]['brier_score']:.3f})</div></div>
  <div class="kpi-box"><div class="kpi-val">{meta['eligible_batters']} / {meta['eligible_bowlers']}</div><div class="kpi-lbl">Qualified Batters / Bowlers</div></div>
  <div class="kpi-box blue"><div class="kpi-val">&rho; &ge; {min(meta['spearman_bat_q70'], meta['spearman_bowl_q70']):.3f}</div><div class="kpi-lbl">Rank Stability Across Specs</div></div>
</div>

<h2>1. Executive Summary &amp; Final Ordered Top-10 Rankings</h2>
<p>To separate genuine high-pressure match-winners from garbage-time stat-padders and small-sample variance across the supplied 2022–2026 IPL and T20I dataset (200,999 balls across 867 ball-by-ball matches up to November 1, 2026), we engineered a <strong>Win-Probability Leverage Pressure Index (PI)</strong> paired with an <strong>Out-of-Fold Player-vs-Player Ridge-Adjusted Plus-Minus (RAPM) &amp; Batting-Order Total Value Added (TVA)</strong> scoring engine. Rather than rewarding naive high-pressure deltas that flatter poor normal-overs performers or inflated tailender wicket-takers, our <strong>Pressure Reliability Score (PRS)</strong> combines 50% Empirical-Bayes shrunk High-Pressure RAPM TVA (blended equally across the top-quartile pressure regime and continuous PI&sup2;-weighted impact), 20% High-Pressure Win Probability Added (WPA), 15% Regression-Adjusted Clutch Residual relative to floored usual baseline, and 15% Cross-Match Consistency.</p>

<h3>Final Ordered Top 10 Batters Under Pressure (Unstop Rank 1–10)</h3>
<table>
  <thead>
    <tr>
      <th style="text-align:center;">Rank</th><th>Batter (Type &amp; IPL 2026 Team)</th><th class="num">PRS</th>
      <th class="num">HP Balls (M)</th><th class="num">HP SR (LP)</th><th class="num">HP TVA/100*</th>
      <th class="num">LP TVA/100*</th><th class="num">Clutch Resid</th><th class="num">HP WPA/100</th><th class="num">Pos. Match %</th>
    </tr>
  </thead>
  <tbody>{bat_rows_html}</tbody>
</table>

<h3>Final Ordered Top 10 Bowlers Under Pressure (Unstop Rank 1–10)</h3>
<table>
  <thead>
    <tr>
      <th style="text-align:center;">Rank</th><th>Bowler (Style &amp; IPL 2026 Team)</th><th class="num">PRS</th>
      <th class="num">HP Balls (M)</th><th class="num">HP Econ (LP)</th><th class="num">HP Wkts (SR)</th><th class="num">HP TVA/100*</th>
      <th class="num">LP TVA/100*</th><th class="num">Clutch Resid</th><th class="num">HP WPA/100</th><th class="num">Pos. Match %</th>
    </tr>
  </thead>
  <tbody>{bowl_rows_html}</tbody>
</table>
<p class="sub">*All TVA/100 columns report Empirical-Bayes shrunk (k=500 balls), opposition-RAPM, batting-order, and context-adjusted Total Value Added per 100 balls; p = 1,000-iteration match-clustered bootstrap probability of positive raw high-pressure TVA.</p>

<div class="callout blue" style="margin-top:8px; margin-bottom:7px; padding:7px 11px; font-size:8.1pt; line-height:1.33;">
  <strong>Independent Super Over Crucible &amp; Negative Clutch Contrast:</strong> Excluding the 6 Super Overs (38 balls) from primary scoring provides an out-of-sample sanity check on ultimate pressure: Batter <strong>#1 Suryakumar Yadav</strong> finished both Indian Super Overs unbeaten (4 off 1 vs SL 2024; 3 off 1 vs SL 2025), while Bowler <strong>#4 Sunil Narine</strong> bowled KKR's IPL 2026 Super Over vs LSG conceding just 1 run and taking 2 wickets in 3 balls. Conversely, among high-volume (&ge;600-ball) players, the steepest performance collapses under pressure occurred in <strong>Aiden Markram</strong> (-14.7 clutch residual), <strong>Charith Asalanka</strong> (-14.3), <strong>Prasidh Krishna</strong> (-12.8), and <strong>Nuwan Thushara</strong> (-11.2).
</div>

<div class="callout" style="margin-top:0; padding:7px 11px; font-size:8.0pt; line-height:1.33;">
  <strong>Unified Cross-Discipline Top-10 Consensus &amp; Dual All-Rounder Reality Check:</strong> Pooling peak PRS across Bat (<code>n=73</code>) and Bowl (<code>n=79</code>) yields one consensus <strong>Master Top 10</strong>: <strong>1. Suryakumar</strong> (85.9), <strong>2. Tim David</strong> (83.2), <strong>3. Krunal Pandya</strong> (79.9), <strong>4. Axar Patel</strong> (78.4), <strong>5. Tilak Varma</strong> (77.5), <strong>6. Shreyas Iyer</strong> (73.9), <strong>7. Bhuvneshwar Kumar</strong> (73.4), <strong>8. Sunil Narine</strong> (73.0), <strong>9. Josh Hazlewood</strong> (72.2), and <strong>10. Jasprit Bumrah</strong> (71.7, edging #7 Bowler <strong>Nathan Ellis</strong> by 0.03). Crucially, evaluating all <strong>13 Dual-Qualified All-Rounders</strong> (&ge;180 HP balls in <em>both</em> skills) prevents roster traps: <strong>Axar</strong> (<strong>+1.40</strong> net HP runs/match: +15.56 bowl / &minus;0.44 bat) and <strong>Krunal</strong> (<strong>+1.38</strong>: +15.84 bowl #1 / &minus;4.60 bat #66—confirming strict overs 7–16 bowling deployment) lead overall, while <strong>Rashid Khan</strong> (+1.29: +10.45 bowl #8 / +6.58 bat #25) is the sole Top-10 bowler also ranking Top-25 with the bat. By contrast, <strong>Hasaranga</strong> (+0.42 net: +9.83 / &minus;8.61 #71 bat) needs #8 batting protection—and departing <strong>Sam Curran</strong> (&minus;4.00 bat #65 / &minus;7.35 bowl #74 = <strong>&minus;1.06 net HP runs/m</strong>) ranked <strong>last (#13/13)</strong> among all two-way players!
</div>

<div class="page-break"></div>

<h2>2. Data Hygiene, Scoring-Era Normalization &amp; Competition Contexts</h2>
<p>Careful auditing of the 6 CSV tables prevented five common ball-by-ball traps before any modeling began:</p>
<ul>
  <li><strong>Legal-Delivery &amp; Extras Accounting:</strong> In <code>ball_by_ball.csv</code>, <code>innings_ball_num</code> increments on wides (7,714 rows) and no-balls (576 <code>Overstep</code> + 264 <code>other</code> rows, all carrying 1 penalty run). Reconstructing cumulative legal balls directly from <code>~extras_type.isin(['wide','Overstep','other'])</code> ensured exact 120-ball maximums and zero mismatches across all 341 non-DLS run-margin results via <code>runs + extras1_runs + extras_2_runs</code>.</li>
  <li><strong>DLS &amp; Super Over Handling:</strong> The 25 matches with <code>first_inn_revised_overs</code> or <code>second_inn_revised_overs</code> store revised limits in <em>balls</em> (e.g., 114, 84, 72, 42, 30), not overs. Four rain-truncated DLS games lacking <code>revised_target</code> and 41 abandoned/no-result games were excluded from Win Probability outcome labels. The 6 Super Overs (innings 3 &amp; 4, 38 balls) were held out as an independent qualitative crunch validation check.</li>
  <li><strong>Dismissal Attribution:</strong> Bowlers are credited only for wickets excluding <code>RunOut</code> (600), <code>Retire Out</code> (6), and <code>Obstructing the field</code> (3); batters are charged only when <code>dismissed_player_id == striker_id</code>.</li>
  <li><strong>IPL Scoring Inflation (+15.7%) &amp; Tier Divergence:</strong> During the Impact Player era (active in 136–142 IPL slots per season from 2023–2026), IPL scoring jumped from <strong>8.54 RPO in 2022</strong> to <strong>9.88 RPO in 2026</strong>, whereas Full-Member T20Is averaged <strong>8.42–8.66 RPO</strong> (2022–2025) and <strong>9.21 RPO</strong> in 2026. Associate-vs-Associate T20Is (7.71 RPO) and Emerging/Rising Asia Cup (7.92 RPO) occupy far lower scoring regimes. Comparing raw Strike Rate or Economy Rate across eras and formats systematically penalizes IPL 2024–26 bowlers and flatters IPL 2026 batters.</li>
</ul>

<div class="fig-wrap">
  <img src="data:image/png;base64,{img_b64('fig1_context_inflation.png')}" alt="Era Inflation and Phase Value Curve">
  <div class="fig-cap"><strong>Figure 1:</strong> (A) IPL scoring surged +15.7% from 8.54 to 9.88 RPO between 2022 and 2026, diverging sharply from T20Is and Emerging/Associate tiers. (B) Empirical T20 run-rate curve across overs 1–20 alongside dynamic wicket run value V_W(s) derived from our monotonic remaining-runs resource model.</div>
</div>

<h2>3. Pressure Definition: Win-Probability Leverage &amp; Stress Indicators</h2>
<p>Traditional binary pressure cutoffs (e.g., "Overs 16–20 with RRR &gt; 10") fail in two fatal directions: they misclassify hopeless dead chases (needing 55 off 12 balls at 2% WP) as "high pressure", while missing acute 1st-innings collapses (25/3 in Over 4) or early 2nd-innings wicket crises. Instead, we define pressure dynamically from first principles on every delivery prior to the ball being bowled:</p>

<h3>3.1 Monotonic Resource &amp; Match-Grouped Win Probability Models</h3>
<ol>
  <li><strong>Leave-One-Match-Out Venue &amp; Tier Environment Par (<code>env_par_rpo</code>):</strong> Each match's expected scoring rate shrinks the venue's leave-one-match-out RPO toward the competition-tier &times; season baseline (prior = 1,440 balls) to prevent target leakage.</li>
  <li><strong>1st-Innings Remaining Runs Resource Model:</strong> A monotonic <code>HistGradientBoostingRegressor</code> (constrained +1 on <code>balls_rem</code>, <code>wickets_rem</code>, and <code>env_par_rpo</code>) achieves out-of-sample chronological 2026 holdout <strong>MAE = {meta['res_mae_2026']:.2f} runs, R&sup2; = {meta['res_r2_2026']:.3f}</strong>, yielding both projected total surplus and dynamic phase-aware wicket value <code>V_W(s) = E_rem(b, w) - E_rem(b, w-1)</code> (averaging <strong>10.6 runs</strong> in Powerplay, <strong>6.6</strong> in Middle overs, and <strong>3.7</strong> at Death).</li>
  <li><strong>Calibrated Win Probability (<code>WP1</code>, <code>WP2</code>):</strong> Comparing Logistic Regression against Gradient Boosted Trees on chronological 2026 holdout revealed severe tree overfitting on match-level binary outcomes (I2 Test LogLoss 0.434 vs 0.375). Regularized Logistic Regression (<code>C=0.1</code>) built on domain-engineered resource features (<code>chase_surplus</code>, <code>surplus_std</code>, <code>log_rrr_ratio</code>) achieved <strong>5-fold match-grouped OOF ROC-AUC = 0.901, Brier = 0.128</strong> in Innings 2 and <strong>AUC = 0.691</strong> in Innings 1, with near-perfect decile calibration.</li>
</ol>

<h3>3.2 Multi-Pillar Composite Pressure Index (PI &isin; [0, 100])</h3>
<p>Using counterfactual transitions after 1 legal ball (dot, +1, +4, +6, wicket) weighted by empirical phase frequencies, we compute <strong>Counterfactual WP Leverage</strong> <code>&sigma;_&Delta;WP(s) = sqrt(&Sigma; p_k (WP(s_k) - &mu;_WP)&sup2;)</code>, gated by a <strong>Contest Liveness Filter</strong> (dropping toward 0 when chasing WP &lt; 3% or &gt; 97%). We combine WP Leverage with three cricketing stress signals and Knockout stakes into role-specific indices:</p>

<div class="formula-box">
  <strong>Batting Pressure (PI_bat):</strong> 100 &middot; clip( 0.38&middot;Lev_norm + 0.34&middot;max(Collapse_stress, 0.6&middot;WktShock_18b) + 0.24&middot;RRR_stress + 0.24&middot;Death_ramp + KO_boost )<br>
  <strong>Bowling Pressure (PI_bowl):</strong> 100 &middot; clip( 0.40&middot;Lev_norm + 0.32&middot;Death_ramp + 0.20&middot;Defense_risk + 0.18&middot;Surge_18b + KO_boost )
</div>

<div class="fig-wrap">
  <img src="data:image/png;base64,{img_b64('fig2_wp_calibration_and_situations.png')}" alt="WP Calibration and Situation Validity">
  <div class="fig-cap"><strong>Figure 2:</strong> (A) Out-of-fold Win Probability calibration aligns tightly with observed outcomes across all deciles. (B) The Pressure Index cleanly separates live 2nd-innings death crunches (81.7 bat / 80.4 bowl) and early batting collapses (39.4) from comfortable Powerplays (15.7) and dead/cruise chases (1.8–6.6).</div>
</div>

<h2>4. Performance Measurement, RAPM Matchup Quality &amp; Scoring Logic</h2>
<p>To compare players across IPL seasons, venues, T20I opponents, batting-order quality, and individual matchups without role or era bias, we evaluate every delivery through a two-stage <strong>Out-of-Fold Context Baseline + Player-vs-Player RAPM Ridge</strong> engine:</p>
<ul>
  <li><strong>Context &amp; Batting-Order Baselines (<code>xR</code>, <code>xW</code>):</strong> Separate 5-fold match-grouped HistGB baselines trained on <code>(match_tier, pace_or_spin, innings, env_par_rpo, overid, wickets_rem)</code> predict batter expectations, while bowling baselines add exact striker <strong><code>batting_order</code></strong>—preventing tailender cheap wickets from artificially inflating death bowlers.</li>
  <li><strong>Out-of-Fold Player-vs-Player RAPM Ridge Quality (<code>&alpha;=1200</code>):</strong> Rather than adjusting only for coarse opponent team strength, we solve two sparse 5-fold OOF Ridge regressions over joint one-hot indicators of <code>(striker_id, bowler_id, team_id)</code> on raw TVA residuals, subtracting strictly the out-of-fold opponent coefficients. The toughest bowlers faced in the dataset score up to <strong>&minus;16.7 TVA/100</strong> for batters (Varun Chakravarthy &minus;16.7, Jasprit Bumrah &minus;16.1, Sunil Narine &minus;14.1, Josh Hazlewood &minus;13.0, Nathan Ellis &minus;12.5), while elite hitters impose up to <strong>&minus;26.7 TVA/100</strong> on bowlers (Abhishek Sharma &minus;26.7, Vaibhav Sooryavanshi &minus;24.5, Suryakumar Yadav &minus;22.4, Tim David &minus;18.2).</li>
  <li><strong>Total Value Added (<code>TVA</code>):</strong> Unifies scoring speed and wicket conservation/taking using dynamic empirical wicket value:
    <div class="formula-box">
      TVA_bat = (runs - xR_bat) + V_W(s)&middot;(xOut_bat - I_out) - RAPM_OppBowlAdj<br>
      TVA_bowl = (xR_bowl - bowler_runs) + V_W(s)&middot;(I_wkt - xWkt_bowl) - RAPM_OppBatAdj
    </div>
  </li>
  <li><strong>Fair Competition Eligibility &amp; Sample Gates:</strong> Associate-only T20Is and Emerging A-team tournaments are excluded from primary rankings (analyzed separately for RR's scouting radar), restricting eligibility to <strong>Top-Flight T20 (IPL + Full-Member T20Is = 182,038 balls across 784 matches)</strong>. Players require <strong>&ge;400 total balls</strong>, <strong>&ge;180 High-Pressure balls</strong> (top quartile: <code>PI_bat &ge; 36.2</code>, <code>PI_bowl &ge; 40.6</code>), and <strong>&ge;15 High-Pressure matches</strong>—filtering out low-volume opener survivorship artifacts (e.g., Yashasvi Jaiswal with 106 HP balls across 12 games) and small-sample noise.</li>
  <li><strong>Blended Impact + Regression-Adjusted Clutch Residual:</strong> To avoid cliff-edge cutoffs at Q75, we blend 50% Empirical-Bayes shrunk Q75 High-Pressure TVA (<code>k=500</code>) with 50% continuous <code>(PI/100)&sup2;</code>-weighted TVA. Comparing against usual performance via OLS regression (slopes: <strong>{meta['bat_hp_lp_slope']:.3f}</strong> for batters, <strong>{meta['bowl_hp_lp_slope']:.3f}</strong> for bowlers) with <code>LP</code> floored at 0 eliminates regression-to-the-mean distortion.</li>
  <li><strong>Composite Pressure Reliability Score (PRS):</strong> <code>PRS_z = 0.50&middot;Z(Blended_HP_TVA) + 0.20&middot;Z(HP_WPA) + 0.15&middot;Z(Clutch_Residual) + 0.15&middot;Z_cons(PosMatchRate, Match_tstat)</code>, rescaled to <code>PRS = 50 + 15&middot;PRS_z</code>.</li>
</ul>

<div class="fig-wrap compact">
  <img src="data:image/png;base64,{img_b64('fig3_batters_quadrant_top10.png')}" alt="Top 10 Batters Quadrant">
  <div class="fig-cap"><strong>Figure 3:</strong> Usual-pressure vs. high-pressure batting TVA across all 73 eligible batters (RAPM-adjusted). Top 10 batters either sustain elite dominance well above expectation (Suryakumar Yadav, Tim David, Tilak Varma, Shreyas Iyer) or lift sharply under crunch stress (David Miller, Dasun Shanaka, Shashank Singh, Ishan Kishan), contrasting with negative clutch dippers (Aiden Markram, Charith Asalanka, Mitchell Marsh).</div>
</div>

<div class="fig-wrap compact">
  <img src="data:image/png;base64,{img_b64('fig4_bowlers_quadrant_top10.png')}" alt="Top 10 Bowlers Quadrant">
  <div class="fig-cap"><strong>Figure 4:</strong> Usual-pressure vs. high-pressure bowling TVA across all 79 eligible bowlers (RAPM &amp; order-adjusted). Krunal Pandya and Axar Patel achieve huge pressure lift (+14.0 and +13.7 clutch residuals), while Bhuvneshwar Kumar, Sunil Narine, Josh Hazlewood, Jasprit Bumrah, and Nathan Ellis combine normal-overs excellence with sustained crunch execution.</div>
</div>

<div class="page-break"></div>

<h2>5. Evidence, 3-Vector Matchups &amp; Micro-Tactical Execution Profiles</h2>
<p>Ball-by-ball telemetry (<code>stroke</code>, <code>delivery_style</code>, <code>bowl_speed</code>, Pace-On vs. Pace-Off vs. Spin splits, and First-10-Ball entry windows) reveals <em>how</em> top pressure performers sustain output—and where hidden tactical vulnerabilities lie (Figure 8).</p>

<div class="fig-wrap">
  <img src="data:image/png;base64,{img_b64('fig8_3vector_and_death_playbook.png')}" alt="3-Vector Matchup Heatmap and Death Delivery Playbook">
  <div class="fig-cap"><strong>Figure 8:</strong> (A) Top-10 Batter 3-Vector &amp; Entry-10 matrix exposes sharp specialization: Suryakumar excels across Pace-On (+45.1), Pace-Off (+91.3), and Spin (+35.8), whereas Tim David (-8.6) and Heinrich Klaasen (-32.8) dip sharply against Pace-Off variations while Tilak Varma (+31.9) and Suryakumar (+29.3) thrive inside their first 10 pressure balls. (B) High-pressure death execution hierarchy proves Yorkers (+78.6 TVA/100, 5.5 Econ) and Carrom balls (+27.3) dominate, while Straight (-31.9) and Full Tosses (-32.1, 12.9 Econ) leak heavily.</div>
</div>

<div class="fig-wrap">
  <img src="data:image/png;base64,{img_b64('fig5_bootstrap_forest_plots.png')}" alt="Bootstrap Confidence Intervals">
  <div class="fig-cap"><strong>Figure 5:</strong> 1,000-iteration match-clustered bootstrap 95% confidence intervals for raw High-Pressure RAPM TVA/100. Every Top-10 batter and bowler shows 89%–100% bootstrap probability of positive pressure value; Suryakumar Yadav, Tim David, Tilak Varma, Shreyas Iyer, Krunal Pandya, Bhuvneshwar Kumar, Sunil Narine, Jasprit Bumrah, Nathan Ellis, and Kuldeep Yadav clear zero across their full 95% intervals.</div>
</div>

<h3>Top 10 Batters: Tactical Execution Profiles</h3>
<div class="profiles-grid">{bat_cards_html}</div>

<h3>Top 10 Bowlers: Tactical Execution Profiles</h3>
<div class="profiles-grid">{bowl_cards_html}</div>

<h3 style="margin-top:12px;">Out-of-Fold RAPM Player-vs-Player Opposition Difficulty Leaders (&ge;500 Balls)</h3>
<p class="sub" style="margin-bottom:4px;">Negative values measure how many RAPM-adjusted TVA runs per 100 balls an elite opponent suppresses opposing hitters (left) or bowlers (right).</p>
<table>
  <thead>
    <tr>
      <th style="text-align:center;">#</th><th>Toughest Bowler Faced</th><th class="num">Balls</th><th class="num">Bat Adj/100</th>
      <th style="width:14px; background:#FFFFFF; border:none;"></th>
      <th style="text-align:center;">#</th><th>Toughest Batter Bowled To</th><th class="num">Balls</th><th class="num">Bowl Adj/100</th>
    </tr>
  </thead>
  <tbody>{rapm_rows_html}</tbody>
</table>

<div class="page-break"></div>

<h2>6. Rajasthan Royals IPL 2026 Autopsy &amp; IPL 2027 Auction Blueprint</h2>
<p>Decomposing Rajasthan Royals’ 16 IPL 2026 matches (9 wins, 7 losses, reaching Qualifier 2 before falling to Gujarat Titans) exposes the exact pressure asymmetry costing RR the title—and quantifies our blueprint for the <strong>December 16, 2026 IPL 2027 Mini-Auction in Goa</strong>:</p>

<div class="callout">
  <strong>RR IPL 2026 Win-vs-Loss Autopsy (Figure 9A &amp; Figure 7A):</strong> In RR’s 9 wins, both High-Pressure Batting (<strong>+18.3 TVA/100</strong>, +0.84 WPA) and High-Pressure Bowling (<strong>+8.4 TVA/100</strong>, +1.06 WPA) dominated opponents. In RR’s 7 losses, HP Batting actually remained positive (<strong>+7.5 TVA/100</strong>, +0.24 WPA), but <strong>High-Pressure Bowling collapsed to &minus;21.4 TVA/100 (&minus;86.15 TVA, &minus;2.71 WPA)</strong>—a massive <strong>29.8 TVA/100</strong> (&minus;3.77 WPA) bowling swing! Across 2022–26 top-flight career pressure splits, RR relied heavily on <strong>Vaibhav Sooryavanshi</strong> (776 runs @ 237.3 SR in IPL 2026), <strong>Yashasvi Jaiswal</strong> (+85.4 career HP TVA, 106b), <strong>Donovan Ferreira</strong> (+28.5 HP TVA), <strong>Dasun Shanaka</strong> (#9 overall, +17.1), <strong>Riyan Parag</strong> (+16.9), and <strong>Jofra Archer</strong> (+13.0 HP TVA, 25 IPL 2026 wkts)—while leaking heavily through <strong>Sam Curran</strong> (&minus;17.9 bowl / &minus;8.0 bat HP TVA), <strong>Tushar Deshpande</strong> (&minus;16.2 HP TVA, 11.85 HP Econ), <strong>Brijesh Sharma</strong> (&minus;8.9 HP TVA, 11.57 HP Econ), <strong>Sandeep Sharma</strong> (&minus;8.7 HP TVA, 10.80 HP Econ + 13 runs conceded in RR’s 2025 Super Over), <strong>Shimron Hetmyer</strong> (+4.0 HP TVA vs +11.9 LP; 78 runs @ 113.0 SR in IPL 2026), and <strong>Ravi Bishnoi</strong> (+1.0 HP vs +10.7 LP).
</div>

<div class="fig-wrap compact">
  <img src="data:image/png;base64,{img_b64('fig9_rr_loss_autopsy_and_strokes.png')}" alt="RR IPL 2026 Win vs Loss Autopsy and Batting Stroke Playbook">
  <div class="fig-cap"><strong>Figure 9:</strong> (A) RR’s IPL 2026 losses were driven by a 29.8 TVA/100 collapse in High-Pressure Bowling (+8.4 in 9 wins vs &minus;21.4 in 7 losses). (B) Across high-pressure batting strokes, clean straight/pull/cover execution (+29.3 to +118.7 TVA/100) outperforms high-risk slogs (&minus;11.7, 17.5% out) and defensive blocks (&minus;78.5 to &minus;81.0).</div>
</div>

<div class="fig-wrap compact">
  <img src="data:image/png;base64,{img_b64('fig7_rr_2027_auction_fit.png')}" alt="RR 2026 Audit and 2027 Auction Targets">
  <div class="fig-cap"><strong>Figure 7:</strong> (A) RR’s IPL 2026 core roster under high vs. usual pressure highlights bowling leaks alongside Jofra Archer. (B) Priority IPL 2027 Goa auction and trade-fit targets mapped directly from our Top-10 and IPL-only RAPM models.</div>
</div>

<p><strong>Quantified Auction Lift Simulation:</strong> Upgrading 1 death pace partner and 1 defensive middle/death spinner to lift RR’s 403 high-pressure bowling deliveries in losses from <code>&minus;21.4 TVA/100</code> to merely league par (<code>0.0</code>) recovers <strong>+86.2 expected runs</strong> across RR’s crunch-losing sequences (~12.3 runs per loss in tight defeats vs KKR/SRH/DC)—and matching Ellis/Narine/Krunal output (<code>+14.0 TVA/100</code>) yields a <strong>+142.6-run pressure swing</strong>.</p>

<div class="page-break"></div>

<h2 style="margin-top:0;">6.1 RR Home Fortress Physics &amp; Playing-XI Role Context</h2>
<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:7px;">
  <div class="callout blue" style="margin:0; padding:7px 9px; font-size:7.9pt; line-height:1.32;">
    <strong>Jaipur &amp; Guwahati Venue Physics (High Pressure):</strong> Across RR’s two home fortresses—<strong>Jaipur</strong> (<code>venue_id=6</code>, 19 RR IPL matches, 57.1% chase win rate) and <strong>Guwahati</strong> (<code>venue_id=1241</code>, 9 RR IPL matches)—<strong>Spin systematically dominates both Pace-On and Pace-Off</strong> under crunch conditions:<br>
    &bull; <strong>Jaipur HP:</strong> Spin <strong>9.38 Econ</strong> (18.8% bdry) vs. Pace-On <strong>10.96</strong> (25.5%) &amp; Pace-Off <strong>11.49</strong> (25.7%).<br>
    &bull; <strong>Guwahati HP:</strong> Spin <strong>8.12 Econ</strong> (15.7% bdry) vs. Pace-On <strong>9.89</strong> (23.5%) &amp; Pace-Off <strong>10.80</strong> (21.7%).
  </div>
  <div class="callout" style="margin:0; padding:7px 9px; font-size:7.9pt; line-height:1.32;">
    <strong>Playing-XI Sub Penalty &amp; Leadership Signal:</strong> Integrating <code>playing_eleven.csv</code> exposes structural selection traps inside the Impact Player era (IPL 2023–26 HP):<br>
    &bull; <strong>Impact-Sub Penalty:</strong> Starting-XI batters (<strong>+3.02</strong> TVA/100, 147.4 SR) beat Impact Subs (<strong>&minus;2.73</strong>, 137.0 SR); Starting-XI bowlers (<strong>+1.19</strong>, 9.62 Econ) beat Impact Subs (<strong>&minus;1.98</strong>, 9.40 Econ).<br>
    &bull; <strong>Leadership Alpha:</strong> Top-flight <strong>Captains (+7.16 TVA/100)</strong> outscore <strong>Non-Captains (+1.54)</strong> under HP (confirming Suryakumar, Iyer, Shanaka, Rashid).
  </div>
</div>

<h3 style="margin:6px 0 3px 0;">6.2 Pre-Retention Front-Office Trade Audit (Live Ahead of Nov 17, 2026 Deadline &amp; Dec 16 Goa Auction)</h3>
<p style="font-size:8.1pt; margin-bottom:4px;">Scoring reported pre-retention IPL 2027 trades (under the <strong>₹157 Cr</strong> cap) through our 2022–26 HP RAPM model reveals major market inefficiencies and explains why our primary pace-death board must pivot immediately:</p>
<table class="compact-tbl" style="margin:2px 0 5px 0; font-size:7.6pt;">
  <thead>
    <tr><th>Reported Trade Corridor (Nov 2026)</th><th>Player (Pool Rank)</th><th class="num">PRS</th><th class="num">HP Balls</th><th class="num">HP Econ</th><th class="num">HP TVA/100*</th></tr>
  </thead>
  <tbody>{trade_rows_html}</tbody>
</table>
<p style="font-size:7.8pt; color:#334155; margin-bottom:6px;"><em>*Blended EB+RAPM HP TVA/100 for &ge;400b-pool qualifiers; raw HP TVA/100 shown for unqualified low-volume legs.</em> <strong>RR Trade Verdict:</strong> Shedding <strong>Sam Curran (#74, PRS 29.6)</strong> for <strong>Mohammed Shami (#44, PRS 47.8)</strong> yields a clean <strong>+8.5 blended TVA/100</strong> pace upgrade (cutting HP Economy from 10.95 to 8.78), but incoming SLA specialist <strong>Manimaran Siddharth</strong> (56 HP balls, 10.29 Econ, &minus;35.2 raw TVA) is a Powerplay specialist—meaning <em>RR’s crunch-over spin &amp; death pace gaps remain open</em>. Meanwhile, CSK trading <strong>Nathan Ellis (#7, PRS 71.7) &rarr; SRH</strong> removes our #1 overall pace-off death target from the auction pool, forcing a decisive Triple Lock pivot to KKR’s injury/release watch <strong>Matheesha Pathirana (#11)</strong> and DC’s <strong>Lungi Ngidi (#21, #5 IPL-Only PRS 74.0)</strong>.</p>

<h3 style="margin:6px 0 3px 0;">6.3 RR Triple Lock Target Matrix (Dec 16, 2026 Goa Mini-Auction) &amp; Moneyball Value Radar</h3>
<table class="compact-tbl" style="margin:3px 0 5px 0; font-size:7.4pt;">
  <thead>
    <tr><th>RR Structural Need (IPL 2027)</th><th>Target Player &amp; Status</th><th>Role &amp; Slot</th><th>Data-Backed Triple Lock Fit &amp; LHB/RHB Matchup Rationale</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1. Death Pace Partner</strong><br><span class="sub">(Pair with Archer + Shami)</span></td>
      <td><strong>Matheesha Pathirana</strong> (#11, KKR*) /<br><strong>Lungi Ngidi</strong> (#21, #5 IPL-Only, DC)<br><span class="sub">*(Pivot after Ellis CSK&rarr;SRH trade)</span></td>
      <td>Overseas Pace<br>Death Specialist</td>
      <td>With <strong>Nathan Ellis (#7)</strong> traded to SRH, Pathirana (<strong>11.0 HP SR, 34 HP wkts</strong>, ₹18 cr release watch after 1-match/8-ball injury 2026) and Ngidi (<strong>7.15 HP Econ, +63.0 raw TVA</strong> across 120 IPL HPb) are urgent primary bids; also target <strong>Eshan Malinga</strong> (SRH value).</td>
    </tr>
    <tr>
      <td><strong>2. Indian Spin / Control AR</strong><br><span class="sub">(Exploit Jaipur &amp; Guwahati Spin)</span></td>
      <td><strong>Krunal Pandya</strong> (#1, RCB) /<br><strong>Harpreet Brar</strong><br><span class="sub">(Moneyball Radar, PBKS)</span></td>
      <td>Indian Spin AR<br>Overs 7–16 Control</td>
      <td>Krunal (<strong>7.18 HP Econ, +70.4 vs LHB / +41.4 vs RHB, +61.2 Chase Defense</strong>) and value target Brar (<strong>159 IPL HPb, 7.96 Econ, +42.2 raw HP TVA</strong>) convert Jaipur (9.38) &amp; Guwahati (8.12) spin edge into non-OS wickets.</td>
    </tr>
    <tr>
      <td><strong>3. Proven OS Finisher</strong><br><span class="sub">(Upgrade Hetmyer/Curran slot)</span></td>
      <td><strong>David Miller</strong> (#6, DC) /<br><strong>Heinrich Klaasen</strong> (#8, SRH)</td>
      <td>Overseas No. 5/6<br>Crunch Closer</td>
      <td>Miller (+13.0 blended HP TVA, +9.4 clutch resid, +47.5 vs Spin, +42.1 Chase Death) fixes RR’s brittle finisher slot left by Hetmyer (113.0 SR in IPL 2026); LHB/RHB matchup matrix confirms spin/pace-off targets LHBs heavily (Krunal +70.4, Kuldeep +54.2, Bumrah +49.5).</td>
    </tr>
    <tr>
      <td><strong>4. Domestic Moneyball Bat</strong><br><span class="sub">("Next Vignesh Puthur" Uncapped)</span></td>
      <td><strong>Aniket Verma</strong> (SRH) /<br><strong>Ashutosh Sharma</strong> (#5 IPL-Only, DC) /<br><strong>Naman Dhir</strong> (MI)</td>
      <td>Indian Finisher<br>No. 5–7 Closure</td>
      <td>High-upside IPL HP radars (&ge;70b, outside Top 10): uncapped <strong>Aniket Verma</strong> (<strong>101 HPb, 184.2 SR, +56.6 raw TVA</strong>), Ashutosh (<strong>157 HPb, 166.2 SR, +0.51 shrunk WPA/100</strong>), and India newcomer <strong>Naman Dhir</strong> (133 HPb, 174.4 SR, +25.0) pair cleanly with RR retainee <strong>Donovan Ferreira</strong> (181.6 IPL HP SR, +26.5 raw IPL TVA).</td>
    </tr>
    <tr>
      <td><strong>5. Attacking Wrist-Spin Depth</strong><br><span class="sub">(Pair with Bishnoi/Puthur/Jadeja)</span></td>
      <td><strong>Kuldeep Yadav</strong> (#9, DC) /<br><strong>Wanindu Hasaranga</strong> (#10, LSG)</td>
      <td>Strike Wrist-Spin<br>Middle Overs</td>
      <td>Kuldeep (<strong>7.62 IPL HP Econ, +54.2 vs LHB</strong>, 61.8% overall / 78.3% IPL pos. match rate) or ex-Royal Hasaranga (12.9 HP SR) end RR’s middle-over wicket drought alongside Season 1 find <strong>Vignesh Puthur</strong> (₹30L RR squad; 0 IPL 2026 balls in feed after 6 wkts in 5 matches for MI in 2025) and retainee <strong>Nandre Burger</strong> (97 IPL HPb, +42.5 raw IPL TVA).</td>
    </tr>
  </tbody>
</table>
<p style="font-size:7.7pt; margin-bottom:2px;"><strong>Franchise Core Benchmarks vs. Emerging Radar:</strong> While retained Top-10 icons (Suryakumar, Tilak, Bumrah, Shreyas, Narine, Hazlewood, Tim David, Axar, Rashid) are unavailable in Goa, RR retains #9 Batter <strong>Dasun Shanaka</strong> and executes the actionable shortlist above. Across 30 Emerging/Rising Asia Cup games, RR’s <strong>Vaibhav Sooryavanshi</strong> (239 runs in 98 balls, 243.9 SR) confirmed elite ceiling, with <strong>Ripon Mondol</strong> (19 wkts @ 6.91 Econ), <strong>Sufiyan Muqeem</strong> (16 wkts @ 5.03 Econ), and <strong>Sediqullah Atal</strong> (443 runs @ 151.2 SR) on developmental watch—all entering Giles Lindsay’s <strong>Triple Lock</strong> workflow ("Inform, Not Dictate": models + scouts + Sangakkara/Panish Shetty).</p>

<div class="page-break"></div>

<h2 style="margin-top:0;">7. Out-of-Time Scientific Validation, Mechanisms &amp; Limitations</h2>
<p style="font-size:8.2pt; margin-bottom:4px;"><strong>7.1 Temporal Prediction (2022–24 &rarr; 2025–26) &amp; Physical Crunch Mechanisms (Figure 10):</strong> Evaluating true out-of-sample forecasting proves why naive raw metrics fail front-office decisions. On <strong>Batters (n=50)</strong> and <strong>Bowlers (n=54)</strong> active across both eras, <strong>Naive Clutch Delta (<code>HP &minus; LP</code>) carries zero/negative predictive correlation (r = &minus;0.019 bat; r = &minus;0.055 bowl)</strong>. By contrast, overall class + Empirical Bayes shrinkage recovers durable year-to-year signal: <strong>Bowler Overall &minus;Econ (r = +0.362, p = 0.0072)</strong>, <strong>Blended EB+RAPM HP TVA (r = +0.239, &rho; = +0.252 bowl; r = +0.102 bat)</strong>, and <strong>Full RAPM PRS (r = +0.207, &rho; = +0.233 bowl; r = +0.087 bat)</strong>. Mechanistically (Panel B), <strong>Top-10 Batters</strong> sustain boundary output under HP (<strong>21.3% &rarr; 21.3%</strong>) while <strong>Dip-10 Batters (&ge;500b)</strong> collapse (<strong>19.3% &rarr; 13.9%, &minus;5.4 pp</strong>; SR 145.2 &rarr; 123.9). Under crunch bowling, <strong>Top-10 Bowlers</strong> raise elite variations (<strong>9.4% &rarr; 16.1%, +6.8 pp</strong>) and cap <code>Full Toss/Straight</code> error balls at <strong>9.5%</strong>, whereas <strong>Dip-10 Bowlers</strong> lose <strong>7.4 pp of dot balls</strong> (36.5% &rarr; 29.1%) and double error balls to <strong>18.3%</strong>.</p>

<div class="fig-wrap">
  <img src="data:image/png;base64,{img_b64('fig10_mechanisms_and_temporal_proof.png')}" alt="Temporal Prediction Benchmark and Physical Clutch Mechanisms" style="width:82%; max-width:82%;">
  <div class="fig-cap"><strong>Figure 10:</strong> (A) Out-of-time 2022–24 &rarr; 2025–26 validation proves Naive Clutch (HP&minus;LP) chases noise while EB-shrunk RAPM and overall baseline class generalize. (B) Mechanism audit contrasts Top-10 boundary/variation discipline against Dip-10 boundary collapse and doubled HP error rates (18.3% freebies).</div>
</div>

<p style="font-size:8.2pt; margin-bottom:4px;"><strong>7.2 Six-Specification Sensitivity Stability (Figure 6):</strong> Across six alternative specifications—including pressure cutoffs (Q70, Q80), shrinkage priors (k=300, 800), IPL-only, and the <strong>2024–2026 Impact Player era</strong>—Spearman rank correlations with our primary model (<code>Q75, k=500</code>) exceed <strong>&rho; = 0.971 (Q70)</strong>, <strong>&rho; = 0.999 (k=300/800)</strong>, and <strong>&rho; = 0.733 (2024–26)</strong> for batters, and <strong>&rho; = 0.983 (Q70)</strong>, <strong>&rho; = 0.999 (k=300/800)</strong>, and <strong>&rho; = 0.813 (2024–26)</strong> for bowlers.</p>

<div class="fig-wrap">
  <img src="data:image/png;base64,{img_b64('fig6_sensitivity_rank_stability.png')}" alt="Sensitivity Heatmap" style="width:66%; max-width:66%;">
  <div class="fig-cap"><strong>Figure 6:</strong> Rank stability heatmap across Q70/Q75/Q80 cutoffs, k=300/500/800 priors, IPL-only, and 2024–26 recency. T20I specialists shift only on IPL-only slices; 2024 retiree Dinesh Karthik drops naturally in 2024–26.</div>
</div>

<h3 style="margin:4px 0 2px 0;">Methodological Limitations &amp; Guardrails</h3>
<ul style="font-size:7.6pt; line-height:1.26; margin:2px 0 0 16px;">
  <li><strong>Missing Field/Matchup Micro-Context:</strong> The supplied dataset omits ball-tracking pitch coordinates (line/length maps), dew moisture, and fielder placements—all critical Triple Lock inputs supplied by RR’s ground scouts alongside analytics.</li>
  <li><strong>Volume &amp; Role Asymmetry:</strong> Requiring &ge;180 HP balls prevents 100-ball opener survivorship spikes (e.g., Jaiswal) from distorting Top-10 ranks, naturally tilting batters toward #3–#7 roles (#10 Ishan Kishan is the sole regular opener clearing all gates).</li>
  <li><strong>Bootstrap Bubble Proximity &amp; Availability Decoupling:</strong> Bubble ranks sit within narrow PRS margins (e.g., #10 Kishan 61.34 vs #11 Pooran 61.32; #9 Kuldeep 67.31 vs #10 Hasaranga 67.13), while pure historical rankings (e.g., Dinesh Karthik) are explicitly separated from active Dec 16, 2026 Goa auction availability above.</li>
</ul>

</body>
</html>
"""

with open(f'{REP}/SupeRR_Selector_S2_Methodology_Report.html', 'w', encoding='utf-8') as f:
    f.write(report_html)

# ==========================================================
# BUILD SELF-CONTAINED INTERACTIVE DASHBOARD HTML
# ==========================================================
bat_json = ebat[['rank','striker_name','bat_type','ipl_2026_team','t20i_team','tot_b','hp_b','hp_m','hp_sr','blended_hp_tva','lp_tva_eb','clutch_resid','hp_wpa_eb','pos_m_rate','PRS_100']].merge(
    micro_b[['striker_id','striker_name','hp_pon_tva','hp_poff_tva','hp_spin_tva','entry10_tva','recent_24_26_tva']].drop(columns=['striker_id']),
    on='striker_name', how='left'
).round(2).to_json(orient='records')
bowl_json = ebowl[['rank','bowler_name','bowl_style','pace_spin','ipl_2026_team','t20i_team','tot_leg','hp_leg','hp_m','hp_econ','hp_wkts','blended_hp_tva','lp_tva_eb','clutch_resid','hp_wpa_eb','pos_m_rate','PRS_100']].merge(
    micro_bw[['bowler_name','hp_pace_off_pct','hp_death_tva100','recent_24_26_tva']],
    on='bowler_name', how='left'
).round(2).to_json(orient='records')
rr_bat_json = rr_bat[rr_bat['tot_b']>=30].sort_values('tot_b', ascending=False).round(2).to_json(orient='records')
rr_bowl_json = rr_bowl[rr_bowl['tot_leg']>=30].sort_values('tot_leg', ascending=False).round(2).to_json(orient='records')
rr_aut_json = rr26_aut.round(2).to_json(orient='records')
death_pb_json = death_pb.round(2).to_json(orient='records')
rapm_b_json = rapm_b_top.round(2).to_json(orient='records')
rapm_bw_json = rapm_bw_top.round(2).to_json(orient='records')
trade_aud_json = trade_aud.round(2).to_json(orient='records')
temp_val_json = temp_val.round(4).to_json(orient='records')
venue_rr_json = venue_rr[venue_rr['hp_bowl'] == True].sort_values(['venue', 'pace_cat']).round(2).to_json(orient='records')
lhb_piv = lhb_rhb.pivot(index='bowler_name', columns='batting_type', values=['b', 'tva'])
lhb_piv.columns = [f"{a}_{b}" for a, b in lhb_piv.columns]
lhb_piv = top10_bw[['rank', 'bowler_name']].merge(lhb_piv.reset_index(), on='bowler_name', how='left')
lhb_rhb_json = lhb_piv.round(1).to_json(orient='records')
p11_ctx_json = p11_ctx.round(2).to_json(orient='records')
mb_bat_json = mb_bat.round(2).to_json(orient='records')
mb_bowl_json = mb_bowl.round(2).to_json(orient='records')
dual_aud_json = dual_aud.round(2).to_json(orient='records')
uni20_json = uni20.round(2).to_json(orient='records')

dash_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SupeRR Selector S2 — Interactive RAPM Pressure Scouting War Room</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #F1F5F9; color: #0F172A; }}
  header {{ background: linear-gradient(120deg, #131A40, #074EA2 60%, #EA1A85); color: white; padding: 20px 28px; }}
  header h1 {{ margin: 0 0 4px 0; font-size: 22px; }}
  header p {{ margin: 0; color: #E2E8F0; font-size: 13px; }}
  .container {{ max-width: 1340px; margin: 20px auto; padding: 0 20px; }}
  .tabs {{ display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; }}
  .tab-btn {{ padding: 9px 16px; border: none; border-radius: 6px; background: white; color: #131A40; font-weight: 700; cursor: pointer; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .tab-btn.active {{ background: #EA1A85; color: white; }}
  .panel {{ display: none; background: white; border-radius: 8px; padding: 18px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
  .panel.active {{ display: block; }}
  .controls {{ display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; align-items: center; }}
  input, select {{ padding: 7px 10px; border: 1px solid #CBD5E1; border-radius: 6px; font-size: 13px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.2px; }}
  th {{ background: #131A40; color: white; padding: 8px 8px; text-align: left; cursor: pointer; position: sticky; top: 0; }}
  th.num, td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td {{ padding: 7px 8px; border-bottom: 1px solid #E2E8F0; }}
  tr:hover td {{ background: #FDF2F8; }}
  .top-row td {{ background: #FFF7FB; font-weight: 600; }}
  .pos {{ color: #0D9488; font-weight: 700; }}
  .neg {{ color: #DC2626; font-weight: 700; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
  .charts-grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
  .charts-grid img {{ width: 100%; border: 1px solid #E2E8F0; border-radius: 8px; }}
</style>
</head>
<body>
<header>
  <h1>SupeRR Selector Season 2 — RAPM Pressure Analytics &amp; RR IPL 2027 Scouting War Room</h1>
  <p>200,999 Deliveries | 784 Top-Flight Matches (IPL + FM T20Is, 2022–2026) | OOF Player-vs-Player Ridge RAPM + Win-Probability Leverage</p>
</header>
<div class="container">
  <div class="tabs">
    <button class="tab-btn active" onclick="showTab('bat', this)">Batters Leaderboard (n=73)</button>
    <button class="tab-btn" onclick="showTab('bowl', this)">Bowlers Leaderboard (n=79)</button>
    <button class="tab-btn" onclick="showTab('rr', this)">RR Autopsy, Trades &amp; Venue Physics</button>
    <button class="tab-btn" onclick="showTab('playbook', this)">Execution Playbook, OOS Proof &amp; LHB/RHB</button>
    <button class="tab-btn" onclick="showTab('charts', this)">Visual Analytics Gallery (10 Figures)</button>
  </div>

  <div id="bat" class="panel active">
    <div class="controls">
      <input type="text" id="searchBat" placeholder="Search batter, IPL team, or country..." oninput="renderBat()" style="min-width:260px;">
      <select id="filterBatIPL" onchange="renderBat()">
        <option value="all">All Eligible Batters</option>
        <option value="top10">Top 10 Only (With 3-Vector + Entry-10)</option>
        <option value="ipl26">Active IPL 2026 Players</option>
      </select>
    </div>
    <div style="overflow-x:auto; max-height:640px;">
      <table>
        <thead><tr>
          <th onclick="sortBat('rank')">Rank</th><th onclick="sortBat('striker_name')">Batter</th><th>IPL 2026 Team</th><th>T20I</th>
          <th class="num" onclick="sortBat('PRS_100')">PRS</th><th class="num" onclick="sortBat('hp_b')">HP Balls</th><th class="num" onclick="sortBat('hp_m')">M</th>
          <th class="num" onclick="sortBat('hp_sr')">HP SR</th><th class="num" onclick="sortBat('blended_hp_tva')">HP TVA</th>
          <th class="num" onclick="sortBat('lp_tva_eb')">LP TVA</th><th class="num" onclick="sortBat('clutch_resid')">Clutch Resid</th>
          <th class="num">Pace-On</th><th class="num">Pace-Off</th><th class="num">Spin</th><th class="num">Entry-10</th>
          <th class="num" onclick="sortBat('pos_m_rate')">Pos %</th>
        </tr></thead>
        <tbody id="batBody"></tbody>
      </table>
    </div>
  </div>

  <div id="bowl" class="panel">
    <div class="controls">
      <input type="text" id="searchBowl" placeholder="Search bowler, style, IPL team..." oninput="renderBowl()" style="min-width:260px;">
      <select id="filterBowlType" onchange="renderBowl()">
        <option value="all">All Pace &amp; Spin</option>
        <option value="Pace">Pace Only</option>
        <option value="Spin">Spin Only</option>
        <option value="top10">Top 10 Only (With Death &amp; Pace-Off %)</option>
      </select>
    </div>
    <div style="overflow-x:auto; max-height:640px;">
      <table>
        <thead><tr>
          <th onclick="sortBowl('rank')">Rank</th><th onclick="sortBowl('bowler_name')">Bowler</th><th>Type</th><th>IPL 2026 Team</th>
          <th class="num" onclick="sortBowl('PRS_100')">PRS</th><th class="num" onclick="sortBowl('hp_leg')">HP Balls</th><th class="num" onclick="sortBowl('hp_m')">M</th>
          <th class="num" onclick="sortBowl('hp_econ')">HP Econ</th><th class="num" onclick="sortBowl('hp_wkts')">Wkts</th>
          <th class="num" onclick="sortBowl('blended_hp_tva')">HP TVA</th><th class="num" onclick="sortBowl('lp_tva_eb')">LP TVA</th>
          <th class="num" onclick="sortBowl('clutch_resid')">Clutch Resid</th>
          <th class="num">Death TVA</th><th class="num">2024–26 TVA</th><th class="num" onclick="sortBowl('pos_m_rate')">Pos %</th>
        </tr></thead>
        <tbody id="bowlBody"></tbody>
      </table>
    </div>
  </div>

  <div id="rr" class="panel">
    <h3>RR IPL 2026 Win (9 Matches) vs. Loss (7 Matches) Pressure Autopsy</h3>
    <table><thead><tr><th>Outcome</th><th>RR Role</th><th>Pressure Tier</th><th class="num">Balls</th><th class="num">Total TVA</th><th class="num">TVA / 100 Balls</th><th class="num">Total WPA</th></tr></thead><tbody id="rrAutBody"></tbody></table>
    <h3 style="margin-top:18px;">Pre-Retention Front-Office Trade Audit (Nov 2026) &amp; Auction Fit Verdict</h3>
    <table><thead><tr><th>Reported Trade Corridor</th><th>Player</th><th class="num">Rank</th><th class="num">PRS</th><th class="num">HP Balls</th><th class="num">HP Econ</th><th class="num">HP TVA/100*</th></tr></thead><tbody id="tradeAudBody"></tbody></table>
    <div class="grid2" style="margin-top:18px;">
      <div>
        <h3>RR Home Fortresses (High Pressure): Jaipur &amp; Guwahati</h3>
        <table><thead><tr><th>Venue</th><th>Bowling Type</th><th class="num">HP Balls</th><th class="num">HP Econ</th><th class="num">Bdry %</th></tr></thead><tbody id="venueRrBody"></tbody></table>
      </div>
      <div>
        <h3>Playing-XI Role, Impact Sub &amp; Captaincy HP Context</h3>
        <table><thead><tr><th>Cohort / Context</th><th class="num">HP Balls</th><th class="num">HP Rate</th><th class="num">HP TVA/100</th></tr></thead><tbody id="p11CtxBody"></tbody></table>
      </div>
    </div>
    <div class="grid2" style="margin-top:18px;">
      <div>
        <h3>Moneyball Radar: Hidden Gem Batters (&ge;70 IPL HPb)</h3>
        <table><thead><tr><th>Batter</th><th class="num">HP Balls (M)</th><th class="num">HP SR</th><th class="num">Raw HP TVA</th><th class="num">HP WPA/100</th></tr></thead><tbody id="mbBatBody"></tbody></table>
      </div>
      <div>
        <h3>Moneyball Radar: Hidden Gem Bowlers (&ge;70 IPL HPb)</h3>
        <table><thead><tr><th>Bowler</th><th class="num">HP Balls (M)</th><th class="num">HP Econ (W)</th><th class="num">Raw HP TVA</th><th class="num">HP WPA/100</th></tr></thead><tbody id="mbBowlBody"></tbody></table>
      </div>
    </div>
    <h3 style="margin-top:18px;">Rajasthan Royals IPL 2026 Roster — Batting Under Pressure (Top-Flight 2022–2026)</h3>
    <table><thead><tr><th>Player</th><th class="num">Total Balls</th><th class="num">HP Balls</th><th class="num">HP Matches</th><th class="num">HP SR</th><th class="num">HP TVA/100 (Raw)</th><th class="num">LP TVA/100 (Raw)</th></tr></thead><tbody id="rrBatBody"></tbody></table>
    <h3 style="margin-top:18px;">Rajasthan Royals IPL 2026 Roster — Bowling Under Pressure (Top-Flight 2022–2026)</h3>
    <table><thead><tr><th>Player</th><th class="num">Total Legal Balls</th><th class="num">HP Balls</th><th class="num">HP Matches</th><th class="num">HP Econ</th><th class="num">HP TVA/100 (Raw)</th><th class="num">LP TVA/100 (Raw)</th></tr></thead><tbody id="rrBowlBody"></tbody></table>
  </div>

  <div id="playbook" class="panel">
    <h3>High-Pressure Death Delivery Playbook (Overs 16–20, &ge;150 Balls)</h3>
    <table><thead><tr><th>Pace Category</th><th>Delivery Style</th><th class="num">Balls</th><th class="num">Econ</th><th class="num">Wkt %</th><th class="num">Dot %</th><th class="num">Bdry %</th><th class="num">RAPM TVA/100</th></tr></thead><tbody id="deathPbBody"></tbody></table>
    <div class="grid2" style="margin-top:18px;">
      <div>
        <h3>Out-of-Time Temporal Validation (2022–24 &rarr; 2025–26 HP TVA)</h3>
        <table><thead><tr><th>Cohort</th><th>2022–24 Predictor</th><th class="num">Pearson r</th><th class="num">p-val</th><th class="num">Spearman &rho;</th></tr></thead><tbody id="tempValBody"></tbody></table>
      </div>
      <div>
        <h3>Top-10 Bowlers vs LHB &amp; RHB Under High Pressure</h3>
        <table><thead><tr><th>Rank</th><th>Bowler</th><th class="num">LHB (Balls)</th><th class="num">RHB (Balls)</th></tr></thead><tbody id="lhbRhbBody"></tbody></table>
      </div>
    </div>
    <div class="grid2" style="margin-top:18px;">
      <div>
        <h3>Unified Cross-Discipline Top-20 Master Index (Peak PRS)</h3>
        <table><thead><tr><th>#</th><th>Player</th><th>Role</th><th>IPL 2026 Team</th><th class="num">Peak PRS</th><th class="num">Net HP/M</th></tr></thead><tbody id="uni20Body"></tbody></table>
      </div>
      <div>
        <h3>Dual-Qualified All-Rounders Net Reality Check (&ge;180 HPb Both)</h3>
        <table><thead><tr><th>#</th><th>Player</th><th class="num">Bat (# / TVA)</th><th class="num">Bowl (# / TVA)</th><th class="num">Net HP/M</th></tr></thead><tbody id="dualAudBody"></tbody></table>
      </div>
    </div>
    <div class="grid2" style="margin-top:18px;">
      <div>
        <h3>Toughest Bowlers Faced (RAPM Ridge Opponent Adj.)</h3>
        <table><thead><tr><th>Bowler</th><th class="num">Balls*100 Score</th><th class="num">RAPM Bat Adj/100</th></tr></thead><tbody id="rapmBBody"></tbody></table>
      </div>
      <div>
        <h3>Toughest Batters Bowled To (RAPM Ridge Opponent Adj.)</h3>
        <table><thead><tr><th>Batter</th><th class="num">Balls*100 Score</th><th class="num">RAPM Bowl Adj/100</th></tr></thead><tbody id="rapmBwBody"></tbody></table>
      </div>
    </div>
  </div>

  <div id="charts" class="panel">
    <div class="charts-grid">
      <img src="data:image/png;base64,{img_b64('fig1_context_inflation.png')}">
      <img src="data:image/png;base64,{img_b64('fig2_wp_calibration_and_situations.png')}">
      <img src="data:image/png;base64,{img_b64('fig3_batters_quadrant_top10.png')}">
      <img src="data:image/png;base64,{img_b64('fig4_bowlers_quadrant_top10.png')}">
      <img src="data:image/png;base64,{img_b64('fig8_3vector_and_death_playbook.png')}">
      <img src="data:image/png;base64,{img_b64('fig5_bootstrap_forest_plots.png')}">
      <img src="data:image/png;base64,{img_b64('fig9_rr_loss_autopsy_and_strokes.png')}">
      <img src="data:image/png;base64,{img_b64('fig7_rr_2027_auction_fit.png')}">
      <img src="data:image/png;base64,{img_b64('fig10_mechanisms_and_temporal_proof.png')}">
      <img src="data:image/png;base64,{img_b64('fig6_sensitivity_rank_stability.png')}">
    </div>
  </div>
</div>

<script>
const batData = {bat_json};
const bowlData = {bowl_json};
const rrBat = {rr_bat_json};
const rrBowl = {rr_bowl_json};
const rrAut = {rr_aut_json};
const deathPb = {death_pb_json};
const rapmB = {rapm_b_json};
const rapmBw = {rapm_bw_json};
const tradeAud = {trade_aud_json};
const tempVal = {temp_val_json};
const venueRr = {venue_rr_json};
const lhbRhb = {lhb_rhb_json};
const p11Ctx = {p11_ctx_json};
const mbBat = {mb_bat_json};
const mbBowl = {mb_bowl_json};
const dualAud = {dual_aud_json};
const uni20 = {uni20_json};
let batSort = {{k:'rank', asc:true}};
let bowlSort = {{k:'rank', asc:true}};

function showTab(id, btn) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  btn.classList.add('active');
}}
function fmtS(v) {{ return v==null ? '—' : (v>=0?'+':'')+v.toFixed(1); }}
function sortBat(k) {{ batSort = {{k, asc: batSort.k===k ? !batSort.asc : (k==='rank')}}; renderBat(); }}
function sortBowl(k) {{ bowlSort = {{k, asc: bowlSort.k===k ? !bowlSort.asc : (k==='rank')}}; renderBowl(); }}

function renderBat() {{
  const q = document.getElementById('searchBat').value.toLowerCase();
  const f = document.getElementById('filterBatIPL').value;
  let rows = batData.filter(r => {{
    if (f==='top10' && r.rank > 10) return false;
    if (f==='ipl26' && r.ipl_2026_team === 'Non-IPL26') return false;
    return (r.striker_name+' '+r.ipl_2026_team+' '+r.t20i_team).toLowerCase().includes(q);
  }});
  rows.sort((a,b) => (a[batSort.k] > b[batSort.k] ? 1 : -1) * (batSort.asc ? 1 : -1));
  document.getElementById('batBody').innerHTML = rows.map(r => `
    <tr class="${{r.rank<=10?'top-row':''}}">
      <td>#${{r.rank}}</td><td>${{r.striker_name}}</td><td>${{r.ipl_2026_team}}</td><td>${{r.t20i_team}}</td>
      <td class="num"><b>${{r.PRS_100.toFixed(1)}}</b></td><td class="num">${{r.hp_b}}</td><td class="num">${{r.hp_m}}</td>
      <td class="num">${{r.hp_sr.toFixed(1)}}</td>
      <td class="num ${{r.blended_hp_tva>=0?'pos':'neg'}}">${{fmtS(r.blended_hp_tva)}}</td>
      <td class="num">${{fmtS(r.lp_tva_eb)}}</td>
      <td class="num ${{r.clutch_resid>=0?'pos':'neg'}}">${{fmtS(r.clutch_resid)}}</td>
      <td class="num">${{fmtS(r.hp_pon_tva)}}</td><td class="num">${{fmtS(r.hp_poff_tva)}}</td><td class="num">${{fmtS(r.hp_spin_tva)}}</td><td class="num">${{fmtS(r.entry10_tva)}}</td>
      <td class="num">${{Math.round(r.pos_m_rate*100)}}%</td>
    </tr>`).join('');
}}

function renderBowl() {{
  const q = document.getElementById('searchBowl').value.toLowerCase();
  const f = document.getElementById('filterBowlType').value;
  let rows = bowlData.filter(r => {{
    if (f==='top10' && r.rank > 10) return false;
    if ((f==='Pace'||f==='Spin') && r.pace_spin !== f) return false;
    return (r.bowler_name+' '+r.bowl_style+' '+r.ipl_2026_team+' '+r.t20i_team).toLowerCase().includes(q);
  }});
  rows.sort((a,b) => (a[bowlSort.k] > b[bowlSort.k] ? 1 : -1) * (bowlSort.asc ? 1 : -1));
  document.getElementById('bowlBody').innerHTML = rows.map(r => `
    <tr class="${{r.rank<=10?'top-row':''}}">
      <td>#${{r.rank}}</td><td>${{r.bowler_name}}</td><td>${{r.pace_spin}}</td><td>${{r.ipl_2026_team}}</td>
      <td class="num"><b>${{r.PRS_100.toFixed(1)}}</b></td><td class="num">${{r.hp_leg}}</td><td class="num">${{r.hp_m}}</td>
      <td class="num">${{r.hp_econ.toFixed(2)}}</td><td class="num">${{r.hp_wkts}}</td>
      <td class="num ${{r.blended_hp_tva>=0?'pos':'neg'}}">${{fmtS(r.blended_hp_tva)}}</td>
      <td class="num">${{fmtS(r.lp_tva_eb)}}</td>
      <td class="num ${{r.clutch_resid>=0?'pos':'neg'}}">${{fmtS(r.clutch_resid)}}</td>
      <td class="num">${{fmtS(r.hp_death_tva100)}}</td><td class="num">${{fmtS(r.recent_24_26_tva)}}</td>
      <td class="num">${{Math.round(r.pos_m_rate*100)}}%</td>
    </tr>`).join('');
}}

document.getElementById('rrAutBody').innerHTML = rrAut.map(r => `
  <tr><td><b>${{r.rr_won?'RR Win (n=9)':'RR Loss (n=7)'}}</b></td><td>${{r.rr_role}}</td><td>${{r.is_hp?'High Pressure':'Usual Pressure'}}</td>
  <td class="num">${{r.balls}}</td><td class="num">${{fmtS(r.tva_sum)}}</td>
  <td class="num ${{r.tva_100>=0?'pos':'neg'}}">${{fmtS(r.tva_100)}}</td>
  <td class="num ${{r.wpa_sum>=0?'pos':'neg'}}">${{(r.wpa_sum>=0?'+':'')+r.wpa_sum.toFixed(2)}}</td></tr>`).join('');

document.getElementById('rrBatBody').innerHTML = rrBat.map(r => `
  <tr><td><b>${{r.striker_name}}</b></td><td class="num">${{r.tot_b}}</td><td class="num">${{r.hp_b}}</td><td class="num">${{r.hp_m}}</td>
  <td class="num">${{r.hp_sr!=null?r.hp_sr.toFixed(1):'—'}}</td>
  <td class="num ${{r.hp_tva100>=0?'pos':'neg'}}">${{fmtS(r.hp_tva100)}}</td>
  <td class="num">${{fmtS(r.lp_tva100)}}</td></tr>`).join('');

document.getElementById('rrBowlBody').innerHTML = rrBowl.map(r => `
  <tr><td><b>${{r.bowler_name}}</b></td><td class="num">${{r.tot_leg}}</td><td class="num">${{r.hp_leg}}</td><td class="num">${{r.hp_m}}</td>
  <td class="num">${{r.hp_econ!=null?r.hp_econ.toFixed(2):'—'}}</td>
  <td class="num ${{r.hp_tva100>=0?'pos':'neg'}}">${{fmtS(r.hp_tva100)}}</td>
  <td class="num">${{fmtS(r.lp_tva100)}}</td></tr>`).join('');

document.getElementById('deathPbBody').innerHTML = deathPb.map(r => `
  <tr><td>${{r.pace_cat}}</td><td><b>${{r.delivery_style}}</b></td><td class="num">${{r.balls}}</td><td class="num">${{r.econ.toFixed(2)}}</td>
  <td class="num">${{r.wkt_pct.toFixed(1)}}%</td><td class="num">${{r.dot_pct.toFixed(1)}}%</td><td class="num">${{r.bdry_pct.toFixed(1)}}%</td>
  <td class="num ${{r.tva100>=0?'pos':'neg'}}">${{fmtS(r.tva100)}}</td></tr>`).join('');

document.getElementById('rapmBBody').innerHTML = rapmB.map(r => `
  <tr><td><b>${{r.bowler_name}}</b></td><td class="num">${{Math.round(r.count/100)}}</td><td class="num neg">${{r.mean.toFixed(1)}}</td></tr>`).join('');

document.getElementById('rapmBwBody').innerHTML = rapmBw.map(r => `
  <tr><td><b>${{r.striker_name}}</b></td><td class="num">${{Math.round(r.count/100)}}</td><td class="num neg">${{r.mean.toFixed(1)}}</td></tr>`).join('');

document.getElementById('tradeAudBody').innerHTML = tradeAud.map(r => `
  <tr><td><b>${{r.move}}</b></td><td>${{r.player}}</td><td class="num">${{r.rank!=null?'#'+r.rank:'Unqual'}}</td><td class="num">${{r.prs!=null?r.prs.toFixed(1):'—'}}</td>
  <td class="num">${{r.hp_leg}}</td><td class="num">${{r.hp_econ.toFixed(2)}}</td>
  <td class="num ${{(r.hp_tva!=null?r.hp_tva:r.raw_hp_tva)>=0?'pos':'neg'}}">${{fmtS(r.hp_tva!=null?r.hp_tva:r.raw_hp_tva)}}</td></tr>`).join('');

document.getElementById('venueRrBody').innerHTML = venueRr.map(r => `
  <tr><td><b>${{r.venue}}</b></td><td>${{r.pace_cat}}</td><td class="num">${{r.balls}}</td><td class="num">${{r.econ.toFixed(2)}}</td><td class="num">${{r.bdry.toFixed(1)}}%</td></tr>`).join('');

document.getElementById('p11CtxBody').innerHTML = p11Ctx.map(r => `
  <tr><td><b>${{r.factor}}</b></td><td class="num">${{r.n}}</td><td class="num">${{r.rate.toFixed(1)}}</td>
  <td class="num ${{r.tva100>=0?'pos':'neg'}}">${{fmtS(r.tva100)}}</td></tr>`).join('');

document.getElementById('mbBatBody').innerHTML = mbBat.map(r => `
  <tr><td><b>${{r.striker_name}}</b></td><td class="num">${{r.hp_b}} (${{r.hp_m}})</td><td class="num">${{r.hp_sr.toFixed(1)}}</td>
  <td class="num ${{r.hp_tva>=0?'pos':'neg'}}">${{fmtS(r.hp_tva)}}</td><td class="num">${{fmtS(r.hp_wpa)}}</td></tr>`).join('');

document.getElementById('mbBowlBody').innerHTML = mbBowl.map(r => `
  <tr><td><b>${{r.bowler_name}}</b></td><td class="num">${{r.hp_leg}} (${{r.hp_m}})</td><td class="num">${{r.hp_econ.toFixed(2)}} (${{r.hp_w}})</td>
  <td class="num ${{r.hp_tva>=0?'pos':'neg'}}">${{fmtS(r.hp_tva)}}</td><td class="num">${{fmtS(r.hp_wpa)}}</td></tr>`).join('');

document.getElementById('tempValBody').innerHTML = tempVal.map(r => `
  <tr><td>${{r.role}}</td><td><b>${{r.metric}}</b></td>
  <td class="num ${{r.pearson_r>=0?'pos':'neg'}}">${{(r.pearson_r>=0?'+':'')+r.pearson_r.toFixed(3)}}</td>
  <td class="num">${{r.p_val.toFixed(4)}}</td>
  <td class="num ${{r.spearman_rho>=0?'pos':'neg'}}">${{(r.spearman_rho>=0?'+':'')+r.spearman_rho.toFixed(3)}}</td></tr>`).join('');

document.getElementById('lhbRhbBody').innerHTML = lhbRhb.map(r => `
  <tr><td>#${{r.rank}}</td><td><b>${{r.bowler_name}}</b></td>
  <td class="num ${{(r.tva_LHB||0)>=0?'pos':'neg'}}">${{fmtS(r.tva_LHB)}} <span style="color:#64748B;font-weight:400;">(${{r.b_LHB!=null?r.b_LHB:0}})</span></td>
  <td class="num ${{(r.tva_RHB||0)>=0?'pos':'neg'}}">${{fmtS(r.tva_RHB)}} <span style="color:#64748B;font-weight:400;">(${{r.b_RHB!=null?r.b_RHB:0}})</span></td></tr>`).join('');

document.getElementById('uni20Body').innerHTML = uni20.map(r => `
  <tr class="${{r.unified_rank<=10?'top-row':''}}"><td>#${{r.unified_rank}}</td><td><b>${{r.name}}</b></td><td>${{r.primary_role}}</td><td>${{r.team}}</td>
  <td class="num"><b>${{r.best_prs.toFixed(1)}}</b></td>
  <td class="num ${{r.net_hp_tva_per_match>=0?'pos':'neg'}}">${{(r.net_hp_tva_per_match>=0?'+':'')+r.net_hp_tva_per_match.toFixed(2)}}</td></tr>`).join('');

document.getElementById('dualAudBody').innerHTML = dualAud.map(r => `
  <tr><td>#${{r.net_rank}}</td><td><b>${{r.name}}</b></td>
  <td class="num ${{(r.tva_bat||0)>=0?'pos':'neg'}}">#${{Math.round(r.rank_bat)}} (${{fmtS(r.tva_bat)}})</td>
  <td class="num ${{(r.tva_bowl||0)>=0?'pos':'neg'}}">#${{Math.round(r.rank_bowl)}} (${{fmtS(r.tva_bowl)}})</td>
  <td class="num ${{r.net_hp_tva_per_match>=0?'pos':'neg'}}">${{(r.net_hp_tva_per_match>=0?'+':'')+r.net_hp_tva_per_match.toFixed(2)}}</td></tr>`).join('');

renderBat();
renderBowl();
</script>
</body>
</html>"""

with open(f'{REP}/SupeRR_Selector_S2_Dashboard.html', 'w', encoding='utf-8') as f:
    f.write(dash_html)

nav_links = (
    '  <div style="margin-top:9px; display:flex; gap:10px; flex-wrap:wrap;">'
    '<a href="./report/SupeRR_Selector_S2_Methodology_Report.html" target="_blank" style="color:#FFFFFF; background:rgba(255,255,255,0.16); padding:5px 12px; border-radius:6px; text-decoration:none; font-size:12px; font-weight:700; border:1px solid rgba(255,255,255,0.3);">Full Methodology Report (Web HTML)</a>'
    '<a href="./report/SupeRR_Selector_S2_Methodology_Report.pdf" target="_blank" style="color:#131A40; background:#FDE047; padding:5px 12px; border-radius:6px; text-decoration:none; font-size:12px; font-weight:700;">Open 11-Page Submission PDF</a>'
    '</div>\n</header>'
)
root_index = dash_html.replace('</header>', nav_links, 1)
with open(r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\index.html', 'w', encoding='utf-8') as f:
    f.write(root_index)

print("Generated Methodology Report HTML, Interactive Dashboard HTML & root index.html successfully!")
