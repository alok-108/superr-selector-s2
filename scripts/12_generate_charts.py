import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\data_pack\Data Pack'
OUT = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\outputs'
FIG = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\report\figures'
os.makedirs(FIG, exist_ok=True)

# Clean typography & style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial'],
    'axes.edgecolor': '#CBD5E1',
    'axes.linewidth': 0.8,
    'axes.titlesize': 12.5,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': '#FAFCFF',
    'savefig.dpi': 220,
})

PINK = '#EA1A85'
NAVY = '#131A40'
BLUE = '#074EA2'
GOLD = '#E59500'
TEAL = '#0D9488'
CRIMSON = '#DC2626'
SLATE = '#64748B'
LIGHT_SLATE = '#CBD5E1'

with open(f'{OUT}/summary_metrics.json') as f:
    meta = json.load(f)

tier_season = pd.read_csv(f'{OUT}/tier_season_summary.csv')
cal_df = pd.read_csv(f'{OUT}/wp_calibration_bins.csv')
wp_val = pd.read_csv(f'{OUT}/wp_validation_metrics.csv')
sit_df = pd.read_csv(f'{OUT}/situation_validation.csv')
ebat = pd.read_csv(f'{OUT}/eligible_batters.csv')
ebowl = pd.read_csv(f'{OUT}/eligible_bowlers.csv')
top10_b = pd.read_csv(f'{OUT}/top10_batters.csv')
top10_bw = pd.read_csv(f'{OUT}/top10_bowlers.csv')
micro_b = pd.read_csv(f'{OUT}/micro_tactics_batters.csv')
micro_bw = pd.read_csv(f'{OUT}/micro_tactics_bowlers.csv')
death_pb = pd.read_csv(f'{OUT}/death_delivery_playbook.csv')
stroke_pb = pd.read_csv(f'{OUT}/hp_stroke_playbook.csv')
rr26_aut = pd.read_csv(f'{OUT}/rr26_win_loss_autopsy.csv')
sens_b = pd.read_csv(f'{OUT}/sensitivity_batters.csv')
sens_bw = pd.read_csv(f'{OUT}/sensitivity_bowlers.csv')
rr_bat = pd.read_csv(f'{OUT}/rr_roster_bat.csv')
rr_bowl = pd.read_csv(f'{OUT}/rr_roster_bowl.csv')
dip_b = pd.read_csv(f'{OUT}/batters_biggest_dip.csv')
dip_bw = pd.read_csv(f'{OUT}/bowlers_biggest_dip.csv')
pi_samp = pd.read_csv(f'{OUT}/pi_sample_for_plots.csv')

# Also compute exact Over 1..20 RPO from ball_by_ball for Fig 1B
bbb = pd.read_csv(f'{BASE}/ball_by_ball.csv', usecols=['innings','overid','runs','extras1_runs','extras_2_runs','extras_type','is_wicket'], low_memory=False)
reg_b = bbb[bbb['innings'].isin([1,2])].copy()
reg_b['is_legal'] = ~reg_b['extras_type'].isin(['wide','Overstep','other'])
reg_b['tot_runs'] = reg_b['runs'] + reg_b['extras1_runs'] + reg_b['extras_2_runs']
over_curve = reg_b.groupby('overid').agg(runs=('tot_runs','sum'), leg=('is_legal','sum'), wkts=('is_wicket','sum')).reset_index()
over_curve['rpo'] = 6.0 * over_curve['runs'] / over_curve['leg']

# ==========================================
# FIGURE 1: Era Scoring Inflation & T20 Phase Physics
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))

# Panel A: IPL vs FM T20I RPO by year
ipl_ts = tier_season[tier_season['match_tier']=='IPL'].sort_values('season')
t20i_fm = tier_season[(tier_season['match_tier']=='T20I_FM_vs_FM') & (tier_season['season'].isin(['2022','2023','2024','2025','2026']))].sort_values('season')
yrs = [2022, 2023, 2024, 2025, 2026]
ax1.plot(yrs, ipl_ts['rpo'], marker='o', lw=2.8, ms=8, color=PINK, label='IPL (2022–2026)')
ax1.plot(yrs, t20i_fm['rpo'], marker='s', lw=2.5, ms=7, color=BLUE, label='Full-Member T20I')
# Horizontal baselines for Associate & Emerging
assoc_rpo = tier_season[tier_season['match_tier']=='T20I_Assoc_vs_Assoc']['runs'].sum() / (tier_season[tier_season['match_tier']=='T20I_Assoc_vs_Assoc']['legal_balls'].sum()/6)
emerg_rpo = tier_season[tier_season['match_tier']=='Emerging_A']['runs'].sum() / (tier_season[tier_season['match_tier']=='Emerging_A']['legal_balls'].sum()/6)
ax1.axhline(emerg_rpo, ls='--', lw=1.6, color=TEAL, alpha=0.85, label=f'Emerging / Rising Asia Cup ({emerg_rpo:.2f} RPO)')
ax1.axhline(assoc_rpo, ls=':', lw=1.6, color=SLATE, alpha=0.85, label=f'Associate T20Is ({assoc_rpo:.2f} RPO)')

for x, y in zip(yrs, ipl_ts['rpo']):
    ax1.annotate(f'{y:.2f}', (x, y), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=8.5, fontweight='bold', color=PINK)
for x, y in zip(yrs, t20i_fm['rpo']):
    off = -14 if x in [2022, 2026] else -15
    ax1.annotate(f'{y:.2f}', (x, y), xytext=(0, off), textcoords='offset points', ha='center', fontsize=8.5, fontweight='bold', color=BLUE)

ax1.set_title('A. Scoring-Environment Inflation (+15.7% IPL RPO Surge)', pad=12, color=NAVY)
ax1.set_ylabel('Average Run Rate (Runs Per Over)')
ax1.set_xticks(yrs)
ax1.set_ylim(7.2, 10.45)
ax1.grid(axis='y', ls='--', alpha=0.35)
ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor=LIGHT_SLATE, fontsize=8.3)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

# Panel B: Over-by-Over RPO & Phase Dynamic Wicket Value
bars = ax2.bar(over_curve['overid'], over_curve['rpo'], color=[
    BLUE if o <= 6 else (SLATE if o <= 15 else PINK) for o in over_curve['overid']
], alpha=0.85, width=0.72)
ax2.set_title('B. T20 Over Run-Rate Curve & Dynamic Wicket Value', pad=12, color=NAVY)
ax2.set_xlabel('Over Number')
ax2.set_ylabel('Mean Scoring Rate (RPO)', color=NAVY)
ax2.set_xticks(range(1, 21))
ax2.set_ylim(0, 14.0)
ax2.grid(axis='y', ls='--', alpha=0.35)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

# Overlay phase wicket value annotations cleanly
ph_wv = pi_samp.groupby('phase')['wicket_value'].mean()
ax2.text(3.5, 11.4, f"Powerplay (1–6)\nMean V_W = {ph_wv['PP']:.1f} runs", ha='center', fontsize=8.5, fontweight='bold',
         color=BLUE, bbox=dict(boxstyle='round,pad=0.35', facecolor='#EFF6FF', edgecolor=BLUE, alpha=0.9))
ax2.text(11.0, 10.7, f"Middle Overs (7–15)\nMean V_W = {ph_wv['Middle']:.1f} runs", ha='center', fontsize=8.5, fontweight='bold',
         color=NAVY, bbox=dict(boxstyle='round,pad=0.35', facecolor='#F1F5F9', edgecolor=SLATE, alpha=0.9))
ax2.text(18.0, 12.8, f"Death (16–20)\nV_W = {ph_wv['Death']:.1f} runs", ha='center', fontsize=8.5, fontweight='bold',
         color=PINK, bbox=dict(boxstyle='round,pad=0.35', facecolor='#FDF2F8', edgecolor=PINK, alpha=0.9))

plt.tight_layout(pad=2.0)
plt.savefig(f'{FIG}/fig1_context_inflation.png', bbox_inches='tight')
plt.close()

# ==========================================
# FIGURE 2: WP Calibration & Pressure Index Situation Validation
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={'width_ratios': [1.0, 1.35]})

# Panel A: Calibration curve
ax1.plot([0, 1], [0, 1], ls='--', color=SLATE, lw=1.5, label='Perfect Calibration')
ax1.plot(cal_df['mean_pred'], cal_df['actual_win'], marker='o', lw=2.6, ms=7.5, color=PINK, label='Innings 2 WP Model (5-Fold OOF)')
ax1.fill_between(cal_df['mean_pred'], cal_df['mean_pred'] - 0.03, cal_df['mean_pred'] + 0.03, color=PINK, alpha=0.08)
ax1.set_title('A. Win Probability Calibration (94,808 Chase Balls)', pad=12, color=NAVY)
ax1.set_xlabel('Predicted Chasing Win Probability')
ax1.set_ylabel('Observed Match Win Frequency')
ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.02)
ax1.grid(True, ls='--', alpha=0.3)
ax1.legend(loc='upper left', frameon=True, facecolor='white', edgecolor=LIGHT_SLATE, fontsize=8.5)
i2_row = wp_val[wp_val['model'].str.contains('Innings 2')].iloc[0]
ax1.text(0.58, 0.14, f"5-Fold Match-Grouped OOF\nROC-AUC = {i2_row['roc_auc']:.3f}\nBrier Score = {i2_row['brier_score']:.3f}\nLog-Loss = {i2_row['log_loss']:.3f}\nRes. Model R² = {meta['res_r2_2026']:.3f}",
         fontsize=8.5, color=NAVY, bbox=dict(boxstyle='round,pad=0.45', facecolor='white', edgecolor=BLUE, alpha=0.95))
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

# Panel B: Situation Validity Horizontal Grouped Bars
sit_ord = sit_df.sort_values('mean_PI_bat', ascending=True).reset_index(drop=True)
y = np.arange(len(sit_ord))
h = 0.36
ax2.barh(y + h/2, sit_ord['mean_PI_bat'], height=h, color=PINK, label='Batting Pressure Index (PI_bat)')
ax2.barh(y - h/2, sit_ord['mean_PI_bowl'], height=h, color=BLUE, label='Bowling Pressure Index (PI_bowl)')
ax2.axvline(meta['q75_pi_bat'], color=PINK, ls='--', lw=1.2, alpha=0.7)
ax2.axvline(meta['q75_pi_bowl'], color=BLUE, ls=':', lw=1.2, alpha=0.7)
ax2.text(meta['q75_pi_bat']+1.0, 0.3, f"Q75 Cutoff\nBat: {meta['q75_pi_bat']:.1f} | Bowl: {meta['q75_pi_bowl']:.1f}", fontsize=7.8, color=NAVY,
         bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor=LIGHT_SLATE, alpha=0.9))

for idx, row in sit_ord.iterrows():
    ax2.text(row['mean_PI_bat'] + 1.0, idx + h/2, f"{row['mean_PI_bat']:.1f}", va='center', fontsize=8, fontweight='bold', color=PINK)
    ax2.text(row['mean_PI_bowl'] + 1.0, idx - h/2, f"{row['mean_PI_bowl']:.1f}", va='center', fontsize=8, fontweight='bold', color=BLUE)

ax2.set_yticks(y)
ax2.set_yticklabels(sit_ord['situation'], fontsize=8.5)
ax2.set_xlim(0, 95)
ax2.set_xlabel('Mean Pressure Index (0–100 Scale)')
ax2.set_title('B. Pressure Index Separation Across Real Match Contexts', pad=12, color=NAVY)
ax2.grid(axis='x', ls='--', alpha=0.35)
ax2.legend(loc='lower right', frameon=True, facecolor='white', edgecolor=LIGHT_SLATE, fontsize=8.3)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.tight_layout(pad=2.0)
plt.savefig(f'{FIG}/fig2_wp_calibration_and_situations.png', bbox_inches='tight')
plt.close()

# ==========================================
# FIGURE 3: Top 10 Batters Quadrant (Usual vs High Pressure)
# ==========================================
fig, ax = plt.subplots(figsize=(11.5, 6.8))

non_top = ebat[~ebat['striker_id'].isin(top10_b['striker_id']) & ~ebat['striker_id'].isin(dip_b.head(5)['striker_id'])]
ax.scatter(non_top['lp_tva_eb'], non_top['blended_hp_tva'], s=np.sqrt(non_top['hp_b'])*3.8,
           color=SLATE, alpha=0.38, edgecolor='white', lw=0.6, label='Other Eligible Batters (n=73)')

# Regression line (with floored LP visualization for LP>=0 + raw trend)
xs = np.linspace(-15, 32, 100)
ys = meta['bat_hp_lp_intercept'] + meta['bat_hp_lp_slope'] * np.maximum(0.0, xs)
ax.plot(xs, ys, color=NAVY, ls='--', lw=1.8, label=f"Expected HP Baseline (slope={meta['bat_hp_lp_slope']:.2f}, LP floored at 0)")
ax.axhline(0, color='#94A3B8', lw=0.8)
ax.axvline(0, color='#94A3B8', lw=0.8)

# Top 10 Batters
ax.scatter(top10_b['lp_tva_eb'], top10_b['blended_hp_tva'], s=np.sqrt(top10_b['hp_b'])*5.5,
           color=PINK, alpha=0.92, edgecolor=NAVY, lw=1.2, zorder=5, label='Top 10 Batters Under Pressure (Rank 1–10)')

# Steepest Negative Clutch Dippers
dip_show = dip_b.head(5)
ax.scatter(dip_show['lp_tva_eb'], dip_show['blended_hp_tva'], s=np.sqrt(dip_show['hp_b'])*4.5,
           color=CRIMSON, alpha=0.85, marker='X', edgecolor='white', lw=0.8, zorder=4, label='Biggest Pressure Dippers')

# Custom non-overlapping label offsets for Top 10 Batters
offsets_b = {
    'Suryakumar Yadav': (-48, 12),
    'Tim David': (11, -3),
    'Namboori Tilak Varma': (-30, 14),
    'Shreyas Iyer': (11, 8),
    'Shashank Singh': (-58, 15),
    'David Miller': (-48, 14),
    'Dinesh Karthik': (12, -13),
    'Heinrich Klaasen': (11, -8),
    'Dasun Shanaka': (-52, -12),
    'Ishan Kishan': (-14, -20),
}
for _, r in top10_b.iterrows():
    nm = 'Tilak Varma' if 'Tilak' in r['striker_name'] else r['striker_name']
    dx, dy = offsets_b.get(r['striker_name'], (6, 6))
    ax.annotate(f"#{int(r['rank'])} {nm}", (r['lp_tva_eb'], r['blended_hp_tva']),
                xytext=(dx, dy), textcoords='offset points', fontsize=8.4, fontweight='bold', color=NAVY,
                arrowprops=dict(arrowstyle='-', color='#64748B', lw=0.75))

offsets_dip_b = {
    'Aiden Markram': (9, -3),
    'Charith Asalanka': (9, -11),
    'Moeen Ali': (9, -4),
    'Mitchell Marsh': (9, -5),
    'Krunal Pandya': (-25, -15),
}
for _, r in dip_show.iterrows():
    dx, dy = offsets_dip_b.get(r['striker_name'], (6, -10))
    ax.annotate(f"{r['striker_name']}", (r['lp_tva_eb'], r['blended_hp_tva']),
                xytext=(dx, dy), textcoords='offset points', fontsize=8.0, color=CRIMSON, style='italic')

ax.text(21.5, 27.2, "ELITE IN ALL CONTEXTS\n& SURGE UNDER PRESSURE", fontsize=8.2, color=TEAL, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#F0FDFA', edgecolor=TEAL, alpha=0.8))
ax.text(-9.2, 16.5, "PRESSURE LIFTERS\n(Elevate Above Normal Level)", fontsize=8.2, color=BLUE, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#EFF6FF', edgecolor=BLUE, alpha=0.8))
ax.text(19.5, -10.5, "PRESSURE DIPPERS\n(Drop Below Baseline)", fontsize=8.2, color=CRIMSON, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF2F2', edgecolor=CRIMSON, alpha=0.8))

ax.set_title('Batting Under Pressure vs. Usual Performance (73 Eligible Top-Flight Batters, RAPM-Adjusted)', pad=14, color=NAVY)
ax.set_xlabel('Usual / Low-Pressure Performance (EB-Shrunk RAPM Context TVA per 100 Balls)')
ax.set_ylabel('High-Pressure Performance (Blended EB-Shrunk RAPM TVA per 100 Balls)')
ax.set_xlim(-14.5, 31.5); ax.set_ylim(-14.5, 30.0)
ax.grid(True, ls='--', alpha=0.3)
ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor=LIGHT_SLATE, fontsize=8.5)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{FIG}/fig3_batters_quadrant_top10.png', bbox_inches='tight')
plt.close()

# ==========================================
# FIGURE 4: Top 10 Bowlers Quadrant (Usual vs High Pressure)
# ==========================================
fig, ax = plt.subplots(figsize=(11.5, 6.8))

non_top_bw = ebowl[~ebowl['bowler_id'].isin(top10_bw['bowler_id']) & ~ebowl['bowler_id'].isin(dip_bw.head(5)['bowler_id'])]
ax.scatter(non_top_bw['lp_tva_eb'], non_top_bw['blended_hp_tva'], s=np.sqrt(non_top_bw['hp_leg'])*3.8,
           color=SLATE, alpha=0.38, edgecolor='white', lw=0.6, label='Other Eligible Bowlers (n=79)')

xs_bw = np.linspace(-15, 22, 100)
ys_bw = meta['bowl_hp_lp_intercept'] + meta['bowl_hp_lp_slope'] * np.maximum(0.0, xs_bw)
ax.plot(xs_bw, ys_bw, color=NAVY, ls='--', lw=1.8, label=f"Expected HP Baseline (slope={meta['bowl_hp_lp_slope']:.2f}, LP floored at 0)")
ax.axhline(0, color='#94A3B8', lw=0.8)
ax.axvline(0, color='#94A3B8', lw=0.8)

ax.scatter(top10_bw['lp_tva_eb'], top10_bw['blended_hp_tva'], s=np.sqrt(top10_bw['hp_leg'])*5.5,
           color=BLUE, alpha=0.92, edgecolor=NAVY, lw=1.2, zorder=5, label='Top 10 Bowlers Under Pressure (Rank 1–10)')

dip_bw_show = dip_bw.head(5)
ax.scatter(dip_bw_show['lp_tva_eb'], dip_bw_show['blended_hp_tva'], s=np.sqrt(dip_bw_show['hp_leg'])*4.5,
           color=CRIMSON, alpha=0.85, marker='X', edgecolor='white', lw=0.8, zorder=4, label='Biggest Pressure Dippers')

offsets_bw = {
    'Krunal Pandya': (-16, 22),
    'Axar Patel': (14, -10),
    'Bhuvneshwar Kumar': (-95, 22),
    'Sunil Narine': (-38, 28),
    'Josh Hazlewood': (14, 8),
    'Jasprit Bumrah': (14, -18),
    'Nathan Ellis': (14, 24),
    'Rashid Khan': (14, -15),
    'Kuldeep Yadav': (-78, -2),
    'Wanindu Hasaranga': (-92, -16),
}
for _, r in top10_bw.iterrows():
    dx, dy = offsets_bw.get(r['bowler_name'], (6, 6))
    ax.annotate(f"#{int(r['rank'])} {r['bowler_name']}", (r['lp_tva_eb'], r['blended_hp_tva']),
                xytext=(dx, dy), textcoords='offset points', fontsize=8.3, fontweight='bold', color=NAVY,
                arrowprops=dict(arrowstyle='-', color='#64748B', lw=0.8))

offsets_dip_bw = {
    'Prasidh Krishna': (9, -4),
    'Nuwan Thushara': (9, -9),
    'Prince Yadav': (9, -3),
    'Shardul Thakur': (-28, -15),
    'Marco Jansen': (-66, 8),
}
for _, r in dip_bw_show.iterrows():
    dx, dy = offsets_dip_bw.get(r['bowler_name'], (6, -8))
    ax.annotate(f"{r['bowler_name']}", (r['lp_tva_eb'], r['blended_hp_tva']),
                xytext=(dx, dy), textcoords='offset points', fontsize=8.0, color=CRIMSON, style='italic')

ax.set_title('Bowling Under Pressure vs. Usual Performance (79 Eligible Top-Flight Bowlers, RAPM-Adjusted)', pad=14, color=NAVY)
ax.set_xlabel('Usual / Low-Pressure Performance (EB-Shrunk RAPM Context TVA per 100 Balls)')
ax.set_ylabel('High-Pressure Performance (Blended EB-Shrunk RAPM TVA per 100 Balls)')
ax.set_xlim(-13.5, 26.5); ax.set_ylim(-13.5, 20.0)
ax.grid(True, ls='--', alpha=0.3)
ax.legend(loc='upper left', frameon=True, facecolor='white', edgecolor=LIGHT_SLATE, fontsize=8.5)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{FIG}/fig4_bowlers_quadrant_top10.png', bbox_inches='tight')
plt.close()

# ==========================================
# FIGURE 5: 1000-Iter Match-Clustered Bootstrap 95% CIs (Forest Plots)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

for ax, df_t, col_name, color, title in [
    (ax1, top10_b, 'striker_name', PINK, 'A. Top 10 Batters: 95% Bootstrap CI & Positive Rate'),
    (ax2, top10_bw, 'bowler_name', BLUE, 'B. Top 10 Bowlers: 95% Bootstrap CI & Positive Rate'),
]:
    d = df_t.sort_values('rank', ascending=False).reset_index(drop=True)
    y = np.arange(len(d))
    ax.axvline(0, color=CRIMSON, ls='--', lw=1.2, alpha=0.8)
    ax.hlines(y, d['ci_low_95'], d['ci_high_95'], color=color, lw=2.6, alpha=0.75)
    ax.scatter(d['boot_mean'], y, color=NAVY, s=48, zorder=5, label='Bootstrap Mean HP TVA/100')
    ax.scatter(d['ci_low_95'], y, color=color, s=22, marker='|', lw=2)
    ax.scatter(d['ci_high_95'], y, color=color, s=22, marker='|', lw=2)
    
    lbls = [f"#{int(r['rank'])} {('Tilak Varma' if 'Tilak' in r[col_name] else r[col_name])} ({int(r['hp_m'])}m)" for _, r in d.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(lbls, fontsize=8.8, fontweight='bold', color=NAVY)
    for idx, r in d.iterrows():
        ax.text(r['ci_high_95'] + 1.8, idx, f"p>0: {100*r['p_gt_zero']:.0f}% | PosMatch: {100*r['pos_m_rate']:.0f}%",
                va='center', fontsize=7.8, color=SLATE)
    ax.set_xlim(min(-30, d['ci_low_95'].min() - 5), d['ci_high_95'].max() + 50)
    ax.set_xlabel('Unshrunk High-Pressure Context TVA / 100 Balls (95% Match-Clustered CI)')
    ax.set_title(title, pad=12, color=NAVY)
    ax.grid(axis='x', ls='--', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

plt.tight_layout(pad=2.2, w_pad=2.8)
plt.savefig(f'{FIG}/fig5_bootstrap_forest_plots.png', bbox_inches='tight')
plt.close()

# ==========================================
# FIGURE 6: Sensitivity & Rank Stability Matrix (Top 10s across Specs)
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))

for ax, sdf, n_col, title in [
    (ax1, sens_b.head(10), 'striker_name', 'A. Top 10 Batters Rank Stability Across Specs'),
    (ax2, sens_bw.head(10), 'bowler_name', 'B. Top 10 Bowlers Rank Stability Across Specs'),
]:
    cols = ['Primary_Q75_K500', 'Q70_K500', 'Q80_K500', 'Q75_K300', 'Q75_K800', 'IPL_Only', 'Recent_24_26']
    col_lbls = ['Primary\n(Q75,K500)', 'Cutoff\nQ70', 'Cutoff\nQ80', 'Shrink\nK=300', 'Shrink\nK=800', 'IPL\nOnly', 'Recent\n2024–26']
    mat = sdf[cols].values.astype(float)
    # Invert color intensity so Rank 1 = darkest/best, >10 = light, NaN = grey
    disp = np.where(np.isnan(mat), 32, np.clip(mat, 1, 26))
    im = ax.imshow(disp, cmap='RdPu_r' if n_col=='striker_name' else 'Blues_r', aspect='auto', vmin=1, vmax=28)
    
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(col_lbls, fontsize=7.9, fontweight='bold', color=NAVY)
    ax.set_yticks(np.arange(len(sdf)))
    ax.set_yticklabels([f"#{i+1} {('Tilak Varma' if 'Tilak' in nm else nm)}" for i, nm in enumerate(sdf[n_col])], fontsize=8.8, fontweight='bold', color=NAVY)
    
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = '—' if np.isnan(v) else f'{int(v)}'
            txt_col = 'white' if (not np.isnan(v) and v <= 9) else NAVY
            ax.text(j, i, txt, ha='center', va='center', fontsize=8.6, fontweight='bold', color=txt_col)
    ax.set_title(title, pad=12, color=NAVY)

plt.tight_layout(pad=2.0)
plt.savefig(f'{FIG}/fig6_sensitivity_rank_stability.png', bbox_inches='tight')
plt.close()

# ==========================================
# FIGURE 7: Rajasthan Royals IPL 2026 Audit & 2027 Auction Fit
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

# Panel A: RR Core Batters & Bowlers (min hp >= 80 & tot >= 200 to exclude tailender batting noise)
rb_sel = rr_bat[(rr_bat['hp_b'] >= 80) & (rr_bat['tot_b'] >= 200)].copy()
rb_sel['label'] = rb_sel['striker_name'] + ' (Bat, ' + rb_sel['hp_b'].astype(int).astype(str) + 'b)'
rb_sel['role'] = 'Bat'
rbw_sel = rr_bowl[(rr_bowl['hp_leg'] >= 80) & (rr_bowl['tot_leg'] >= 200)].copy()
rbw_sel['label'] = rbw_sel['bowler_name'] + ' (Bowl, ' + rbw_sel['hp_leg'].astype(int).astype(str) + 'b)'
rbw_sel['role'] = 'Bowl'

rr_comb = pd.concat([
    rb_sel[['label','hp_tva100','lp_tva100','role']],
    rbw_sel[['label','hp_tva100','lp_tva100','role']]
]).dropna().sort_values('hp_tva100', ascending=True).reset_index(drop=True)

y = np.arange(len(rr_comb))
for i, r in rr_comb.iterrows():
    c_line = TEAL if r['hp_tva100'] >= 5 else (SLATE if r['hp_tva100'] >= -5 else CRIMSON)
    ax1.plot([r['lp_tva100'], r['hp_tva100']], [i, i], color=c_line, lw=2.2, alpha=0.8)
ax1.scatter(rr_comb['lp_tva100'], y, color=SLATE, s=46, label='Usual Pressure TVA/100', zorder=4)
ax1.scatter(rr_comb['hp_tva100'], y, color=[TEAL if v >= 0 else CRIMSON for v in rr_comb['hp_tva100']],
            s=68, edgecolor=NAVY, lw=0.8, label='High-Pressure TVA/100', zorder=5)
ax1.axvline(0, color=NAVY, ls='--', lw=1.0, alpha=0.7)
ax1.set_yticks(y)
ax1.set_yticklabels(rr_comb['label'], fontsize=8.3, fontweight='bold', color=NAVY)
ax1.set_xlabel('Context-Adjusted TVA per 100 Balls (Raw HP vs Usual)')
ax1.set_title('A. RR Core Roster Audit (≥80 HP Balls): Strengths vs. Leaks', pad=12, color=NAVY)
ax1.grid(axis='x', ls='--', alpha=0.3)
ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor=LIGHT_SLATE, fontsize=8.2)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

# Panel B: Priority Auction / Trade Fit Targets for RR (from Top 10 + IPL-only Top 15 gems!)
targets = [
    ('David Miller (OS Finisher, DC)', ebat, 'striker_name', 'David Miller', PINK),
    ('Heinrich Klaasen (OS Middle/Spin, SRH)', ebat, 'striker_name', 'Heinrich Klaasen', PINK),
    ('Krunal Pandya (IND Spin AR, RCB)', ebowl, 'bowler_name', 'Krunal Pandya', BLUE),
    ('Nathan Ellis (OS Death Pace, CSK→SRH)*', ebowl, 'bowler_name', 'Nathan Ellis', SLATE),
    ('Kuldeep Yadav (IND Wrist-Spin, DC)', ebowl, 'bowler_name', 'Kuldeep Yadav', BLUE),
    ('Wanindu Hasaranga (OS Wrist-Spin, LSG)', ebowl, 'bowler_name', 'Wanindu Hasaranga', BLUE),
    ('Matheesha Pathirana (OS Death Pace, KKR)', ebowl, 'bowler_name', 'Matheesha Pathirana', BLUE),
    ('Lungi Ngidi (OS Death Pace, DC)**', ebowl, 'bowler_name', 'Lungi Ngidi', GOLD),
    ('Ashutosh Sharma (IND Finisher, DC)**', pd.read_csv(f'{OUT}/ipl_only_top15_batters.csv'), 'striker_name', 'Ashutosh Sharma', GOLD),
    ('Harpreet Brar (IND Spin, PBKS)**', pd.read_csv(f'{OUT}/ipl_only_top15_bowlers.csv'), 'bowler_name', 'Harpreet Brar', GOLD),
]
t_rows = []
for lbl, df_src, ncol, nm, col in targets:
    row = df_src[df_src[ncol]==nm].iloc[0]
    t_rows.append({'label': lbl, 'prs': row['PRS_100'], 'hp_tva': row['blended_hp_tva'], 'col': col})
tdf = pd.DataFrame(t_rows).sort_values('prs', ascending=True).reset_index(drop=True)

y2 = np.arange(len(tdf))
ax2.barh(y2, tdf['prs'], color=tdf['col'], height=0.64, alpha=0.88, edgecolor=NAVY, lw=0.7)
for i, r in tdf.iterrows():
    ax2.text(r['prs'] + 0.8, i, f"PRS {r['prs']:.1f}  (+{r['hp_tva']:.1f} TVA)", va='center', fontsize=7.8, fontweight='bold', color=NAVY)
ax2.set_yticks(y2)
ax2.set_yticklabels(tdf['label'], fontsize=8.0, fontweight='bold', color=NAVY)
ax2.set_xlim(45, 108)
ax2.set_xlabel('Pressure Reliability Score (PRS, 50 = Avg, +15/SD)')
ax2.set_title('B. High-Priority IPL 2027 RR Targets (*Traded CSK→SRH Nov 26)', pad=12, color=NAVY)
ax2.grid(axis='x', ls='--', alpha=0.3)
p1 = mpatches.Patch(color=PINK, label='Top-10 Batter')
p2 = mpatches.Patch(color=BLUE, label='Top-11 Bowler')
p3 = mpatches.Patch(color=GOLD, label='IPL-Only / Value**')
p4 = mpatches.Patch(color=SLATE, label='Locked via Trade*')
ax2.legend(handles=[p1, p2, p3, p4], loc='upper center', bbox_to_anchor=(0.48, -0.14), ncol=4, frameon=True, facecolor='white', edgecolor=LIGHT_SLATE, fontsize=7.5)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.tight_layout(pad=2.0)
plt.savefig(f'{FIG}/fig7_rr_2027_auction_fit.png', bbox_inches='tight')
plt.close()

# ==========================================
# FIGURE 8: 3-Vector Matchup Heatmap & High-Pressure Death Execution Playbook
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.6), gridspec_kw={'width_ratios': [1.05, 1.15]})

# Panel A: Top-10 Batters 3-Vector (Pace-On, Pace-Off, Spin) + First-10-Ball Entry Impact
m_ord = top10_b[['striker_id','rank','striker_name']].merge(micro_b, on=['striker_id','striker_name']).sort_values('rank')
vec_cols = ['hp_pon_tva', 'hp_poff_tva', 'hp_spin_tva', 'entry10_tva']
vec_lbls = ['Pace-On\nHP TVA', 'Pace-Off\nHP TVA', 'Spin\nHP TVA', 'Entry 1–10b\nHP TVA']
mat_v = m_ord[vec_cols].values.astype(float)
im1 = ax1.imshow(np.clip(mat_v, -35, 75), cmap='RdYlGn', aspect='auto', vmin=-35, vmax=75)
ax1.set_xticks(np.arange(len(vec_cols)))
ax1.set_xticklabels(vec_lbls, fontsize=8.4, fontweight='bold', color=NAVY)
ax1.set_yticks(np.arange(len(m_ord)))
ax1.set_yticklabels([f"#{int(r['rank'])} {('Tilak Varma' if 'Tilak' in r['striker_name'] else r['striker_name'])}" for _, r in m_ord.iterrows()],
                    fontsize=8.6, fontweight='bold', color=NAVY)
for i in range(mat_v.shape[0]):
    for j in range(mat_v.shape[1]):
        v = mat_v[i, j]
        txt = '—' if np.isnan(v) else f'{v:+.1f}'
        tcol = 'white' if (not np.isnan(v) and (v > 48 or v < -22)) else NAVY
        ax1.text(j, i, txt, ha='center', va='center', fontsize=8.4, fontweight='bold', color=tcol)
ax1.set_title('A. Top-10 Batters: Pace-On vs. Pace-Off vs. Spin & Entry Impact', pad=12, color=NAVY)

# Panel B: High-Pressure Death Delivery Playbook (Best 6 vs Worst 6 by TVA/100)
dp_top = death_pb.head(6)
dp_bot = death_pb.tail(6)
dp_show = pd.concat([dp_bot, dp_top]).drop_duplicates().sort_values('tva100', ascending=True).reset_index(drop=True)
y_dp = np.arange(len(dp_show))
cols_dp = [CRIMSON if v < 0 else (TEAL if v > 15 else BLUE) for v in dp_show['tva100']]
ax2.barh(y_dp, dp_show['tva100'], color=cols_dp, height=0.64, alpha=0.88, edgecolor=NAVY, lw=0.6)
ax2.axvline(0, color=NAVY, lw=1.0)
for i, r in dp_show.iterrows():
    lbl_val = f" {r['tva100']:+.1f}  ({r['econ']:.1f} Econ | {r['wkt_pct']:.1f}% Wkt)"
    if r['tva100'] >= 0:
        ax2.text(r['tva100'] + 1.5, i, lbl_val, va='center', ha='left', fontsize=7.8, fontweight='bold', color=NAVY)
    else:
        ax2.text(1.5, i, lbl_val, va='center', ha='left', fontsize=7.8, fontweight='bold', color=CRIMSON)
ax2.set_yticks(y_dp)
ax2.set_yticklabels([f"{r['delivery_style']} ({r['pace_cat']})" for _, r in dp_show.iterrows()], fontsize=8.3, fontweight='bold', color=NAVY)
ax2.set_xlim(-40, 128)
ax2.set_xlabel('High-Pressure Death RAPM Context TVA / 100 Balls')
ax2.set_title('B. High-Pressure Death Delivery Playbook (Ov 16–20, ≥150 Balls)', pad=12, color=NAVY)
ax2.grid(axis='x', ls='--', alpha=0.3)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.tight_layout(pad=2.0)
plt.savefig(f'{FIG}/fig8_3vector_and_death_playbook.png', bbox_inches='tight')
plt.close()

# ==========================================
# FIGURE 9: RR IPL 2026 Win-vs-Loss Autopsy & High-Pressure Shot Playbook
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.1), gridspec_kw={'width_ratios': [1.0, 1.15]})

# Panel A: RR 2026 Win (n=9) vs Loss (n=7) TVA/100 breakdown
cats = [
    ('HP Bowling\n(The Decider)', 'RR_Bowl', True),
    ('LP Bowling', 'RR_Bowl', False),
    ('HP Batting', 'RR_Bat', True),
    ('LP Batting', 'RR_Bat', False),
]
win_vals = []; loss_vals = []
for _, role, hp in cats:
    w_r = rr26_aut[(rr26_aut['rr_won']==True) & (rr26_aut['rr_role']==role) & (rr26_aut['is_hp']==hp)]
    l_r = rr26_aut[(rr26_aut['rr_won']==False) & (rr26_aut['rr_role']==role) & (rr26_aut['is_hp']==hp)]
    win_vals.append(float(w_r['tva_100'].iloc[0]))
    loss_vals.append(float(l_r['tva_100'].iloc[0]))

y_a = np.arange(len(cats))
h_a = 0.36
ax1.barh(y_a + h_a/2, win_vals, height=h_a, color=TEAL, edgecolor=NAVY, lw=0.6, label='RR Wins (n=9 Matches)')
ax1.barh(y_a - h_a/2, loss_vals, height=h_a, color=CRIMSON, edgecolor=NAVY, lw=0.6, label='RR Losses (n=7 Matches)')
ax1.axvline(0, color=NAVY, lw=1.0)
for i, (wv, lv) in enumerate(zip(win_vals, loss_vals)):
    ax1.text(wv + (1.2 if wv >= 0 else -1.2), i + h_a/2, f"{wv:+.1f}", va='center', ha='left' if wv >= 0 else 'right', fontsize=8.4, fontweight='bold', color=TEAL)
    ax1.text(lv + (1.2 if lv >= 0 else -1.2), i - h_a/2, f"{lv:+.1f}", va='center', ha='left' if lv >= 0 else 'right', fontsize=8.4, fontweight='bold', color=CRIMSON)
ax1.set_yticks(y_a)
ax1.set_yticklabels([c[0] for c in cats], fontsize=8.8, fontweight='bold', color=NAVY)
ax1.set_xlim(-30, 40)
ax1.set_xlabel('Team Context-Adjusted TVA per 100 Balls')
ax1.set_title('A. RR IPL 2026 Win-vs-Loss Autopsy: 29.8 TVA/100 HP Bowling Swing', pad=12, color=NAVY)
ax1.grid(axis='x', ls='--', alpha=0.3)
ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor=LIGHT_SLATE, fontsize=8.3)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

# Panel B: High-Pressure Batting Stroke Playbook (Best 6 Attacking vs Worst 6 Leaking/Defensive)
st_ord = pd.concat([stroke_pb.head(6), stroke_pb.tail(6)]).drop_duplicates().sort_values('tva100', ascending=True).reset_index(drop=True)
y_st = np.arange(len(st_ord))
cols_st = [CRIMSON if v < 0 else PINK for v in st_ord['tva100']]
ax2.barh(y_st, st_ord['tva100'], color=cols_st, height=0.64, alpha=0.88, edgecolor=NAVY, lw=0.6)
ax2.axvline(0, color=NAVY, lw=1.0)
for i, r in st_ord.iterrows():
    lbl_s = f" {r['tva100']:+.1f}  (SR {r['sr']:.0f} | Out {r['out_pct']:.1f}%)"
    if r['tva100'] >= 0:
        ax2.text(r['tva100'] + 2.0, i, lbl_s, va='center', ha='left', fontsize=7.6, fontweight='bold', color=NAVY)
    else:
        ax2.text(2.5, i, lbl_s, va='center', ha='left', fontsize=7.6, fontweight='bold', color=CRIMSON)
ax2.set_yticks(y_st)
ax2.set_yticklabels(st_ord['stroke'], fontsize=8.4, fontweight='bold', color=NAVY)
ax2.set_xlim(-95, 178)
ax2.set_xlabel('High-Pressure Batting RAPM Context TVA / 100 Balls')
ax2.set_title('B. High-Pressure Batting Stroke Value Hierarchy (≥200 Balls)', pad=12, color=NAVY)
ax2.grid(axis='x', ls='--', alpha=0.3)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.tight_layout(pad=2.0)
plt.savefig(f'{FIG}/fig9_rr_loss_autopsy_and_strokes.png', bbox_inches='tight')
plt.close()

# ==========================================
# FIGURE 10: Temporal Out-of-Sample Proof & Causal Pressure Mechanisms
# ==========================================
temp_df = pd.read_csv(f'{OUT}/temporal_prediction_benchmark.csv')
mech_df = pd.read_csv(f'{OUT}/clutch_mechanisms_summary.csv')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), gridspec_kw={'width_ratios': [1.05, 1.15]})

# Panel A: 2022-24 -> 2025-26 Out-of-Time Correlation with Future HP RAPM TVA/100
metrics_ord = [
    'Naive Clutch Delta (HP - LP)',
    'Unshrunk HP RAPM TVA',
    'Full RAPM PRS Model',
    'Raw HP Rate (HP SR / -HP Econ)',
    'Blended EB+RAPM HP TVA',
    'Overall Baseline (SR / -Econ)',
]
short_mlbl = [
    'Naive Clutch\n(Raw HP - LP)',
    'Unshrunk HP\nRAPM TVA',
    'Full RAPM\nPRS Model',
    'Raw HP Rate\n(SR / -Econ)',
    'Blended EB +\nRAPM HP TVA',
    'Overall Class\n(SR / -Econ)',
]
b_vals = [float(temp_df[(temp_df['role'].str.contains('Batters')) & (temp_df['metric']==m)]['pearson_r'].iloc[0]) for m in metrics_ord]
bw_vals = [float(temp_df[(temp_df['role'].str.contains('Bowlers')) & (temp_df['metric']==m)]['pearson_r'].iloc[0]) for m in metrics_ord]

y_m = np.arange(len(metrics_ord))
h_m = 0.36
ax1.barh(y_m + h_m/2, b_vals, height=h_m, color=PINK, edgecolor=NAVY, lw=0.6, label='Batters (n=50)')
ax1.barh(y_m - h_m/2, bw_vals, height=h_m, color=BLUE, edgecolor=NAVY, lw=0.6, label='Bowlers (n=54)')
ax1.axvline(0, color=CRIMSON, lw=1.2, ls='--')
for i, (bv, bwv) in enumerate(zip(b_vals, bw_vals)):
    ax1.text(bv + (0.012 if bv>=0 else -0.012), i + h_m/2, f"{bv:+.3f}", va='center', ha='left' if bv>=0 else 'right', fontsize=8.0, fontweight='bold', color=PINK)
    ax1.text(bwv + (0.012 if bwv>=0 else -0.012), i - h_m/2, f"{bwv:+.3f}", va='center', ha='left' if bwv>=0 else 'right', fontsize=8.0, fontweight='bold', color=BLUE)
ax1.set_yticks(y_m)
ax1.set_yticklabels(short_mlbl, fontsize=8.3, fontweight='bold', color=NAVY)
ax1.set_xlim(-0.14, 0.46)
ax1.set_xlabel('Pearson Correlation (r) With Future 2025–26 HP RAPM TVA/100')
ax1.set_title('A. Temporal Prediction (2022–24 → 2025–26): Naive Clutch Is Noise', pad=12, color=NAVY)
ax1.grid(axis='x', ls='--', alpha=0.3)
ax1.legend(loc='lower right', frameon=True, facecolor='white', edgecolor=LIGHT_SLATE, fontsize=8.2)
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False)

# Panel B: What Physically Changes from Usual (LP) -> High Pressure (HP)?
def get_d(co, reg_v, col):
    return float(mech_df[(mech_df['cohort']==co) & (mech_df['regime']==reg_v)][col].iloc[0])

mech_labels = [
    'Bat: Boundary % Shift\n(HP - LP)',
    'Bat: Dot Ball % Shift\n(HP - LP)',
    'Bowl: Dot Ball % Shift\n(HP - LP)',
    'Bowl: Elite Var. % Shift\n(Yorker/Cutter/Carrom)',
    'Bowl: Full Toss/Straight\nError Rate under HP',
]
top_m = [
    get_d('Top-10 Batters','HP','bdry_pct') - get_d('Top-10 Batters','LP','bdry_pct'),
    get_d('Top-10 Batters','HP','dot_pct') - get_d('Top-10 Batters','LP','dot_pct'),
    get_d('Top-10 Bowlers','HP','dot_pct') - get_d('Top-10 Bowlers','LP','dot_pct'),
    get_d('Top-10 Bowlers','HP','var_pct') - get_d('Top-10 Bowlers','LP','var_pct'),
    get_d('Top-10 Bowlers','HP','err_pct'),
]
dip_m = [
    get_d('Dip-10 Batters (>=500b)','HP','bdry_pct') - get_d('Dip-10 Batters (>=500b)','LP','bdry_pct'),
    get_d('Dip-10 Batters (>=500b)','HP','dot_pct') - get_d('Dip-10 Batters (>=500b)','LP','dot_pct'),
    get_d('Dip-10 Bowlers (>=500b)','HP','dot_pct') - get_d('Dip-10 Bowlers (>=500b)','LP','dot_pct'),
    get_d('Dip-10 Bowlers (>=500b)','HP','var_pct') - get_d('Dip-10 Bowlers (>=500b)','LP','var_pct'),
    get_d('Dip-10 Bowlers (>=500b)','HP','err_pct'),
]
y_mc = np.arange(len(mech_labels))
ax2.barh(y_mc + h_m/2, top_m, height=h_m, color=TEAL, edgecolor=NAVY, lw=0.6, label='Top-10 Clutch Performers')
ax2.barh(y_mc - h_m/2, dip_m, height=h_m, color=CRIMSON, edgecolor=NAVY, lw=0.6, label='Bottom-10 Clutch Dippers (≥500b)')
ax2.axvline(0, color=NAVY, lw=1.0)
for i, (tv, dv) in enumerate(zip(top_m, dip_m)):
    fmt_t = f"{tv:.1f}%" if 'under HP' in mech_labels[i] else f"{tv:+.1f} pp"
    fmt_d = f"{dv:.1f}%" if 'under HP' in mech_labels[i] else f"{dv:+.1f} pp"
    ax2.text(tv + (0.6 if tv>=0 else -0.6), i + h_m/2, fmt_t, va='center', ha='left' if tv>=0 else 'right', fontsize=7.9, fontweight='bold', color=TEAL)
    ax2.text(dv + (0.6 if dv>=0 else -0.6), i - h_m/2, fmt_d, va='center', ha='left' if dv>=0 else 'right', fontsize=7.9, fontweight='bold', color=CRIMSON)
ax2.set_yticks(y_mc)
ax2.set_yticklabels(mech_labels, fontsize=8.1, fontweight='bold', color=NAVY)
ax2.set_xlim(-11.5, 24.5)
ax2.set_xlabel('Magnitude / Percentage-Point Shift (LP → HP)')
ax2.set_title('B. Mechanisms: Why Top-10 Hold Firm While Dippers Collapse', pad=12, color=NAVY)
ax2.grid(axis='x', ls='--', alpha=0.3)
ax2.legend(loc='lower right', frameon=True, facecolor='white', edgecolor=LIGHT_SLATE, fontsize=8.0)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

plt.tight_layout(pad=2.0)
plt.savefig(f'{FIG}/fig10_mechanisms_and_temporal_proof.png', bbox_inches='tight')
plt.close()

print("All 10 figures generated successfully in:", FIG)
