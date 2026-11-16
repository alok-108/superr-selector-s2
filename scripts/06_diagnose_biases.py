import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

OUT = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\outputs'
reg = pd.read_pickle(f'{OUT}/enriched_balls.pkl')

# Re-compute wicket_value & refined PI_bat, PI_bowl quickly
i1_full = reg[(reg['innings']==1)&(reg['max_balls']==120)&(reg['has_valid_outcome'])].copy()
i1_full['runs_to_go'] = i1_full['inn1_total'] - i1_full['cum_runs_before']
mdl_rem = HistGradientBoostingRegressor(max_iter=80, max_depth=4, min_samples_leaf=500, l2_regularization=20.0, monotonic_cst=[1,1,1], random_state=42)
mdl_rem.fit(i1_full[['balls_rem','wickets_rem','env_par_rpo']], i1_full['runs_to_go'])

df_minus = reg[['balls_rem','wickets_rem','env_par_rpo']].copy()
df_minus['wickets_rem'] = (df_minus['wickets_rem'] - 1).clip(lower=0)
rem_w_minus = np.where(reg['wickets_rem']==1, 0.0, np.maximum(0.0, mdl_rem.predict(df_minus)))
reg['wicket_value'] = np.clip(reg['exp_rem_runs'] - rem_w_minus, 2.0, 28.0)

live_gate = np.where(
    reg['innings'] == 2,
    np.clip((np.minimum(reg['wp_before'], 1.0 - reg['wp_before']) - 0.03) / 0.14, 0.0, 1.0),
    np.clip((np.minimum(reg['wp_before'], 1.0 - reg['wp_before']) - 0.05) / 0.15, 0.35, 1.0)
)
exp_wkts = 0.042 * reg['legal_balls_before']
collapse_excess = np.maximum(0.0, reg['cum_wkts_before'] - exp_wkts)
collapse_stress = np.where(reg['cum_wkts_before'] >= 2, np.tanh(collapse_excess / 1.5), 0.0) * live_gate
bat_shock = np.tanh(np.maximum(0.0, reg['wkts_last18'] - 1.0) / 1.2) * live_gate
reg['rrr'] = np.where(reg['innings']==2, 6.0 * np.maximum(0, reg['target'] - reg['cum_runs_before']) / reg['balls_rem'], 0.0)
rrr_excess = np.maximum(0.0, reg['rrr'] - reg['env_par_rpo'])
rrr_stress = np.where(reg['innings']==2, np.tanh(rrr_excess / 4.0) * live_gate, 0.0)
death_ramp = np.where(reg['overid'] >= 16, 0.55 + 0.45 * (reg['overid'] - 16) / 4.0, 0.0) * live_gate
lev_score = np.where(reg['innings'] == 2, np.tanh(reg['wp_lev'] / 0.048), 0.55 * np.tanh(reg['wp_lev'] / 0.025) * live_gate)
bowl_defense_risk = np.where(reg['innings'] == 2, np.clip((reg['wp_before'] - 0.12) / 0.50, 0.0, 1.0) * live_gate, 0.22 * live_gate)
bowl_surge = np.tanh(np.maximum(0.0, reg['runs_last18'] - 26.0) / 15.0) * live_gate
ko_boost = np.where(reg['is_knockout'] == 1, 0.12, 0.0) * np.maximum(0.35, live_gate)

reg['PI_bat'] = 100.0 * np.clip(0.38*lev_score + 0.34*np.maximum(collapse_stress, 0.6*bat_shock) + 0.24*rrr_stress + 0.24*death_ramp + ko_boost, 0, 1)
reg['PI_bowl'] = 100.0 * np.clip(0.40*lev_score + 0.32*death_ramp + 0.20*bowl_defense_risk + 0.18*bowl_surge + ko_boost, 0, 1)

reg['raa_bat'] = reg['batter_runs'] - reg['exp_bat_runs']
reg['waa_bat'] = reg['wicket_value'] * (reg['exp_bat_out'] - reg['is_striker_out'].astype(float))
reg['tva_bat'] = reg['raa_bat'] + reg['waa_bat']

reg['raa_bowl'] = reg['exp_bowl_runs'] - reg['bowler_runs']
reg['waa_bowl'] = reg['wicket_value'] * (reg['is_bowler_wkt'].astype(float) - reg['exp_bowl_wkt'])
reg['tva_bowl'] = reg['raa_bowl'] + reg['waa_bowl']

# 1. Check Empirical Bayes Variance Components (sigma_ball^2 vs tau_player^2) -> optimal k!
tf = reg[reg['match_tier'].isin(['IPL', 'T20I_FM_vs_FM'])].copy()
q75_bat = tf.loc[tf['is_batter_ball'], 'PI_bat'].quantile(0.75)
q75_bowl = tf['PI_bowl'].quantile(0.75)

bat_hp = tf[tf['is_batter_ball'] & (tf['PI_bat'] >= q75_bat)]
g_b = bat_hp.groupby('striker_id')['tva_bat'].agg(['count','mean','var'])
g_b_big = g_b[g_b['count'] >= 150]
sigma2_b = bat_hp['tva_bat'].var() # per-ball variance (in runs^2, multiply by 100^2 if scaled per 100 balls)
# Observe Var(mean_i) = tau^2 + E[sigma^2 / n_i]
tau2_b = max(1e-4, g_b_big['mean'].var() - (sigma2_b / g_b_big['count']).mean())
k_bat = sigma2_b / tau2_b
print(f'Empirical Bayes Bat HP: sigma_ball={np.sqrt(sigma2_b):.3f}, tau_player={np.sqrt(tau2_b):.4f} (per 100 balls: {100*np.sqrt(tau2_b):.2f}), optimal k_bat={k_bat:.1f} balls')

bowl_hp = tf[tf['PI_bowl'] >= q75_bowl]
g_bw = bowl_hp.groupby('bowler_id')['tva_bowl'].agg(['count','mean','var'])
g_bw_big = g_bw[g_bw['count'] >= 150]
sigma2_bw = bowl_hp['tva_bowl'].var()
tau2_bw = max(1e-4, g_bw_big['mean'].var() - (sigma2_bw / g_bw_big['count']).mean())
k_bowl = sigma2_bw / tau2_bw
print(f'Empirical Bayes Bowl HP: sigma_ball={np.sqrt(sigma2_bw):.3f}, tau_player={np.sqrt(tau2_bw):.4f} (per 100 balls: {100*np.sqrt(tau2_bw):.2f}), optimal k_bowl={k_bowl:.1f} balls')

# 2. Check Famous / Key Batters & Bowlers sample sizes & stats!
famous_bat = ['Virat Kohli','Jos Buttler','Suryakumar Yadav','Hardik Pandya','Heinrich Klaasen',
              'Nicholas Pooran','Rinku Singh','Sanju Samson','Lokesh Rahul','Yashasvi Jaiswal',
              'Namboori Tilak Varma','Shreyas Iyer','Tim David','David Miller','Dhruv Jurel',
              'Riyan Parag','Vaibhav Sooryavanshi','Shivam Dube','Rishabh Pant','Abhishek Sharma']
b_tf = tf[tf['is_batter_ball']].copy()
b_tf['is_hp'] = b_tf['PI_bat'] >= q75_bat
print('\n=== Key Batters HP vs LP Breakdown ===')
rows = []
for name in famous_bat:
    sub = b_tf[b_tf['striker_name']==name]
    if len(sub)==0: continue
    hp = sub[sub['is_hp']]; lp = sub[~sub['is_hp']]
    rows.append({
        'name': name, 'tot_b': len(sub), 'hp_b': len(hp), 'hp_m': hp['match_id'].nunique(),
        'hp_pct': 100*len(hp)/len(sub),
        'hp_sr': 100*hp['batter_runs'].mean() if len(hp) else 0,
        'lp_sr': 100*lp['batter_runs'].mean() if len(lp) else 0,
        'hp_tva100': 100*hp['tva_bat'].mean() if len(hp) else 0,
        'lp_tva100': 100*lp['tva_bat'].mean() if len(lp) else 0,
        'hp_wpa100': 100*hp['wpa_bat'].mean() if len(hp) else 0,
        'hp_balls_faced_before_mean': hp['batter_balls_before'].mean() if len(hp) else 0,
    })
print(pd.DataFrame(rows).round(2).to_string(index=False))

famous_bowl = ['Jasprit Bumrah','Arshdeep Singh','Rashid Khan','Sunil Narine','Kuldeep Yadav',
               'Axar Patel','Krunal Pandya','Varun Chakravarthy','Trent Boult','Bhuvneshwar Kumar',
               'Nathan Ellis','Patrick Cummins','Josh Hazlewood','Kagiso Rabada','Mitchell Starc',
               'Jofra Archer','Ravi Bishnoi','Yuzvendra Chahal','Matheesha Pathirana','Harshal Patel','Wanindu Hasaranga','Umran Malik']
bw_tf = tf.copy()
bw_tf['is_hp'] = bw_tf['PI_bowl'] >= q75_bowl
print('\n=== Key Bowlers HP vs LP Breakdown ===')
rows_bw = []
for name in famous_bowl:
    sub = bw_tf[bw_tf['bowler_name']==name]
    if len(sub)==0: continue
    hp = sub[sub['is_hp']]; lp = sub[~sub['is_hp']]
    rows_bw.append({
        'name': name, 'tot_leg': int(sub['is_legal'].sum()), 'hp_leg': int(hp['is_legal'].sum()),
        'hp_m': hp['match_id'].nunique(),
        'hp_pct': 100*hp['is_legal'].sum()/max(1,sub['is_legal'].sum()),
        'hp_econ': 6*hp['bowler_runs'].sum()/max(1,hp['is_legal'].sum()),
        'lp_econ': 6*lp['bowler_runs'].sum()/max(1,lp['is_legal'].sum()),
        'hp_wkts': int(hp['is_bowler_wkt'].sum()),
        'hp_tva100': 100*hp['tva_bowl'].mean() if len(hp) else 0,
        'lp_tva100': 100*lp['tva_bowl'].mean() if len(lp) else 0,
        'hp_wpa100': 100*hp['wpa_bowl'].mean() if len(hp) else 0,
    })
print(pd.DataFrame(rows_bw).round(2).to_string(index=False))

# 3. Check Team Strength differences across T20I Full Members & IPL!
print('\n=== Batting TVA & Bowling TVA by Opposition Team in Top-Flight ===')
# Wait: check how much team strength differs (e.g., Zimbabwe/Ireland vs India/Australia/IPL)
team_names = pd.read_csv(r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\data_pack\Data Pack\teams.csv')[['team_id','team_name']].drop_duplicates()
b_opp = b_tf.merge(team_names, left_on='bowling_team_id', right_on='team_id', how='left')
print('Mean Batting RAA per 100 balls vs Bowling Team (T20I_FM_vs_FM only):')
print(b_opp[b_opp['match_tier']=='T20I_FM_vs_FM'].groupby('team_name').agg(
    balls=('is_batter_ball','sum'),
    raa_100=('raa_bat', lambda x: 100*x.mean()),
    tva_100=('tva_bat', lambda x: 100*x.mean())
).sort_values('tva_100', ascending=False).round(2).to_string())
