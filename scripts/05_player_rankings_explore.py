import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\outputs'
reg = pd.read_pickle(f'{OUT}/enriched_balls.pkl')

# 1. Dynamic Wicket Run-Value V_W(s) using phase-specific empirical resource drop!
# Estimate empirical E_rem(balls_rem, wkts_rem) directly from full I1 matches -> smooth 2D grid
i1_full = reg[(reg['innings']==1)&(reg['max_balls']==120)&(reg['has_valid_outcome'])].copy()
i1_full['runs_to_go'] = i1_full['inn1_total'] - i1_full['cum_runs_before']
# Simple smooth regression or grouped curve by (over_bucket, wkts_rem)
# Actually we already have reg['exp_rem_runs'] from monotonic HGBR! Let's verify average runs-value of a wicket by over & wkts_rem:
# Fit a quick smooth 2D polynomial/spline or check phase-level V_W:
#Wait: let's compute E[runs_to_go | balls_rem, wkts_rem] via smoothed group means or simple monotonic fit:
from sklearn.ensemble import HistGradientBoostingRegressor
res_df = i1_full[['balls_rem','wickets_rem','env_par_rpo','runs_to_go']]
mdl_rem = HistGradientBoostingRegressor(max_iter=80, max_depth=4, min_samples_leaf=500, l2_regularization=20.0, monotonic_cst=[1,1,1], random_state=42)
mdl_rem.fit(res_df[['balls_rem','wickets_rem','env_par_rpo']], res_df['runs_to_go'])

# V_W(s) = E_rem(b, w, rpo) - E_rem(b, w-1, rpo)
df_minus = reg[['balls_rem','wickets_rem','env_par_rpo']].copy()
df_minus['wickets_rem'] = (df_minus['wickets_rem'] - 1).clip(lower=0)
rem_w_minus = np.where(reg['wickets_rem']==1, 0.0, mdl_rem.predict(df_minus))
rem_w_minus = np.maximum(0.0, rem_w_minus)
reg['wicket_value'] = np.clip(reg['exp_rem_runs'] - rem_w_minus, 2.0, 28.0)
# Wait: check V_W by phase!
print('=== Empirical Dynamic Wicket Value V_W by Phase ===')
print(reg.groupby('phase')['wicket_value'].describe().round(2).to_string())

# 2. Calibrate PI_bat and PI_bowl cleanly so BOTH I1 crisis/death AND I2 chase/defense register fairly!
reg['u_live'] = 4.0 * reg['wp_before'] * (1.0 - reg['wp_before'])
live_gate = np.where(
    reg['innings'] == 2,
    np.clip((np.minimum(reg['wp_before'], 1.0 - reg['wp_before']) - 0.03) / 0.14, 0.0, 1.0),
    np.clip((np.minimum(reg['wp_before'], 1.0 - reg['wp_before']) - 0.05) / 0.15, 0.35, 1.0)
)
# Notice in I1:
# - Early/Mid Collapse: 2+ down in PP, 3+ down by ov 10, 4+ down by ov 15 -> collapse_excess = cum_wkts_before - 0.042*legal_balls_before
exp_wkts = 0.042 * reg['legal_balls_before']
collapse_excess = np.maximum(0.0, reg['cum_wkts_before'] - exp_wkts)
collapse_stress = np.where(reg['cum_wkts_before'] >= 2, np.tanh(collapse_excess / 1.5), 0.0) * live_gate
bat_shock = np.tanh(np.maximum(0.0, reg['wkts_last18'] - 1.0) / 1.2) * live_gate

# - I2 RRR tension:
reg['rrr'] = np.where(reg['innings']==2, 6.0 * np.maximum(0, reg['target'] - reg['cum_runs_before']) / reg['balls_rem'], 0.0)
rrr_excess = np.maximum(0.0, reg['rrr'] - reg['env_par_rpo'])
rrr_stress = np.where(reg['innings']==2, np.tanh(rrr_excess / 4.0) * live_gate, 0.0)

# - Death urgency (ov 16-20):
death_ramp = np.where(reg['overid'] >= 16, 0.55 + 0.45 * (reg['overid'] - 16) / 4.0, 0.0) * live_gate

# - WP Leverage normalized cleanly across innings (using calibrated scale so I1 live death/collapse reaches ~55-75 and I2 crunch reaches ~65-100)
lev_score = np.where(
    reg['innings'] == 2,
    np.tanh(reg['wp_lev'] / 0.048),
    0.55 * np.tanh(reg['wp_lev'] / 0.025) * live_gate
)

bowl_defense_risk = np.where(
    reg['innings'] == 2,
    np.clip((reg['wp_before'] - 0.12) / 0.50, 0.0, 1.0) * live_gate,
    0.22 * live_gate
)
bowl_surge = np.tanh(np.maximum(0.0, reg['runs_last18'] - 26.0) / 15.0) * live_gate
ko_boost = np.where(reg['is_knockout'] == 1, 0.12, 0.0) * np.maximum(0.35, live_gate)

reg['PI_bat'] = 100.0 * np.clip(
    0.38 * lev_score + 0.34 * np.maximum(collapse_stress, 0.6 * bat_shock) + 0.24 * rrr_stress + 0.24 * death_ramp + ko_boost,
    0.0, 1.0
)
reg['PI_bowl'] = 100.0 * np.clip(
    0.40 * lev_score + 0.32 * death_ramp + 0.20 * bowl_defense_risk + 0.18 * bowl_surge + ko_boost,
    0.0, 1.0
)

print('\n=== Calibrated PI_bat & PI_bowl Situation Check ===')
situations = {
    '1. I2 Death Close (ov17-20, 0.2<=WP<=0.8)': (reg['innings']==2)&(reg['overid']>=17)&(reg['wp_before'].between(0.2,0.8)),
    '2. I2 Dead Chase (ov17-20, WP<0.03)': (reg['innings']==2)&(reg['overid']>=17)&(reg['wp_before']<0.03),
    '3. I2 Cruise Chase (ov17-20, WP>0.95)': (reg['innings']==2)&(reg['overid']>=17)&(reg['wp_before']>0.95),
    '4. Early Collapse (ov1-8, wkts>=3, WP>=0.08)': (reg['overid']<=8)&(reg['cum_wkts_before']>=3)&(reg['wp_before']>=0.08),
    '5. Mid Collapse (ov7-14, wkts>=4, WP>=0.08)': (reg['overid'].between(7,14))&(reg['cum_wkts_before']>=4)&(reg['wp_before']>=0.08),
    '6. Comfortable PP (ov1-6, wkts<=1, I1)': (reg['innings']==1)&(reg['overid']<=6)&(reg['cum_wkts_before']<=1),
    '7. Routine Middle (ov7-15, wkts<=2, I1)': (reg['innings']==1)&(reg['overid'].between(7,15))&(reg['cum_wkts_before']<=2),
    '8. I1 Death Competitive (ov17-20, 0.25<=WP<=0.75)': (reg['innings']==1)&(reg['overid']>=17)&(reg['wp_before'].between(0.25,0.75)),
    '9. Knockout Playoffs (all balls)': (reg['is_knockout']==1),
}
for label, mask in situations.items():
    sub = reg[mask]
    print(f'{label:48s} | n={len(sub):6d} | PI_bat={sub["PI_bat"].mean():4.1f} | PI_bowl={sub["PI_bowl"].mean():4.1f}')

print('\n=== Overall Quantiles (All Regular Balls) ===')
print('PI_bat :', reg['PI_bat'].quantile([0.25, 0.5, 0.70, 0.75, 0.80, 0.90]).round(1).to_dict())
print('PI_bowl:', reg['PI_bowl'].quantile([0.25, 0.5, 0.70, 0.75, 0.80, 0.90]).round(1).to_dict())

# Compute ball-level Total Value Added (TVA)
reg['raa_bat'] = reg['batter_runs'] - reg['exp_bat_runs']
reg['waa_bat'] = reg['wicket_value'] * (reg['exp_bat_out'] - reg['is_striker_out'].astype(float))
reg['tva_bat'] = reg['raa_bat'] + reg['waa_bat']

reg['raa_bowl'] = reg['exp_bowl_runs'] - reg['bowler_runs']
reg['waa_bowl'] = reg['wicket_value'] * (reg['is_bowler_wkt'].astype(float) - reg['exp_bowl_wkt'])
reg['tva_bowl'] = reg['raa_bowl'] + reg['waa_bowl']

# 3. Evaluate Player Universe across Top-Flight T20 (IPL + T20I_FM_vs_FM) vs All Matches
top_flight = reg[reg['match_tier'].isin(['IPL', 'T20I_FM_vs_FM'])].copy()
bat_tf = top_flight[top_flight['is_batter_ball']].copy()
bowl_tf = top_flight.copy()

# Check distribution of High Pressure share (using Q75 of PI_bat ~ 43-45, let's check exact Q75!)
q75_bat = bat_tf['PI_bat'].quantile(0.75)
q75_bowl = bowl_tf['PI_bowl'].quantile(0.75)
print(f'\nTop-flight Q75 thresholds: PI_bat={q75_bat:.1f}, PI_bowl={q75_bowl:.1f}')

# Compute batting summary at PI_bat >= q75_bat (HP) vs PI_bat < q75_bat & PI_bat >= 5 (Normal competitive, or all < q75_bat)
#Wait: check if excluding garbage time (<5) from "Usual" changes Normal baseline cleanly!
# Actually, all PI < q75_bat vs live-normal (10 <= PI < q75_bat): let's check both, or keep all < q75_bat with valid outcome!
# Actually keeping PI < q75_bat (all non-HP) is standard and transparent; let's inspect!
bat_tf['is_hp'] = bat_tf['PI_bat'] >= q75_bat
bowl_tf['is_hp'] = bowl_tf['PI_bowl'] >= q75_bowl

print('HP share in I1 vs I2 (Batting):', bat_tf.groupby('innings')['is_hp'].mean().round(3).to_dict())
print('HP share in I1 vs I2 (Bowling):', bowl_tf.groupby('innings')['is_hp'].mean().round(3).to_dict())

def eval_batters(df, min_tot_balls=350, min_hp_balls=100, min_hp_matches=10):
    hp = df[df['is_hp']]
    lp = df[~df['is_hp']]
    
    # Per-match HP stats for consistency
    m_hp = hp.groupby(['striker_id','striker_name','match_id']).agg(
        m_balls=('is_batter_ball','sum'),
        m_tva=('tva_bat','sum'),
        m_wpa=('wpa_bat','sum')
    ).reset_index()
    
    agg_tot = df.groupby(['striker_id','striker_name']).agg(
        tot_balls=('is_batter_ball','sum'),
        tot_runs=('batter_runs','sum'),
        tot_outs=('is_striker_out','sum'),
        tot_tva_100=('tva_bat', lambda x: 100.0 * x.mean()),
        ipl_balls=('match_tier', lambda x: (x=='IPL').sum()),
        t20i_balls=('match_tier', lambda x: (x=='T20I_FM_vs_FM').sum()),
    ).reset_index()
    
    agg_hp = hp.groupby('striker_id').agg(
        hp_balls=('is_batter_ball','sum'),
        hp_matches=('match_id','nunique'),
        hp_runs=('batter_runs','sum'),
        hp_outs=('is_striker_out','sum'),
        hp_raa_100=('raa_bat', lambda x: 100.0 * x.mean()),
        hp_waa_100=('waa_bat', lambda x: 100.0 * x.mean()),
        hp_tva_100=('tva_bat', lambda x: 100.0 * x.mean()),
        hp_wpa_100=('wpa_bat', lambda x: 100.0 * x.mean()),
        hp_sr=('batter_runs', lambda x: 100.0 * x.mean()),
        hp_avg=('batter_runs', lambda x: x.sum() / max(1, hp.loc[x.index, 'is_striker_out'].sum())),
        hp_bdry=('batter_runs', lambda x: 100.0 * x.isin([4,6]).mean()),
        hp_dot=('batter_runs', lambda x: 100.0 * (x==0).mean()),
    ).reset_index()
    
    agg_lp = lp.groupby('striker_id').agg(
        lp_balls=('is_batter_ball','sum'),
        lp_runs=('batter_runs','sum'),
        lp_outs=('is_striker_out','sum'),
        lp_tva_100=('tva_bat', lambda x: 100.0 * x.mean()),
        lp_sr=('batter_runs', lambda x: 100.0 * x.mean()),
    ).reset_index()
    
    # Match consistency stats
    m_cons = m_hp.groupby('striker_id').agg(
        pos_match_rate=('m_tva', lambda x: (x > 0).mean()),
        med_match_tva=('m_tva', 'median'),
        match_tva_mean=('m_tva', 'mean'),
        match_tva_std=('m_tva', 'std'),
    ).reset_index()
    
    res = agg_tot.merge(agg_hp, on='striker_id').merge(agg_lp, on='striker_id').merge(m_cons, on='striker_id')
    res = res[(res['tot_balls'] >= min_tot_balls) & (res['hp_balls'] >= min_hp_balls) & (res['hp_matches'] >= min_hp_matches)].copy()
    
    # Empirical Bayes Shrinkage (toward 0 for context-adjusted TVA)
    # Using prior k_hp = 150 balls, k_lp = 200 balls (we'll estimate empirical k too!)
    res['hp_tva_shrunk'] = (res['hp_balls'] / (res['hp_balls'] + 150.0)) * res['hp_tva_100']
    res['lp_tva_shrunk'] = (res['lp_balls'] / (res['lp_balls'] + 200.0)) * res['lp_tva_100']
    res['clutch_delta'] = res['hp_tva_shrunk'] - res['lp_tva_shrunk']
    res['hp_wpa_shrunk'] = (res['hp_balls'] / (res['hp_balls'] + 150.0)) * res['hp_wpa_100']
    
    # Match t-stat / LCB of match TVA
    res['match_se'] = res['match_tva_std'].fillna(10.0) / np.sqrt(res['hp_matches'])
    res['match_lcb'] = res['match_tva_mean'] - 1.0 * res['match_se'] # 1-SE lower confidence bound!
    
    # Z-scores for composite PRS
    def z(s): return (s - s.mean()) / (s.std() + 1e-9)
    res['z_hp_tva'] = z(res['hp_tva_shrunk'])
    res['z_hp_wpa'] = z(res['hp_wpa_shrunk'])
    res['z_delta']  = z(res['clutch_delta'])
    res['z_cons']   = 0.5 * z(res['pos_match_rate']) + 0.5 * z(res['match_lcb'])
    
    # Composite Pressure Reliability Score (0-100 rescaled or z-composite)
    res['PRS_raw'] = 0.45 * res['z_hp_tva'] + 0.20 * res['z_hp_wpa'] + 0.20 * res['z_delta'] + 0.15 * res['z_cons']
    res['PRS_pure_hp'] = 0.65 * res['z_hp_tva'] + 0.20 * res['z_hp_wpa'] + 0.15 * res['z_cons']
    return res.sort_values('PRS_raw', ascending=False)

def eval_bowlers(df, min_tot_balls=350, min_hp_balls=100, min_hp_matches=10):
    hp = df[df['is_hp']]
    lp = df[~df['is_hp']]
    
    m_hp = hp.groupby(['bowler_id','bowler_name','match_id']).agg(
        m_balls=('is_legal','sum'),
        m_tva=('tva_bowl','sum'),
        m_wpa=('wpa_bowl','sum')
    ).reset_index()
    
    agg_tot = df.groupby(['bowler_id','bowler_name']).agg(
        tot_legal=('is_legal','sum'),
        tot_runs=('bowler_runs','sum'),
        tot_wkts=('is_bowler_wkt','sum'),
        tot_tva_100=('tva_bowl', lambda x: 100.0 * x.mean()),
        ipl_balls=('match_tier', lambda x: (x=='IPL').sum()),
        t20i_balls=('match_tier', lambda x: (x=='T20I_FM_vs_FM').sum()),
    ).reset_index()
    
    agg_hp = hp.groupby('bowler_id').agg(
        hp_deliveries=('bowler_runs','size'),
        hp_legal=('is_legal','sum'),
        hp_matches=('match_id','nunique'),
        hp_runs=('bowler_runs','sum'),
        hp_wkts=('is_bowler_wkt','sum'),
        hp_raa_100=('raa_bowl', lambda x: 100.0 * x.mean()),
        hp_waa_100=('waa_bowl', lambda x: 100.0 * x.mean()),
        hp_tva_100=('tva_bowl', lambda x: 100.0 * x.mean()),
        hp_wpa_100=('wpa_bowl', lambda x: 100.0 * x.mean()),
        hp_dot=('bowler_runs', lambda x: 100.0 * (x==0).mean()),
        hp_bdry=('runs', lambda x: 100.0 * x.isin([4,6]).mean()),
    ).reset_index()
    agg_hp['hp_econ'] = 6.0 * agg_hp['hp_runs'] / agg_hp['hp_legal'].clip(lower=1)
    
    agg_lp = lp.groupby('bowler_id').agg(
        lp_deliveries=('bowler_runs','size'),
        lp_legal=('is_legal','sum'),
        lp_runs=('bowler_runs','sum'),
        lp_wkts=('is_bowler_wkt','sum'),
        lp_tva_100=('tva_bowl', lambda x: 100.0 * x.mean()),
    ).reset_index()
    agg_lp['lp_econ'] = 6.0 * agg_lp['lp_runs'] / agg_lp['lp_legal'].clip(lower=1)
    
    m_cons = m_hp.groupby('bowler_id').agg(
        pos_match_rate=('m_tva', lambda x: (x > 0).mean()),
        med_match_tva=('m_tva', 'median'),
        match_tva_mean=('m_tva', 'mean'),
        match_tva_std=('m_tva', 'std'),
    ).reset_index()
    
    res = agg_tot.merge(agg_hp, on='bowler_id').merge(agg_lp, on='bowler_id').merge(m_cons, on='bowler_id')
    res = res[(res['tot_legal'] >= min_tot_balls) & (res['hp_legal'] >= min_hp_balls) & (res['hp_matches'] >= min_hp_matches)].copy()
    
    res['hp_tva_shrunk'] = (res['hp_deliveries'] / (res['hp_deliveries'] + 150.0)) * res['hp_tva_100']
    res['lp_tva_shrunk'] = (res['lp_deliveries'] / (res['lp_deliveries'] + 200.0)) * res['lp_tva_100']
    res['clutch_delta'] = res['hp_tva_shrunk'] - res['lp_tva_shrunk']
    res['hp_wpa_shrunk'] = (res['hp_deliveries'] / (res['hp_deliveries'] + 150.0)) * res['hp_wpa_100']
    
    res['match_se'] = res['match_tva_std'].fillna(10.0) / np.sqrt(res['hp_matches'])
    res['match_lcb'] = res['match_tva_mean'] - 1.0 * res['match_se']
    
    def z(s): return (s - s.mean()) / (s.std() + 1e-9)
    res['z_hp_tva'] = z(res['hp_tva_shrunk'])
    res['z_hp_wpa'] = z(res['hp_wpa_shrunk'])
    res['z_delta']  = z(res['clutch_delta'])
    res['z_cons']   = 0.5 * z(res['pos_match_rate']) + 0.5 * z(res['match_lcb'])
    
    res['PRS_raw'] = 0.45 * res['z_hp_tva'] + 0.20 * res['z_hp_wpa'] + 0.20 * res['z_delta'] + 0.15 * res['z_cons']
    res['PRS_pure_hp'] = 0.65 * res['z_hp_tva'] + 0.20 * res['z_hp_wpa'] + 0.15 * res['z_cons']
    return res.sort_values('PRS_raw', ascending=False)

bat_res = eval_batters(bat_tf, min_tot_balls=350, min_hp_balls=100, min_hp_matches=10)
bowl_res = eval_bowlers(bowl_tf, min_tot_balls=350, min_hp_balls=100, min_hp_matches=10)

print(f'\nEligible Batters: {len(bat_res)}, Eligible Bowlers: {len(bowl_res)}')
print('\n=== TOP 20 BATTERS (PRS_raw) ===')
cols_b = ['striker_name','tot_balls','hp_balls','hp_matches','hp_sr','lp_sr','hp_tva_shrunk','lp_tva_shrunk','clutch_delta','hp_wpa_shrunk','pos_match_rate','PRS_raw','PRS_pure_hp']
print(bat_res[cols_b].head(20).round(2).to_string(index=False))

print('\n=== TOP 20 BOWLERS (PRS_raw) ===')
cols_bw = ['bowler_name','tot_legal','hp_legal','hp_matches','hp_econ','lp_econ','hp_wkts','hp_tva_shrunk','lp_tva_shrunk','clutch_delta','hp_wpa_shrunk','pos_match_rate','PRS_raw','PRS_pure_hp']
print(bowl_res[cols_bw].head(20).round(2).to_string(index=False))

print('\n=== Rank correlation PRS_raw vs PRS_pure_hp ===')
print('Batters Spearman r:', round(spearmanr(bat_res['PRS_raw'], bat_res['PRS_pure_hp']).statistic, 3))
print('Bowlers Spearman r:', round(spearmanr(bowl_res['PRS_raw'], bowl_res['PRS_pure_hp']).statistic, 3))

# Also check Bottom 10 among high-volume players (who dips most under pressure?!)
print('\n=== High-volume Batters (>=600 balls) with biggest NEGATIVE clutch delta (chokers/dippers) ===')
print(bat_res[bat_res['tot_balls']>=600].sort_values('clutch_delta').head(10)[['striker_name','tot_balls','hp_balls','hp_tva_shrunk','lp_tva_shrunk','clutch_delta','PRS_raw']].round(2).to_string(index=False))

print('\n=== High-volume Bowlers (>=600 balls) with biggest NEGATIVE clutch delta ===')
print(bowl_res[bowl_res['tot_legal']>=600].sort_values('clutch_delta').head(10)[['bowler_name','tot_legal','hp_legal','hp_tva_shrunk','lp_tva_shrunk','clutch_delta','PRS_raw']].round(2).to_string(index=False))
