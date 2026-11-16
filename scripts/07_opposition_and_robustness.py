import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

BASE = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\data_pack\Data Pack'
OUT = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\outputs'

reg = pd.read_pickle(f'{OUT}/enriched_balls.pkl')
teams = pd.read_csv(f'{BASE}/teams.csv')[['team_id','team_name']].drop_duplicates()

# Re-attach wicket_value & calibrated PI_bat, PI_bowl
from sklearn.ensemble import HistGradientBoostingRegressor
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

# Base TVA before opponent adjustment
reg['raa_bat'] = reg['batter_runs'] - reg['exp_bat_runs']
reg['waa_bat'] = reg['wicket_value'] * (reg['exp_bat_out'] - reg['is_striker_out'].astype(float))
reg['tva_bat'] = reg['raa_bat'] + reg['waa_bat']

reg['raa_bowl'] = reg['exp_bowl_runs'] - reg['bowler_runs']
reg['waa_bowl'] = reg['wicket_value'] * (reg['is_bowler_wkt'].astype(float) - reg['exp_bowl_wkt'])
reg['tva_bowl'] = reg['raa_bowl'] + reg['waa_bowl']

# --- OPPONENT TEAM STRENGTH ADJUSTMENT (Out-of-fold across Top-Flight + all matches!) ---
# Compute out-of-fold average residual TVA allowed by each bowling_team_id (when batting) and each batting_team_id (when bowling)!
# Use shrinkage toward 0 (prior = 3000 balls) per team, computed out-of-fold by match_id!
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(reg, groups=reg['match_id']))

reg['opp_bowl_adj'] = 0.0 # subtracted from tva_bat (if opponent bowling allows +0.08 TVA/ball, subtract 0.08!)
reg['opp_bat_adj'] = 0.0  # subtracted from tva_bowl (if opponent batting produces +0.08 TVA/ball against bowlers, add/adjust!)
# Wait: check sign!
# If Team X bowling allows +0.08 batting TVA/ball (weak bowling like WI/IRE), then batter facing Team X gets -0.08 opp_adj!
# If Team Y batting generates -0.08 bowling TVA/ball (strong batting like India/SRH), then bowler bowling to Team Y gets +0.08 opp_adj!

for tr_i, va_i in folds:
    tr = reg.iloc[tr_i]
    va = reg.iloc[va_i]
    
    # Opponent bowling team weakness (batting TVA allowed by bowling_team_id)
    tr_b = tr[tr['is_batter_ball']]
    opp_bw = tr_b.groupby('bowling_team_id')['tva_bat'].agg(['sum','count'])
    opp_bw_map = opp_bw['sum'] / (opp_bw['count'] + 2000.0)
    reg.iloc[va_i, reg.columns.get_loc('opp_bowl_adj')] = va['bowling_team_id'].map(opp_bw_map).fillna(0.0).values
    
    # Opponent batting team strength (bowling TVA conceded against batting_team_id: note positive = weak batting, negative = strong batting)
    opp_bt = tr.groupby('batting_team_id')['tva_bowl'].agg(['sum','count'])
    opp_bt_map = opp_bt['sum'] / (opp_bt['count'] + 2000.0)
    reg.iloc[va_i, reg.columns.get_loc('opp_bat_adj')] = va['batting_team_id'].map(opp_bt_map).fillna(0.0).values

reg['tva_bat_opp'] = reg['tva_bat'] - reg['opp_bowl_adj']
reg['tva_bowl_opp'] = reg['tva_bowl'] - reg['opp_bat_adj']

# Check opponent adjustment spread across teams in Top-Flight
tf = reg[reg['match_tier'].isin(['IPL', 'T20I_FM_vs_FM'])].copy()
print('Opp bowl adj range (per 100 balls) across teams:', (100*tf.groupby('bowling_team_id')['opp_bowl_adj'].mean()).round(2).agg(['min','max']).to_dict())
print('Opp bat adj range (per 100 balls) across teams:', (100*tf.groupby('batting_team_id')['opp_bat_adj'].mean()).round(2).agg(['min','max']).to_dict())

# --- EVALUATE RANKINGS WITH STRICT ELIGIBILITY & ROBUST CLUTCH RESILIENCE ---
q75_bat = tf.loc[tf['is_batter_ball'], 'PI_bat'].quantile(0.75)
q75_bowl = tf['PI_bowl'].quantile(0.75)
bat_tf = tf[tf['is_batter_ball']].copy()
bat_tf['is_hp'] = bat_tf['PI_bat'] >= q75_bat
bowl_tf = tf.copy()
bowl_tf['is_hp'] = bowl_tf['PI_bowl'] >= q75_bowl

# Compare Empirical Bayes Shrinkage k values & Eligibility thresholds!
def rank_batters(df, tva_col='tva_bat_opp', k_hp=400.0, k_lp=400.0, min_tot=400, min_hp=180, min_hp_m=15):
    hp = df[df['is_hp']]
    lp = df[~df['is_hp']]
    m_hp = hp.groupby(['striker_id','match_id'])[tva_col].sum().reset_index()
    
    tot = df.groupby(['striker_id','striker_name']).agg(
        tot_b=('is_batter_ball','sum'),
        tot_runs=('batter_runs','sum'),
        ipl_pct=('match_tier', lambda x: 100*(x=='IPL').mean()),
    ).reset_index()
    
    ahp = hp.groupby('striker_id').agg(
        hp_b=('is_batter_ball','sum'),
        hp_m=('match_id','nunique'),
        hp_runs=('batter_runs','sum'),
        hp_outs=('is_striker_out','sum'),
        hp_sr=('batter_runs', lambda x: 100*x.mean()),
        hp_tva100=(tva_col, lambda x: 100*x.mean()),
        hp_wpa100=('wpa_bat', lambda x: 100*x.mean()),
        hp_bdry=('batter_runs', lambda x: 100*x.isin([4,6]).mean()),
        hp_dot=('batter_runs', lambda x: 100*(x==0).mean()),
    ).reset_index()
    ahp['hp_avg'] = ahp['hp_runs'] / ahp['hp_outs'].clip(lower=1)
    
    alp = lp.groupby('striker_id').agg(
        lp_b=('is_batter_ball','sum'),
        lp_sr=('batter_runs', lambda x: 100*x.mean()),
        lp_tva100=(tva_col, lambda x: 100*x.mean()),
    ).reset_index()
    
    mc = m_hp.groupby('striker_id')[tva_col].agg(
        pos_m_rate=lambda x: (x > 0).mean(),
        m_mean='mean',
        m_std='std',
        m_med='median',
    ).reset_index()
    
    r = tot.merge(ahp, on='striker_id').merge(alp, on='striker_id').merge(mc, on='striker_id')
    r = r[(r['tot_b']>=min_tot) & (r['hp_b']>=min_hp) & (r['hp_m']>=min_hp_m)].copy()
    
    r['hp_tva_eb'] = (r['hp_b'] / (r['hp_b'] + k_hp)) * r['hp_tva100']
    r['lp_tva_eb'] = (r['lp_b'] / (r['lp_b'] + k_lp)) * r['lp_tva100']
    r['hp_wpa_eb'] = (r['hp_b'] / (r['hp_b'] + k_hp)) * r['hp_wpa100']
    
    # Resilience vs Usual Baseline (capped so negative LP doesn't give fake bonus!)
    # Effective baseline = clip(lp_tva_eb, 0, None) -> measures elevation above par OR own strong normal level!
    r['raw_delta'] = r['hp_tva_eb'] - r['lp_tva_eb']
    r['resil_delta'] = r['hp_tva_eb'] - np.maximum(0.0, r['lp_tva_eb'])
    r['m_lcb'] = r['m_mean'] - r['m_std'].fillna(10.0) / np.sqrt(r['hp_m'])
    
    def z(s): return (s - s.mean()) / (s.std() + 1e-9)
    # Composite Pressure Reliability Score:
    # 50% Absolute High-Pressure TVA (EB-shrunk)
    # 20% High-Pressure Win Probability Added (EB-shrunk)
    # 15% Resilience vs Own Baseline (clipped above 0, penalizes chokers while rewarding step-ups!)
    # 15% Match-to-Match Consistency (Positive HP Match Rate + Match LCB)
    r['PRS'] = (0.50*z(r['hp_tva_eb']) + 0.20*z(r['hp_wpa_eb']) +
                0.15*z(r['resil_delta']) + 0.15*(0.5*z(r['pos_m_rate']) + 0.5*z(r['m_lcb'])))
    # Also compute Continuous PI^2 weighted TVA across ALL balls for robustness check!
    return r.sort_values('PRS', ascending=False)

def rank_bowlers(df, tva_col='tva_bowl_opp', k_hp=400.0, k_lp=400.0, min_tot=400, min_hp=180, min_hp_m=15):
    hp = df[df['is_hp']]
    lp = df[~df['is_hp']]
    m_hp = hp.groupby(['bowler_id','match_id'])[tva_col].sum().reset_index()
    
    tot = df.groupby(['bowler_id','bowler_name']).agg(
        tot_leg=('is_legal','sum'),
        tot_wkts=('is_bowler_wkt','sum'),
        ipl_pct=('match_tier', lambda x: 100*(x=='IPL').mean()),
    ).reset_index()
    
    ahp = hp.groupby('bowler_id').agg(
        hp_deliv=('bowler_runs','size'),
        hp_leg=('is_legal','sum'),
        hp_m=('match_id','nunique'),
        hp_runs=('bowler_runs','sum'),
        hp_wkts=('is_bowler_wkt','sum'),
        hp_tva100=(tva_col, lambda x: 100*x.mean()),
        hp_wpa100=('wpa_bowl', lambda x: 100*x.mean()),
        hp_dot=('bowler_runs', lambda x: 100*(x==0).mean()),
    ).reset_index()
    ahp['hp_econ'] = 6.0 * ahp['hp_runs'] / ahp['hp_leg'].clip(lower=1)
    
    alp = lp.groupby('bowler_id').agg(
        lp_deliv=('bowler_runs','size'),
        lp_leg=('is_legal','sum'),
        lp_runs=('bowler_runs','sum'),
        lp_tva100=(tva_col, lambda x: 100*x.mean()),
    ).reset_index()
    alp['lp_econ'] = 6.0 * alp['lp_runs'] / alp['lp_leg'].clip(lower=1)
    
    mc = m_hp.groupby('bowler_id')[tva_col].agg(
        pos_m_rate=lambda x: (x > 0).mean(),
        m_mean='mean',
        m_std='std',
        m_med='median',
    ).reset_index()
    
    r = tot.merge(ahp, on='bowler_id').merge(alp, on='bowler_id').merge(mc, on='bowler_id')
    r = r[(r['tot_leg']>=min_tot) & (r['hp_leg']>=min_hp) & (r['hp_m']>=min_hp_m)].copy()
    
    r['hp_tva_eb'] = (r['hp_deliv'] / (r['hp_deliv'] + k_hp)) * r['hp_tva100']
    r['lp_tva_eb'] = (r['lp_deliv'] / (r['lp_deliv'] + k_lp)) * r['lp_tva100']
    r['hp_wpa_eb'] = (r['hp_deliv'] / (r['hp_deliv'] + k_hp)) * r['hp_wpa100']
    
    r['raw_delta'] = r['hp_tva_eb'] - r['lp_tva_eb']
    r['resil_delta'] = r['hp_tva_eb'] - np.maximum(0.0, r['lp_tva_eb'])
    r['m_lcb'] = r['m_mean'] - r['m_std'].fillna(10.0) / np.sqrt(r['hp_m'])
    
    def z(s): return (s - s.mean()) / (s.std() + 1e-9)
    r['PRS'] = (0.50*z(r['hp_tva_eb']) + 0.20*z(r['hp_wpa_eb']) +
                0.15*z(r['resil_delta']) + 0.15*(0.5*z(r['pos_m_rate']) + 0.5*z(r['m_lcb'])))
    return r.sort_values('PRS', ascending=False)

for k_val in [300.0, 600.0]:
    print(f'\n================ K_EB = {k_val}, min_hp=180, min_hp_m=15 (Opp-Adjusted) ================')
    rb = rank_batters(bat_tf, k_hp=k_val, k_lp=k_val)
    print(f'--- TOP 15 BATTERS (Eligible: {len(rb)}) ---')
    print(rb[['striker_name','tot_b','hp_b','hp_m','ipl_pct','hp_sr','lp_sr','hp_tva_eb','lp_tva_eb','resil_delta','hp_wpa_eb','pos_m_rate','PRS']].head(15).round(2).to_string(index=False))
    
    rbw = rank_bowlers(bowl_tf, k_hp=k_val, k_lp=k_val)
    print(f'--- TOP 15 BOWLERS (Eligible: {len(rbw)}) ---')
    print(rbw[['bowler_name','tot_leg','hp_leg','hp_m','ipl_pct','hp_econ','lp_econ','hp_wkts','hp_tva_eb','lp_tva_eb','resil_delta','hp_wpa_eb','pos_m_rate','PRS']].head(15).round(2).to_string(index=False))
