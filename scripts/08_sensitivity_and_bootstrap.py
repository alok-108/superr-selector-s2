import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

OUT = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\outputs'
reg = pd.read_pickle(f'{OUT}/enriched_balls.pkl')

# 1. Re-compute dynamic wicket_value & PI_bat, PI_bowl
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

# Out-of-fold opposition team strength adjustment
gkf = GroupKFold(n_splits=5)
reg['opp_bowl_adj'] = 0.0
reg['opp_bat_adj'] = 0.0
for tr_i, va_i in gkf.split(reg, groups=reg['match_id']):
    tr = reg.iloc[tr_i]; va = reg.iloc[va_i]
    opp_bw = tr[tr['is_batter_ball']].groupby('bowling_team_id')['tva_bat'].agg(['sum','count'])
    reg.iloc[va_i, reg.columns.get_loc('opp_bowl_adj')] = va['bowling_team_id'].map(opp_bw['sum'] / (opp_bw['count'] + 2000.0)).fillna(0.0).values
    opp_bt = tr.groupby('batting_team_id')['tva_bowl'].agg(['sum','count'])
    reg.iloc[va_i, reg.columns.get_loc('opp_bat_adj')] = va['batting_team_id'].map(opp_bt['sum'] / (opp_bt['count'] + 2000.0)).fillna(0.0).values

reg['tva_bat_opp'] = reg['tva_bat'] - reg['opp_bowl_adj']
reg['tva_bowl_opp'] = reg['tva_bowl'] - reg['opp_bat_adj']

tf = reg[reg['match_tier'].isin(['IPL', 'T20I_FM_vs_FM'])].copy()

def compute_bat_rankings(df, q_thresh=0.75, k_hp=500.0, k_lp=500.0, min_tot=400, min_hp=180, min_hp_m=15):
    b = df[df['is_batter_ball']].copy()
    cut = b['PI_bat'].quantile(q_thresh)
    b['is_hp'] = b['PI_bat'] >= cut
    hp = b[b['is_hp']]; lp = b[~b['is_hp']]
    
    m_hp = hp.groupby(['striker_id','match_id'])['tva_bat_opp'].sum().reset_index()
    tot = b.groupby(['striker_id','striker_name']).agg(
        tot_b=('is_batter_ball','sum'), tot_runs=('batter_runs','sum'),
        ipl_b=('match_tier', lambda x: (x=='IPL').sum()),
        t20i_b=('match_tier', lambda x: (x=='T20I_FM_vs_FM').sum()),
        cont_w_tva100=('tva_bat_opp', lambda x: 100.0 * np.average(x, weights=(b.loc[x.index,'PI_bat']/100.0)**2 + 1e-6)),
    ).reset_index()
    
    ahp = hp.groupby('striker_id').agg(
        hp_b=('is_batter_ball','sum'), hp_m=('match_id','nunique'),
        hp_runs=('batter_runs','sum'), hp_outs=('is_striker_out','sum'),
        hp_sr=('batter_runs', lambda x: 100*x.mean()),
        hp_raa100=('raa_bat', lambda x: 100*x.mean()),
        hp_waa100=('waa_bat', lambda x: 100*x.mean()),
        hp_tva100=('tva_bat_opp', lambda x: 100*x.mean()),
        hp_wpa100=('wpa_bat', lambda x: 100*x.mean()),
        hp_bdry=('batter_runs', lambda x: 100*x.isin([4,6]).mean()),
        hp_dot=('batter_runs', lambda x: 100*(x==0).mean()),
    ).reset_index()
    
    alp = lp.groupby('striker_id').agg(
        lp_b=('is_batter_ball','sum'), lp_sr=('batter_runs', lambda x: 100*x.mean()),
        lp_tva100=('tva_bat_opp', lambda x: 100*x.mean()),
    ).reset_index()
    
    mc = m_hp.groupby('striker_id')['tva_bat_opp'].agg(
        pos_m_rate=lambda x: (x > 0).mean(), m_mean='mean', m_std='std', m_med='median'
    ).reset_index()
    
    r = tot.merge(ahp, on='striker_id').merge(alp, on='striker_id').merge(mc, on='striker_id')
    r = r[(r['tot_b']>=min_tot) & (r['hp_b']>=min_hp) & (r['hp_m']>=min_hp_m)].copy()
    
    r['hp_tva_eb'] = (r['hp_b'] / (r['hp_b'] + k_hp)) * r['hp_tva100']
    r['lp_tva_eb'] = (r['lp_b'] / (r['lp_b'] + k_lp)) * r['lp_tva100']
    r['hp_wpa_eb'] = (r['hp_b'] / (r['hp_b'] + k_hp)) * r['hp_wpa100']
    r['cont_tva_eb'] = (r['hp_b'] / (r['hp_b'] + k_hp)) * r['cont_w_tva100']
    
    # Regression-adjusted clutch residual: HP - E[HP | LP]
    lr = LinearRegression().fit(r[['lp_tva_eb']], r['hp_tva_eb'])
    r['exp_hp_from_lp'] = lr.predict(r[['lp_tva_eb']])
    r['clutch_resid'] = r['hp_tva_eb'] - r['exp_hp_from_lp']
    r['raw_delta'] = r['hp_tva_eb'] - r['lp_tva_eb']
    r['m_lcb'] = r['m_mean'] - r['m_std'].fillna(10.0) / np.sqrt(r['hp_m'])
    
    def z(s): return (s - s.mean()) / (s.std() + 1e-9)
    # 50% Absolute High-Pressure TVA (EB), 20% High-Pressure WPA (EB), 15% Regression-Adjusted Clutch Residual vs Usual Level, 15% Cross-Match Consistency
    r['PRS'] = 0.50*z(r['hp_tva_eb']) + 0.20*z(r['hp_wpa_eb']) + 0.15*z(r['clutch_resid']) + 0.15*(0.5*z(r['pos_m_rate']) + 0.5*z(r['m_lcb']))
    r = r.sort_values('PRS', ascending=False).reset_index(drop=True)
    r['rank'] = np.arange(1, len(r)+1)
    return r, cut, lr.coef_[0], lr.intercept_

def compute_bowl_rankings(df, q_thresh=0.75, k_hp=500.0, k_lp=500.0, min_tot=400, min_hp=180, min_hp_m=15):
    bw = df.copy()
    cut = bw['PI_bowl'].quantile(q_thresh)
    bw['is_hp'] = bw['PI_bowl'] >= cut
    hp = bw[bw['is_hp']]; lp = bw[~bw['is_hp']]
    
    m_hp = hp.groupby(['bowler_id','match_id'])['tva_bowl_opp'].sum().reset_index()
    tot = bw.groupby(['bowler_id','bowler_name']).agg(
        tot_leg=('is_legal','sum'), tot_wkts=('is_bowler_wkt','sum'),
        ipl_b=('match_tier', lambda x: (x=='IPL').sum()),
        t20i_b=('match_tier', lambda x: (x=='T20I_FM_vs_FM').sum()),
        cont_w_tva100=('tva_bowl_opp', lambda x: 100.0 * np.average(x, weights=(bw.loc[x.index,'PI_bowl']/100.0)**2 + 1e-6)),
    ).reset_index()
    
    ahp = hp.groupby('bowler_id').agg(
        hp_deliv=('bowler_runs','size'), hp_leg=('is_legal','sum'), hp_m=('match_id','nunique'),
        hp_runs=('bowler_runs','sum'), hp_wkts=('is_bowler_wkt','sum'),
        hp_raa100=('raa_bowl', lambda x: 100*x.mean()),
        hp_waa100=('waa_bowl', lambda x: 100*x.mean()),
        hp_tva100=('tva_bowl_opp', lambda x: 100*x.mean()),
        hp_wpa100=('wpa_bowl', lambda x: 100*x.mean()),
        hp_dot=('bowler_runs', lambda x: 100*(x==0).mean()),
    ).reset_index()
    ahp['hp_econ'] = 6.0 * ahp['hp_runs'] / ahp['hp_leg'].clip(lower=1)
    
    alp = lp.groupby('bowler_id').agg(
        lp_deliv=('bowler_runs','size'), lp_leg=('is_legal','sum'), lp_runs=('bowler_runs','sum'),
        lp_tva100=('tva_bowl_opp', lambda x: 100*x.mean()),
    ).reset_index()
    alp['lp_econ'] = 6.0 * alp['lp_runs'] / alp['lp_leg'].clip(lower=1)
    
    mc = m_hp.groupby('bowler_id')['tva_bowl_opp'].agg(
        pos_m_rate=lambda x: (x > 0).mean(), m_mean='mean', m_std='std', m_med='median'
    ).reset_index()
    
    r = tot.merge(ahp, on='bowler_id').merge(alp, on='bowler_id').merge(mc, on='bowler_id')
    r = r[(r['tot_leg']>=min_tot) & (r['hp_leg']>=min_hp) & (r['hp_m']>=min_hp_m)].copy()
    
    r['hp_tva_eb'] = (r['hp_deliv'] / (r['hp_deliv'] + k_hp)) * r['hp_tva100']
    r['lp_tva_eb'] = (r['lp_deliv'] / (r['lp_deliv'] + k_lp)) * r['lp_tva100']
    r['hp_wpa_eb'] = (r['hp_deliv'] / (r['hp_deliv'] + k_hp)) * r['hp_wpa100']
    r['cont_tva_eb'] = (r['hp_deliv'] / (r['hp_deliv'] + k_hp)) * r['cont_w_tva100']
    
    lr = LinearRegression().fit(r[['lp_tva_eb']], r['hp_tva_eb'])
    r['exp_hp_from_lp'] = lr.predict(r[['lp_tva_eb']])
    r['clutch_resid'] = r['hp_tva_eb'] - r['exp_hp_from_lp']
    r['raw_delta'] = r['hp_tva_eb'] - r['lp_tva_eb']
    r['m_lcb'] = r['m_mean'] - r['m_std'].fillna(10.0) / np.sqrt(r['hp_m'])
    
    def z(s): return (s - s.mean()) / (s.std() + 1e-9)
    r['PRS'] = 0.50*z(r['hp_tva_eb']) + 0.20*z(r['hp_wpa_eb']) + 0.15*z(r['clutch_resid']) + 0.15*(0.5*z(r['pos_m_rate']) + 0.5*z(r['m_lcb']))
    r = r.sort_values('PRS', ascending=False).reset_index(drop=True)
    r['rank'] = np.arange(1, len(r)+1)
    return r, cut, lr.coef_[0], lr.intercept_

rb75, cut_b, beta_b, int_b = compute_bat_rankings(tf, q_thresh=0.75, k_hp=500.0)
rbw75, cut_bw, beta_bw, int_bw = compute_bowl_rankings(tf, q_thresh=0.75, k_hp=500.0)

print(f'Bat HP ~ LP regression: slope={beta_b:.3f}, intercept={int_b:.3f} (cutoff PI_bat={cut_b:.1f})')
print(f'Bowl HP ~ LP regression: slope={beta_bw:.3f}, intercept={int_bw:.3f} (cutoff PI_bowl={cut_bw:.1f})')

print('\n=== PRIMARY TOP 15 BATTERS (Q75, K=500, Reg-Adjusted Clutch) ===')
print(rb75[['rank','striker_name','tot_b','hp_b','hp_m','hp_sr','lp_sr','hp_tva_eb','lp_tva_eb','clutch_resid','hp_wpa_eb','pos_m_rate','PRS']].head(15).round(2).to_string(index=False))

print('\n=== PRIMARY TOP 15 BOWLERS (Q75, K=500, Reg-Adjusted Clutch) ===')
print(rbw75[['rank','bowler_name','tot_leg','hp_leg','hp_m','hp_econ','lp_econ','hp_wkts','hp_tva_eb','lp_tva_eb','clutch_resid','hp_wpa_eb','pos_m_rate','PRS']].head(15).round(2).to_string(index=False))

# --- SENSITIVITY ANALYSIS ACROSS CUTOFFS (Q70, Q80) AND K (300, 800) AND CONTINUOUS WEIGHTS ---
rb70, _, _, _ = compute_bat_rankings(tf, q_thresh=0.70, k_hp=500.0)
rb80, _, _, _ = compute_bat_rankings(tf, q_thresh=0.80, k_hp=500.0, min_hp=140, min_hp_m=12)
rb300, _, _, _ = compute_bat_rankings(tf, q_thresh=0.75, k_hp=300.0)
rb800, _, _, _ = compute_bat_rankings(tf, q_thresh=0.75, k_hp=800.0)

rbw70, _, _, _ = compute_bowl_rankings(tf, q_thresh=0.70, k_hp=500.0)
rbw80, _, _, _ = compute_bowl_rankings(tf, q_thresh=0.80, k_hp=500.0, min_hp=140, min_hp_m=12)
rbw300, _, _, _ = compute_bowl_rankings(tf, q_thresh=0.75, k_hp=300.0)
rbw800, _, _, _ = compute_bowl_rankings(tf, q_thresh=0.75, k_hp=800.0)

print('\n=== SENSITIVITY COMPARISON: TOP 12 BATTERS RANKS ACROSS SPECS ===')
sens_b = rb75[['striker_id','striker_name','rank']].head(15).rename(columns={'rank':'Q75_K500'})
for lbl, df_s in [('Q70_K500', rb70), ('Q80_K500', rb80), ('Q75_K300', rb300), ('Q75_K800', rb800)]:
    sens_b = sens_b.merge(df_s[['striker_id','rank']].rename(columns={'rank':lbl}), on='striker_id', how='left')
print(sens_b.drop(columns=['striker_id']).to_string(index=False))

print('\n=== SENSITIVITY COMPARISON: TOP 12 BOWLERS RANKS ACROSS SPECS ===')
sens_bw = rbw75[['bowler_id','bowler_name','rank']].head(15).rename(columns={'rank':'Q75_K500'})
for lbl, df_s in [('Q70_K500', rbw70), ('Q80_K500', rbw80), ('Q75_K300', rbw300), ('Q75_K800', rbw800)]:
    sens_bw = sens_bw.merge(df_s[['bowler_id','rank']].rename(columns={'rank':lbl}), on='bowler_id', how='left')
print(sens_bw.drop(columns=['bowler_id']).to_string(index=False))

# Check rank correlations across entire eligible pool
for lbl, df_s in [('Q70', rb70), ('Q75_K300', rb300), ('Q75_K800', rb800)]:
    mrg = rb75[['striker_id','PRS']].merge(df_s[['striker_id','PRS']], on='striker_id')
    print(f'Batters Spearman (Q75_K500 vs {lbl}): {spearmanr(mrg["PRS_x"], mrg["PRS_y"]).statistic:.3f}')
for lbl, df_s in [('Q70', rbw70), ('Q75_K300', rbw300), ('Q75_K800', rbw800)]:
    mrg = rbw75[['bowler_id','PRS']].merge(df_s[['bowler_id','PRS']], on='bowler_id')
    print(f'Bowlers Spearman (Q75_K500 vs {lbl}): {spearmanr(mrg["PRS_x"], mrg["PRS_y"]).statistic:.3f}')

# --- MATCH-CLUSTERED BOOTSTRAP STABILITY (1000 resamples per Top-15 player) ---
rng = np.random.default_rng(42)
b_tf_hp = tf[tf['is_batter_ball'] & (tf['PI_bat'] >= cut_b)]
bw_tf_hp = tf[tf['PI_bowl'] >= cut_bw]

def bootstrap_ci(hp_df, id_col, val_col, top_ids):
    rows = []
    for pid in top_ids:
        sub = hp_df[hp_df[id_col] == pid]
        m_stats = sub.groupby('match_id')[val_col].agg(['sum','count']).values
        n_m = len(m_stats)
        boots = np.zeros(1000)
        for i in range(1000):
            idx = rng.integers(0, n_m, size=n_m)
            samp = m_stats[idx]
            boots[i] = 100.0 * samp[:, 0].sum() / max(1.0, samp[:, 1].sum())
        rows.append({
            id_col: pid,
            'boot_mean': boots.mean(),
            'ci_low_95': np.percentile(boots, 2.5),
            'ci_high_95': np.percentile(boots, 97.5),
            'p_gt_zero': (boots > 0).mean()
        })
    return pd.DataFrame(rows)

boot_b = bootstrap_ci(b_tf_hp, 'striker_id', 'tva_bat_opp', rb75.head(15)['striker_id'])
rb_boot = rb75.head(15)[['rank','striker_id','striker_name','hp_tva100']].merge(boot_b, on='striker_id').drop(columns=['striker_id'])
print('\n=== TOP 15 BATTERS: 1000-ITER MATCH-CLUSTERED BOOTSTRAP (95% CI on Raw HP TVA/100) ===')
print(rb_boot.round(2).to_string(index=False))

boot_bw = bootstrap_ci(bw_tf_hp, 'bowler_id', 'tva_bowl_opp', rbw75.head(15)['bowler_id'])
rbw_boot = rbw75.head(15)[['rank','bowler_id','bowler_name','hp_tva100']].merge(boot_bw, on='bowler_id').drop(columns=['bowler_id'])
print('\n=== TOP 15 BOWLERS: 1000-ITER MATCH-CLUSTERED BOOTSTRAP (95% CI on Raw HP TVA/100) ===')
print(rbw_boot.round(2).to_string(index=False))

rb75.to_csv(f'{OUT}/all_eligible_batters.csv', index=False)
rbw75.to_csv(f'{OUT}/all_eligible_bowlers.csv', index=False)
