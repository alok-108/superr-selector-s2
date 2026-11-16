import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

BASE = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\data_pack\Data Pack'
OUT = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\outputs'

reg = pd.read_pickle(f'{OUT}/enriched_balls.pkl')
ctpm = pd.read_csv(f'{BASE}/competition_team_player_mapping.csv')
comp = pd.read_csv(f'{BASE}/competitions.csv')
cm = ctpm.merge(comp, on=['comp_id','comp_name'])
ipl26_map = cm[cm['comp_name']=='IPL 2026'].groupby('player_id')['team_name'].first().to_dict()
ipl_last = cm[cm['comp_name'].str.startswith('IPL')].sort_values('season').groupby('player_id').agg(
    last_ipl_season=('season','last'), last_ipl_team=('team_name','last')
)
t20i_nat = cm[cm['match_format']=='T20I'].groupby('player_id')['team_name'].first().to_dict()

# Dynamic wicket_value & PI
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
b_tf = tf[tf['is_batter_ball']].copy()
bw_tf = tf.copy()

def score_batters(df, q_thresh=0.75, k=500.0, min_tot=400, min_hp=180, min_hp_m=15):
    b = df.copy()
    cut = b['PI_bat'].quantile(q_thresh)
    b['is_hp'] = b['PI_bat'] >= cut
    b['w_pi'] = (b['PI_bat'] / 100.0) ** 2
    mean_w_hp = b.loc[b['is_hp'], 'w_pi'].mean()
    
    hp = b[b['is_hp']]; lp = b[~b['is_hp']]
    m_hp = hp.groupby(['striker_id','match_id'])['tva_bat_opp'].sum().reset_index()
    
    def w_agg(g):
        w = g['w_pi'].values
        ws = w.sum()
        return pd.Series({
            'w_n_eff': ws / mean_w_hp,
            'w_tva100': 100.0 * np.sum(w * g['tva_bat_opp'].values) / max(1e-9, ws),
            'w_wpa100': 100.0 * np.sum(w * g['wpa_bat'].values) / max(1e-9, ws),
        })
    cw = b.groupby(['striker_id','striker_name']).apply(w_agg, include_groups=False).reset_index()
    tot = b.groupby('striker_id').agg(
        tot_b=('is_batter_ball','sum'), tot_runs=('batter_runs','sum'),
        tot_m=('match_id','nunique'),
        ipl_b=('match_tier', lambda x: (x=='IPL').sum()),
        t20i_b=('match_tier', lambda x: (x=='T20I_FM_vs_FM').sum()),
        last_date=('match_date','max'),
    ).reset_index()
    ahp = hp.groupby('striker_id').agg(
        hp_b=('is_batter_ball','sum'), hp_m=('match_id','nunique'), hp_runs=('batter_runs','sum'),
        hp_outs=('is_striker_out','sum'),
        hp_sr=('batter_runs', lambda x: 100*x.mean()),
        hp_tva100=('tva_bat_opp', lambda x: 100*x.mean()),
        hp_wpa100=('wpa_bat', lambda x: 100*x.mean()),
        hp_raa100=('raa_bat', lambda x: 100*x.mean()),
        hp_waa100=('waa_bat', lambda x: 100*x.mean()),
        hp_bdry=('batter_runs', lambda x: 100*x.isin([4,6]).mean()),
        hp_dot=('batter_runs', lambda x: 100*(x==0).mean()),
    ).reset_index()
    ahp['hp_avg'] = ahp['hp_runs'] / ahp['hp_outs'].clip(lower=1)
    alp = lp.groupby('striker_id').agg(
        lp_b=('is_batter_ball','sum'), lp_sr=('batter_runs', lambda x: 100*x.mean()),
        lp_tva100=('tva_bat_opp', lambda x: 100*x.mean()),
        lp_bdry=('batter_runs', lambda x: 100*x.isin([4,6]).mean()),
        lp_dot=('batter_runs', lambda x: 100*(x==0).mean()),
    ).reset_index()
    mc = m_hp.groupby('striker_id')['tva_bat_opp'].agg(
        pos_m_rate=lambda x: (x > 0).mean(), m_mean='mean', m_std='std', m_med='median'
    ).reset_index()
    
    r = cw.merge(tot, on='striker_id').merge(ahp, on='striker_id').merge(alp, on='striker_id').merge(mc, on='striker_id')
    r = r[(r['tot_b']>=min_tot) & (r['hp_b']>=min_hp) & (r['hp_m']>=min_hp_m)].copy()
    
    r['hp_tva_eb'] = (r['hp_b'] / (r['hp_b'] + k)) * r['hp_tva100']
    r['cont_tva_eb'] = (r['w_n_eff'] / (r['w_n_eff'] + k)) * r['w_tva100']
    r['blended_hp_tva'] = 0.5 * r['hp_tva_eb'] + 0.5 * r['cont_tva_eb']
    r['lp_tva_eb'] = (r['lp_b'] / (r['lp_b'] + k)) * r['lp_tva100']
    r['hp_wpa_eb'] = 0.5*(r['hp_b']/(r['hp_b']+k))*r['hp_wpa100'] + 0.5*(r['w_n_eff']/(r['w_n_eff']+k))*r['w_wpa100']
    
    # Fit HP ~ LP regression, then floor LP at 0 in expectation so negative LP gives NO artificial credit!
    lr = LinearRegression().fit(r[['lp_tva_eb']], r['blended_hp_tva'])
    lp_floored = np.maximum(0.0, r['lp_tva_eb'])
    r['exp_hp_from_lp'] = lr.intercept_ + lr.coef_[0] * lp_floored
    r['clutch_resid'] = r['blended_hp_tva'] - r['exp_hp_from_lp']
    r['m_lcb'] = r['m_mean'] - r['m_std'].fillna(10.0) / np.sqrt(r['hp_m'])
    
    def z(s): return (s - s.mean()) / (s.std() + 1e-9)
    r['PRS_z'] = 0.50*z(r['blended_hp_tva']) + 0.20*z(r['hp_wpa_eb']) + 0.15*z(r['clutch_resid']) + 0.15*(0.5*z(r['pos_m_rate']) + 0.5*z(r['m_lcb']))
    # Also compute 0-100 scaled PRS score for readability (50 + 15*z, clipped or logistic? Linear 0-100 min-max or percentile/T-score!)
    r = r.sort_values('PRS_z', ascending=False).reset_index(drop=True)
    r['rank'] = np.arange(1, len(r)+1)
    r['ipl_2026_team'] = r['striker_id'].map(ipl26_map).fillna('Non-IPL26')
    r['last_ipl_season'] = r['striker_id'].map(ipl_last['last_ipl_season'])
    r['last_ipl_team'] = r['striker_id'].map(ipl_last['last_ipl_team']).fillna('None')
    r['t20i_team'] = r['striker_id'].map(t20i_nat).fillna('Unmapped')
    return r, cut

def score_bowlers(df, q_thresh=0.75, k=500.0, min_tot=400, min_hp=180, min_hp_m=15):
    bw = df.copy()
    cut = bw['PI_bowl'].quantile(q_thresh)
    bw['is_hp'] = bw['PI_bowl'] >= cut
    bw['w_pi'] = (bw['PI_bowl'] / 100.0) ** 2
    mean_w_hp = bw.loc[bw['is_hp'], 'w_pi'].mean()
    
    hp = bw[bw['is_hp']]; lp = bw[~bw['is_hp']]
    m_hp = hp.groupby(['bowler_id','match_id'])['tva_bowl_opp'].sum().reset_index()
    
    def w_agg(g):
        w = g['w_pi'].values
        ws = w.sum()
        return pd.Series({
            'w_n_eff': ws / mean_w_hp,
            'w_tva100': 100.0 * np.sum(w * g['tva_bowl_opp'].values) / max(1e-9, ws),
            'w_wpa100': 100.0 * np.sum(w * g['wpa_bowl'].values) / max(1e-9, ws),
        })
    cw = bw.groupby(['bowler_id','bowler_name']).apply(w_agg, include_groups=False).reset_index()
    tot = bw.groupby('bowler_id').agg(
        tot_leg=('is_legal','sum'), tot_wkts=('is_bowler_wkt','sum'),
        tot_m=('match_id','nunique'),
        ipl_b=('match_tier', lambda x: (x=='IPL').sum()),
        t20i_b=('match_tier', lambda x: (x=='T20I_FM_vs_FM').sum()),
        last_date=('match_date','max'),
    ).reset_index()
    ahp = hp.groupby('bowler_id').agg(
        hp_deliv=('bowler_runs','size'), hp_leg=('is_legal','sum'), hp_m=('match_id','nunique'),
        hp_runs=('bowler_runs','sum'), hp_wkts=('is_bowler_wkt','sum'),
        hp_tva100=('tva_bowl_opp', lambda x: 100*x.mean()),
        hp_wpa100=('wpa_bowl', lambda x: 100*x.mean()),
        hp_raa100=('raa_bowl', lambda x: 100*x.mean()),
        hp_waa100=('waa_bowl', lambda x: 100*x.mean()),
        hp_dot=('bowler_runs', lambda x: 100*(x==0).mean()),
        hp_bdry=('runs', lambda x: 100*x.isin([4,6]).mean()),
    ).reset_index()
    ahp['hp_econ'] = 6.0 * ahp['hp_runs'] / ahp['hp_leg'].clip(lower=1)
    ahp['hp_sr'] = ahp['hp_leg'] / ahp['hp_wkts'].clip(lower=1)
    alp = lp.groupby('bowler_id').agg(
        lp_deliv=('bowler_runs','size'), lp_leg=('is_legal','sum'), lp_runs=('bowler_runs','sum'),
        lp_tva100=('tva_bowl_opp', lambda x: 100*x.mean()),
        lp_dot=('bowler_runs', lambda x: 100*(x==0).mean()),
    ).reset_index()
    alp['lp_econ'] = 6.0 * alp['lp_runs'] / alp['lp_leg'].clip(lower=1)
    mc = m_hp.groupby('bowler_id')['tva_bowl_opp'].agg(
        pos_m_rate=lambda x: (x > 0).mean(), m_mean='mean', m_std='std', m_med='median'
    ).reset_index()
    
    r = cw.merge(tot, on='bowler_id').merge(ahp, on='bowler_id').merge(alp, on='bowler_id').merge(mc, on='bowler_id')
    r = r[(r['tot_leg']>=min_tot) & (r['hp_leg']>=min_hp) & (r['hp_m']>=min_hp_m)].copy()
    
    r['hp_tva_eb'] = (r['hp_deliv'] / (r['hp_deliv'] + k)) * r['hp_tva100']
    r['cont_tva_eb'] = (r['w_n_eff'] / (r['w_n_eff'] + k)) * r['w_tva100']
    r['blended_hp_tva'] = 0.5 * r['hp_tva_eb'] + 0.5 * r['cont_tva_eb']
    r['lp_tva_eb'] = (r['lp_deliv'] / (r['lp_deliv'] + k)) * r['lp_tva100']
    r['hp_wpa_eb'] = 0.5*(r['hp_deliv']/(r['hp_deliv']+k))*r['hp_wpa100'] + 0.5*(r['w_n_eff']/(r['w_n_eff']+k))*r['w_wpa100']
    
    lr = LinearRegression().fit(r[['lp_tva_eb']], r['blended_hp_tva'])
    lp_floored = np.maximum(0.0, r['lp_tva_eb'])
    r['exp_hp_from_lp'] = lr.intercept_ + lr.coef_[0] * lp_floored
    r['clutch_resid'] = r['blended_hp_tva'] - r['exp_hp_from_lp']
    r['m_lcb'] = r['m_mean'] - r['m_std'].fillna(10.0) / np.sqrt(r['hp_m'])
    
    def z(s): return (s - s.mean()) / (s.std() + 1e-9)
    r['PRS_z'] = 0.50*z(r['blended_hp_tva']) + 0.20*z(r['hp_wpa_eb']) + 0.15*z(r['clutch_resid']) + 0.15*(0.5*z(r['pos_m_rate']) + 0.5*z(r['m_lcb']))
    r = r.sort_values('PRS_z', ascending=False).reset_index(drop=True)
    r['rank'] = np.arange(1, len(r)+1)
    r['ipl_2026_team'] = r['bowler_id'].map(ipl26_map).fillna('Non-IPL26')
    r['last_ipl_season'] = r['bowler_id'].map(ipl_last['last_ipl_season'])
    r['last_ipl_team'] = r['bowler_id'].map(ipl_last['last_ipl_team']).fillna('None')
    r['t20i_team'] = r['bowler_id'].map(t20i_nat).fillna('Unmapped')
    return r, cut

rb, cut_b = score_batters(b_tf)
rbw, cut_bw = score_bowlers(bw_tf)

print('=== FINAL CANDIDATE TOP 15 BATTERS (Floored LP Clutch Resid + Continuous PI^2 Blended) ===')
print(rb[['rank','striker_name','tot_b','hp_b','hp_m','hp_sr','blended_hp_tva','lp_tva_eb','clutch_resid','hp_wpa_eb','pos_m_rate','PRS_z','last_ipl_season','last_ipl_team','t20i_team']].head(15).round(2).to_string(index=False))

print('\n=== FINAL CANDIDATE TOP 15 BOWLERS (Floored LP Clutch Resid + Continuous PI^2 Blended) ===')
print(rbw[['rank','bowler_name','tot_leg','hp_leg','hp_m','hp_econ','hp_wkts','blended_hp_tva','lp_tva_eb','clutch_resid','hp_wpa_eb','pos_m_rate','PRS_z','last_ipl_season','last_ipl_team','t20i_team']].head(15).round(2).to_string(index=False))

# Check IPL-only rank comparison too!
rb_ipl, _ = score_batters(b_tf[b_tf['match_tier']=='IPL'], min_tot=250, min_hp=100, min_hp_m=10)
rbw_ipl, _ = score_bowlers(bw_tf[bw_tf['match_tier']=='IPL'], min_tot=250, min_hp=100, min_hp_m=10)
print('\n=== IPL-ONLY TOP 12 BATTERS (min_tot=250, min_hp=100) ===')
print(rb_ipl[['rank','striker_name','tot_b','hp_b','hp_m','hp_sr','blended_hp_tva','PRS_z']].head(12).round(2).to_string(index=False))
print('\n=== IPL-ONLY TOP 12 BOWLERS (min_tot=250, min_hp=100) ===')
print(rbw_ipl[['rank','bowler_name','tot_leg','hp_leg','hp_m','hp_econ','blended_hp_tva','PRS_z']].head(12).round(2).to_string(index=False))
