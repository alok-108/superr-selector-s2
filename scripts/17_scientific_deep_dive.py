import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

BASE = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\data_pack\Data Pack'
OUT = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\outputs'

comp = pd.read_csv(f'{BASE}/competitions.csv')
matches = pd.read_csv(f'{BASE}/matches.csv')
bbb = pd.read_csv(f'{BASE}/ball_by_ball.csv', low_memory=False)
ctpm = pd.read_csv(f'{BASE}/competition_team_player_mapping.csv')
p11 = pd.read_csv(f'{BASE}/playing_eleven.csv')

eb = pd.read_csv(f'{OUT}/eligible_batters.csv')
ebw = pd.read_csv(f'{OUT}/eligible_bowlers.csv')
top10_b = pd.read_csv(f'{OUT}/top10_batters.csv')
top10_bw = pd.read_csv(f'{OUT}/top10_bowlers.csv')

print("Step 1: Reconstructing exact delivery frame with Match Roles (Captain/WK/Impact) & Venues...")
m = matches.merge(comp, on='comp_id', how='left')
full_members = {
    'India','Australia','England','South Africa','New Zealand','Pakistan',
    'West Indies','Sri Lanka','Bangladesh','Afghanistan','Ireland','Zimbabwe'
}
def classify_tier(row):
    if str(row['comp_name']).startswith('IPL'): return 'IPL'
    if 'Emerging' in str(row['comp_name']) or 'Rising' in str(row['comp_name']): return 'Emerging_A'
    if row['home_team'] in full_members and row['away_team'] in full_members: return 'T20I_FM_vs_FM'
    if row['home_team'] in full_members or row['away_team'] in full_members: return 'T20I_FM_vs_Assoc'
    return 'T20I_Assoc_vs_Assoc'

m['match_tier'] = m.apply(classify_tier, axis=1)
m['year'] = pd.to_datetime(m['match_date']).dt.year
m['is_knockout'] = ((m['knock_out'] == 1) | (m['stage'].isin(['Final','Qualifier','Eliminator','Semi Final']))).astype(int)

reg = bbb[bbb['innings'].isin([1, 2])].sort_values(['match_id', 'innings', 'overid', 'over_ball_num', 'innings_ball_num']).reset_index(drop=True)
reg['is_wide'] = reg['extras_type'] == 'wide'
reg['is_noball'] = reg['extras_type'].isin(['Overstep', 'other'])
reg['is_legal'] = ~(reg['is_wide'] | reg['is_noball'])
reg['is_batter_ball'] = ~reg['is_wide']
reg['tot_runs'] = reg['runs'] + reg['extras1_runs'] + reg['extras_2_runs']
reg['batter_runs'] = reg['runs']
reg['bowler_runs'] = np.where(
    reg['extras_type'].isin(['byes', 'legbyes', 'Penalty']), 0,
    np.where(reg['is_wide'], reg['extras1_runs'],
             reg['runs'] + np.where(reg['is_noball'], reg['extras1_runs'], 0))
)
non_bowler_dismissals = {'RunOut', 'Retire Out', 'Obstracting the field', 'Obstructing the field'}
reg['is_bowler_wkt'] = (reg['is_wicket']) & (~reg['dismissal_type'].isin(non_bowler_dismissals))
reg['is_striker_out'] = (reg['is_wicket']) & (reg['dismissed_player_id'] == reg['striker_id'])

g = reg.groupby(['match_id', 'innings'], sort=False)
reg['cum_runs_before'] = g['tot_runs'].cumsum() - reg['tot_runs']
reg['cum_wkts_before'] = g['is_wicket'].cumsum() - reg['is_wicket'].astype(int)
reg['legal_balls_before'] = g['is_legal'].cumsum() - reg['is_legal'].astype(int)
reg['runs_last18'] = reg['cum_runs_before'] - g['cum_runs_before'].shift(18).fillna(0)
reg['wkts_last18'] = reg['cum_wkts_before'] - g['cum_wkts_before'].shift(18).fillna(0)
gb = reg.groupby(['match_id', 'innings', 'striker_id'], sort=False)
reg['batter_balls_before'] = gb['is_batter_ball'].cumsum() - reg['is_batter_ball'].astype(int)

pace_off_styles = {'Off Cutter', 'Slow Bouncer', 'Leg Cut', 'Back of Hand', 'Knuckle', 'Slower Ball'}
reg['pace_cat'] = np.where(
    reg['pace_or_spin'] == 'Spin', 'Spin',
    np.where(reg['delivery_style'].isin(pace_off_styles) | ((reg['bowl_speed'] > 40) & (reg['bowl_speed'] < 123)), 'Pace-Off', 'Pace-On')
)

i1_totals = reg[reg['innings'] == 1].groupby('match_id')['tot_runs'].sum().rename('inn1_total')
m = m.merge(i1_totals, on='match_id', how='left')
m['max_balls_1'] = m['first_inn_revised_overs'].fillna(120).astype(int)
m['max_balls_2'] = m['second_inn_revised_overs'].fillna(120).astype(int)
m['target'] = np.where(m['revised_target'].notna(), m['revised_target'], m['inn1_total'] + 1)
dls_mid_cut = m['result'].str.contains('D/L|DLS', case=False, na=False) & m['revised_target'].isna()
m['has_valid_outcome'] = m['win_team_id'].notna() & (~dls_mid_cut)

reg = reg.merge(m[['match_id','comp_name','season','year','stage','is_knockout','venue_id','match_tier',
                   'max_balls_1','max_balls_2','target','inn1_total','win_team_id','has_valid_outcome','home_team_id']],
                on='match_id', how='left')
reg['max_balls'] = np.where(reg['innings'] == 1, reg['max_balls_1'], reg['max_balls_2'])
reg['balls_rem'] = (reg['max_balls'] - reg['legal_balls_before']).clip(lower=1)
reg['wickets_rem'] = (10 - reg['cum_wkts_before']).clip(lower=0)
reg['bat_won'] = (reg['batting_team_id'] == reg['win_team_id']).astype(float)
reg['phase'] = np.where(reg['overid'] <= 6, 'PP', np.where(reg['overid'] <= 15, 'Middle', 'Death'))

# Venue env_par_rpo (leave-one-match-out)
match_rpo = reg.groupby(['match_id','match_tier','season','venue_id']).agg(m_runs=('tot_runs','sum'), m_balls=('is_legal','sum')).reset_index()
ts_sum = match_rpo.groupby(['match_tier','season'])[['m_runs','m_balls']].transform('sum')
ts_loo_rpo = 6.0 * (ts_sum['m_runs'] - match_rpo['m_runs']) / (ts_sum['m_balls'] - match_rpo['m_balls']).clip(lower=1)
t_sum = match_rpo.groupby('match_tier')[['m_runs','m_balls']].transform('sum')
t_loo_rpo = 6.0 * (t_sum['m_runs'] - match_rpo['m_runs']) / (t_sum['m_balls'] - match_rpo['m_balls']).clip(lower=1)
ts_loo_rpo = ts_loo_rpo.fillna(t_loo_rpo)
v_sum = match_rpo.groupby('venue_id')[['m_runs','m_balls']].transform('sum')
match_rpo['env_par_rpo'] = 6.0 * ((v_sum['m_runs'] - match_rpo['m_runs']) + 1440.0 * (ts_loo_rpo / 6.0)) / ((v_sum['m_balls'] - match_rpo['m_balls']) + 1440.0)
reg = reg.merge(match_rpo[['match_id','env_par_rpo']], on='match_id', how='left')

# Fast resource + WP + PI (same exact production spec as 15_)
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(reg, groups=reg['match_id']))
res_feats = ['balls_rem', 'wickets_rem', 'env_par_rpo']
full_i1_all = reg[(reg['innings']==1)&(reg['max_balls']==120)&(reg['has_valid_outcome'])].copy()
full_i1_all['runs_to_go'] = full_i1_all['inn1_total'] - full_i1_all['cum_runs_before']
mdl_rem = HistGradientBoostingRegressor(max_iter=100, max_depth=4, min_samples_leaf=500, l2_regularization=20.0, monotonic_cst=[1,1,1], random_state=42).fit(full_i1_all[res_feats], full_i1_all['runs_to_go'])

b_grid = np.arange(1, 121); w_grid = np.arange(0, 11); r_grid = np.round(np.arange(5.0, 13.0, 0.25), 2)
bb, ww, rr = np.meshgrid(b_grid, w_grid, r_grid, indexing='ij')
grid_df = pd.DataFrame({'balls_rem': bb.ravel(), 'wickets_rem': ww.ravel(), 'env_par_rpo': rr.ravel()})
grid_pred = np.where(grid_df['wickets_rem'].values == 0, 0.0, np.maximum(0.0, mdl_rem.predict(grid_df)))
rem_lut = grid_pred.reshape(120, 11, len(r_grid))
def fast_rem_lookup(balls_rem_arr, wkts_rem_arr, env_rpo_arr):
    bi = np.clip(np.round(balls_rem_arr).astype(int), 1, 120) - 1
    wi = np.clip(np.round(wkts_rem_arr).astype(int), 0, 10)
    ri = np.clip(np.round((env_rpo_arr - 5.0) / 0.25).astype(int), 0, len(r_grid) - 1)
    return rem_lut[bi, wi, ri]

reg['exp_rem_runs'] = fast_rem_lookup(reg['balls_rem'].values, reg['wickets_rem'].values, reg['env_par_rpo'].values)
er_w_minus = np.where(reg['wickets_rem'].values <= 1, 0.0, fast_rem_lookup(reg['balls_rem'].values, np.maximum(0, reg['wickets_rem'].values - 1), reg['env_par_rpo'].values))
reg['wicket_value'] = np.clip(reg['exp_rem_runs'].values - er_w_minus, 2.0, 28.0)

def build_i1_feats(cum_r, lbb, br, wr, mb, env_r, is_ipl, exp_rem):
    par_tot = env_r * (mb / 6.0); proj_tot = cum_r + exp_rem; prog = lbb / mb; ps = proj_tot - par_tot
    return np.column_stack([ps, ps * (0.3 + 0.7 * prog), cum_r, wr, br, prog, env_r, is_ipl])
def build_i2_feats(cum_r, lbb, br, wr, mb, env_r, tgt, exp_rem):
    rn = np.maximum(0.0, tgt - cum_r); rrr = 6.0 * rn / np.maximum(1.0, br); cs = exp_rem - rn; prog = lbb / mb
    return np.column_stack([cs, cs / np.sqrt(np.maximum(1.0, br) + 6.0), np.log((rrr + 1.0)/(env_r + 1.0)), rn, br, wr, env_r, prog, wr * prog])

is_ipl_arr = (reg['match_tier'] == 'IPL').astype(float).values
cum_r = reg['cum_runs_before'].values; lbb = reg['legal_balls_before'].values; br = reg['balls_rem'].values
wr = reg['wickets_rem'].values; mb = reg['max_balls'].values; env_r = reg['env_par_rpo'].values
tgt = reg['target'].fillna(160).values; exp_r = reg['exp_rem_runs'].values
X1_all = build_i1_feats(cum_r, lbb, br, wr, mb, env_r, is_ipl_arr, exp_r)
X2_all = build_i2_feats(cum_r, lbb, br, wr, mb, env_r, tgt, exp_r)
def make_cf(add_r, add_w):
    cr_c = cum_r + add_r; lbb_c = np.minimum(mb, lbb + 1); br_c = np.maximum(1, br - 1); wr_c = np.maximum(0, wr - add_w)
    er_c = fast_rem_lookup(br_c, wr_c, env_r)
    return build_i1_feats(cr_c, lbb_c, br_c, wr_c, mb, env_r, is_ipl_arr, er_c), build_i2_feats(cr_c, lbb_c, br_c, wr_c, mb, env_r, tgt, er_c)
X1_0, X2_0 = make_cf(0, 0); X1_1, X2_1 = make_cf(1, 0); X1_4, X2_4 = make_cf(4, 0); X1_6, X2_6 = make_cf(6, 0); X1_w, X2_w = make_cf(0, 1)
wp_b = np.zeros(len(reg)); wp_0 = np.zeros(len(reg)); wp_1 = np.zeros(len(reg)); wp_4 = np.zeros(len(reg)); wp_6 = np.zeros(len(reg)); wp_w = np.zeros(len(reg))
for tr_idx, va_idx in folds:
    tr_m = np.zeros(len(reg), dtype=bool); tr_m[tr_idx] = True; va_m = np.zeros(len(reg), dtype=bool); va_m[va_idx] = True
    m1 = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=0.1))]).fit(X1_all[tr_m & (reg['innings'].values==1) & reg['has_valid_outcome'].values], reg.loc[tr_m & (reg['innings'].values==1) & reg['has_valid_outcome'].values, 'bat_won'])
    m2 = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=0.1))]).fit(X2_all[tr_m & (reg['innings'].values==2) & reg['has_valid_outcome'].values], reg.loc[tr_m & (reg['innings'].values==2) & reg['has_valid_outcome'].values, 'bat_won'])
    va1 = va_m & (reg['innings'].values==1); va2 = va_m & (reg['innings'].values==2)
    for arr, X in [(wp_b, X1_all),(wp_0, X1_0),(wp_1, X1_1),(wp_4, X1_4),(wp_6, X1_6),(wp_w, X1_w)]: arr[va1] = m1.predict_proba(X[va1])[:, 1]
    for arr, X in [(wp_b, X2_all),(wp_0, X2_0),(wp_1, X2_1),(wp_4, X2_4),(wp_6, X2_6),(wp_w, X2_w)]: arr[va2] = m2.predict_proba(X[va2])[:, 1]
i2_m = (reg['innings'].values == 2); last_b = (br == 1); all_out_w = (wr <= 1)
for arr, add_r, add_w in [(wp_0,0,0),(wp_1,1,0),(wp_4,4,0),(wp_6,6,0),(wp_w,0,1)]:
    won = i2_m & ((cum_r + add_r) >= tgt); lost = i2_m & (~won) & (last_b | ((add_w == 1) & all_out_w)); tie = i2_m & last_b & ((cum_r + add_r) == (tgt - 1))
    arr[won] = 1.0; arr[lost] = 0.0; arr[tie] = 0.5
reg['wp_before'] = wp_b
p_dot = ((reg['tot_runs']==0) & (~reg['is_wicket'])).groupby(reg['phase']).mean()
p_123 = ((reg['tot_runs'].isin([1,2,3])) & (~reg['is_wicket'])).groupby(reg['phase']).mean()
p_4 = ((reg['runs']==4) & (~reg['is_wicket'])).groupby(reg['phase']).mean()
p_6 = ((reg['runs']==6) & (~reg['is_wicket'])).groupby(reg['phase']).mean()
p_w = reg['is_wicket'].groupby(reg['phase']).mean()
w0, w1, w4, w6, ww = [reg['phase'].map(s).values for s in [p_dot, p_123, p_4, p_6, p_w]]
ws = w0 + w1 + w4 + w6 + ww; w0, w1, w4, w6, ww = w0/ws, w1/ws, w4/ws, w6/ws, ww/ws
mu_cf = w0*wp_0 + w1*wp_1 + w4*wp_4 + w6*wp_6 + ww*wp_w
reg['wp_lev'] = np.sqrt(w0*(wp_0-mu_cf)**2 + w1*(wp_1-mu_cf)**2 + w4*(wp_4-mu_cf)**2 + w6*(wp_6-mu_cf)**2 + ww*(wp_w-mu_cf)**2)

live_gate = np.where(reg['innings']==2, np.clip((np.minimum(reg['wp_before'], 1-reg['wp_before'])-0.03)/0.14, 0, 1), np.clip((np.minimum(reg['wp_before'], 1-reg['wp_before'])-0.05)/0.15, 0.35, 1))
collapse_excess = np.maximum(0.0, reg['cum_wkts_before'] - 0.042*reg['legal_balls_before'])
collapse_stress = np.where(reg['cum_wkts_before'] >= 2, np.tanh(collapse_excess / 1.5), 0.0) * live_gate
bat_shock = np.tanh(np.maximum(0.0, reg['wkts_last18'] - 1.0) / 1.2) * live_gate
reg['rrr'] = np.where(reg['innings']==2, 6.0 * np.maximum(0, reg['target'] - reg['cum_runs_before']) / reg['balls_rem'], 0.0)
rrr_stress = np.where(reg['innings']==2, np.tanh(np.maximum(0.0, reg['rrr'] - reg['env_par_rpo']) / 4.0) * live_gate, 0.0)
death_ramp = np.where(reg['overid'] >= 16, 0.55 + 0.45 * (reg['overid'] - 16) / 4.0, 0.0) * live_gate
lev_score = np.where(reg['innings']==2, np.tanh(reg['wp_lev'] / 0.048), 0.55 * np.tanh(reg['wp_lev'] / 0.025) * live_gate)
bowl_defense_risk = np.where(reg['innings']==2, np.clip((reg['wp_before'] - 0.12) / 0.50, 0.0, 1.0) * live_gate, 0.22 * live_gate)
bowl_surge = np.tanh(np.maximum(0.0, reg['runs_last18'] - 26.0) / 15.0) * live_gate
ko_boost = np.where(reg['is_knockout'] == 1, 0.12, 0.0) * np.maximum(0.35, live_gate)
reg['PI_bat'] = 100.0 * np.clip(0.38*lev_score + 0.34*np.maximum(collapse_stress, 0.6*bat_shock) + 0.24*rrr_stress + 0.24*death_ramp + ko_boost, 0, 1)
reg['PI_bowl'] = 100.0 * np.clip(0.40*lev_score + 0.32*death_ramp + 0.20*bowl_defense_risk + 0.18*bowl_surge + ko_boost, 0, 1)

# Fit OOF xR, xW & RAPM Ridge
reg['exp_bat_runs'] = np.nan; reg['exp_bat_out'] = np.nan; reg['exp_bowl_runs'] = np.nan; reg['exp_bowl_wkt'] = np.nan
X_ctx_bat = reg[['match_tier','pace_or_spin','innings','env_par_rpo','overid','wickets_rem']].copy()
X_ctx_bowl = reg[['match_tier','pace_or_spin','innings','env_par_rpo','overid','wickets_rem','batting_order']].copy()
for c in ['match_tier','pace_or_spin']: X_ctx_bat[c] = X_ctx_bat[c].astype('category'); X_ctx_bowl[c] = X_ctx_bowl[c].astype('category')
for tr_idx, va_idx in folds:
    tr_b = tr_idx[reg.iloc[tr_idx]['is_batter_ball'].values]; va_b = va_idx[reg.iloc[va_idx]['is_batter_ball'].values]
    reg.iloc[va_b, reg.columns.get_loc('exp_bat_runs')] = HistGradientBoostingRegressor(max_iter=80, max_depth=4, min_samples_leaf=400, categorical_features=['match_tier','pace_or_spin'], random_state=42).fit(X_ctx_bat.iloc[tr_b], reg.iloc[tr_b]['batter_runs']).predict(X_ctx_bat.iloc[va_b])
    reg.iloc[va_b, reg.columns.get_loc('exp_bat_out')] = HistGradientBoostingClassifier(max_iter=80, max_depth=4, min_samples_leaf=400, categorical_features=['match_tier','pace_or_spin'], random_state=42).fit(X_ctx_bat.iloc[tr_b], reg.iloc[tr_b]['is_striker_out'].astype(int)).predict_proba(X_ctx_bat.iloc[va_b])[:, 1]
    reg.iloc[va_idx, reg.columns.get_loc('exp_bowl_runs')] = HistGradientBoostingRegressor(max_iter=80, max_depth=4, min_samples_leaf=400, categorical_features=['match_tier','pace_or_spin'], random_state=42).fit(X_ctx_bowl.iloc[tr_idx], reg.iloc[tr_idx]['bowler_runs']).predict(X_ctx_bowl.iloc[va_idx])
    reg.iloc[va_idx, reg.columns.get_loc('exp_bowl_wkt')] = HistGradientBoostingClassifier(max_iter=80, max_depth=4, min_samples_leaf=400, categorical_features=['match_tier','pace_or_spin'], random_state=42).fit(X_ctx_bowl.iloc[tr_idx], reg.iloc[tr_idx]['is_bowler_wkt'].astype(int)).predict_proba(X_ctx_bowl.iloc[va_idx])[:, 1]

reg['tva_bat'] = (reg['batter_runs'] - reg['exp_bat_runs']) + reg['wicket_value'] * (reg['exp_bat_out'] - reg['is_striker_out'].astype(float))
reg['tva_bowl'] = (reg['exp_bowl_runs'] - reg['bowler_runs']) + reg['wicket_value'] * (reg['is_bowler_wkt'].astype(float) - reg['exp_bowl_wkt'])

ohe_bat = OneHotEncoder(handle_unknown='ignore'); ohe_bowl = OneHotEncoder(handle_unknown='ignore')
ohe_bteam = OneHotEncoder(handle_unknown='ignore'); ohe_bwteam = OneHotEncoder(handle_unknown='ignore')
S_all = ohe_bat.fit_transform(reg[['striker_id']]); B_all = ohe_bowl.fit_transform(reg[['bowler_id']])
BT_all = ohe_bteam.fit_transform(reg[['batting_team_id']]); BWT_all = ohe_bwteam.fit_transform(reg[['bowling_team_id']])
X_rapm_bat = hstack([S_all, B_all, BWT_all]).tocsr(); X_rapm_bowl = hstack([B_all, S_all, BT_all]).tocsr()
n_s = S_all.shape[1]; n_b = B_all.shape[1]
reg['rapm_opp_bat_adj'] = 0.0; reg['rapm_opp_bowl_adj'] = 0.0
for tr_idx, va_idx in folds:
    tr_b = tr_idx[reg.iloc[tr_idx]['is_batter_ball'].values]; va_b = va_idx[reg.iloc[va_idx]['is_batter_ball'].values]
    rd1 = Ridge(alpha=1200.0, fit_intercept=True).fit(X_rapm_bat[tr_b], reg.iloc[tr_b]['tva_bat'])
    reg.iloc[va_b, reg.columns.get_loc('rapm_opp_bat_adj')] = X_rapm_bat[va_b][:, n_s:].dot(rd1.coef_[n_s:])
    rd2 = Ridge(alpha=1200.0, fit_intercept=True).fit(X_rapm_bowl[tr_idx], reg.iloc[tr_idx]['tva_bowl'])
    reg.iloc[va_idx, reg.columns.get_loc('rapm_opp_bowl_adj')] = X_rapm_bowl[va_idx][:, n_b:].dot(rd2.coef_[n_b:])
reg['tva_bat_opp'] = reg['tva_bat'] - reg['rapm_opp_bat_adj']
reg['tva_bowl_opp'] = reg['tva_bowl'] - reg['rapm_opp_bowl_adj']

tf = reg[reg['match_tier'].isin(['IPL','T20I_FM_vs_FM'])].copy()
cut_b = tf.loc[tf['is_batter_ball'], 'PI_bat'].quantile(0.75)
cut_bw = tf['PI_bowl'].quantile(0.75)
tf['hp_bat'] = tf['PI_bat'] >= cut_b
tf['hp_bowl'] = tf['PI_bowl'] >= cut_bw

print("\n============================================================")
print("A. TEMPORAL OUT-OF-SAMPLE PREDICTIVE TEST (2022-24 -> 2025-26)")
print("============================================================")
b_tr = tf[tf['is_batter_ball'] & (tf['year']<=2024)].copy()
b_te = tf[tf['is_batter_ball'] & (tf['year']>=2025)].copy()

def full_bat_window_score(df_sub, min_hp=80):
    k = 500.0
    d = df_sub.copy()
    d['w'] = (d['PI_bat'] / 100.0) ** 2
    mw = d.loc[d['hp_bat'], 'w'].mean()
    hp = d[d['hp_bat']]; lp = d[~d['hp_bat']]
    cw = d.groupby('striker_id').apply(lambda gr: pd.Series({
        'wn': gr['w'].sum() / mw,
        'wtva': 100.0 * (gr['w'].values * gr['tva_bat_opp'].values).sum() / max(1e-9, gr['w'].sum()),
        'wwpa': 100.0 * (gr['w'].values * gr['wpa_bat'].values).sum() / max(1e-9, gr['w'].sum()) if 'wpa_bat' in gr else 0.0
    }), include_groups=False).reset_index()
    tot = d.groupby('striker_id').agg(tot_b=('is_batter_ball','sum'), raw_sr=('batter_runs', lambda x: 100*x.mean())).reset_index()
    ahp = hp.groupby('striker_id').agg(hp_b=('is_batter_ball','sum'), hp_m=('match_id','nunique'), hp_sr=('batter_runs', lambda x: 100*x.mean()), hp_tva=('tva_bat_opp', lambda x: 100*x.mean()), hp_wpa=('wpa_bat', lambda x: 100*x.mean())).reset_index()
    alp = lp.groupby('striker_id').agg(lp_b=('is_batter_ball','sum'), lp_tva=('tva_bat_opp', lambda x: 100*x.mean())).reset_index()
    m_hp = hp.groupby(['striker_id','match_id'])['tva_bat_opp'].sum().reset_index()
    mc = m_hp.groupby('striker_id')['tva_bat_opp'].agg(pos_m=lambda x:(x>0).mean(), m_mean='mean', m_std='std').reset_index()
    r = cw.merge(tot, on='striker_id').merge(ahp, on='striker_id').merge(alp, on='striker_id').merge(mc, on='striker_id')
    r = r[r['hp_b']>=min_hp].copy()
    r['naive_clutch'] = r['hp_tva'] - r['lp_tva']
    r['blended'] = 0.5*(r['hp_b']/(r['hp_b']+k))*r['hp_tva'] + 0.5*(r['wn']/(r['wn']+k))*r['wtva']
    r['lp_eb'] = (r['lp_b']/(r['lp_b']+k))*r['lp_tva']
    r['wpa_eb'] = 0.5*(r['hp_b']/(r['hp_b']+k))*r['hp_wpa'] + 0.5*(r['wn']/(r['wn']+k))*r['wwpa']
    lr = LinearRegression().fit(r[['lp_eb']], r['blended'])
    r['cr'] = r['blended'] - (lr.intercept_ + lr.coef_[0]*np.maximum(0.0, r['lp_eb']))
    r['tstat'] = r['m_mean'] / (r['m_std'].fillna(10.0) / np.sqrt(r['hp_m']))
    def z(s): return (s - s.mean()) / (s.std() + 1e-9)
    r['prs_pred'] = 0.50*z(r['blended']) + 0.20*z(r['wpa_eb']) + 0.15*z(r['cr']) + 0.15*(0.5*z(r['pos_m']) + 0.5*z(r['tstat']))
    return r

# Ensure wpa_bat/wpa_bowl are populated
reg['wpa_bat'] = 0.0; reg['wpa_bowl'] = 0.0
br_a = np.maximum(1, mb - np.minimum(mb, lbb + reg['is_legal'].astype(int).values))
wr_a = np.maximum(0, wr - reg['is_wicket'].astype(int).values)
cr_a = cum_r + reg['tot_runs'].values
lbb_a = np.minimum(mb, lbb + reg['is_legal'].astype(int).values)
er_a = fast_rem_lookup(br_a, wr_a, env_r)
X1_a = build_i1_feats(cr_a, lbb_a, br_a, wr_a, mb, env_r, is_ipl_arr, er_a)
X2_a = build_i2_feats(cr_a, lbb_a, br_a, wr_a, mb, env_r, tgt, er_a)
wp_a = np.zeros(len(reg))
for tr_idx, va_idx in folds:
    tr_m = np.zeros(len(reg), dtype=bool); tr_m[tr_idx] = True; va_m = np.zeros(len(reg), dtype=bool); va_m[va_idx] = True
    m1 = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=0.1))]).fit(X1_all[tr_m & (reg['innings'].values==1) & reg['has_valid_outcome'].values], reg.loc[tr_m & (reg['innings'].values==1) & reg['has_valid_outcome'].values, 'bat_won'])
    m2 = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=0.1))]).fit(X2_all[tr_m & (reg['innings'].values==2) & reg['has_valid_outcome'].values], reg.loc[tr_m & (reg['innings'].values==2) & reg['has_valid_outcome'].values, 'bat_won'])
    va1 = va_m & (reg['innings'].values==1); va2 = va_m & (reg['innings'].values==2)
    wp_a[va1] = m1.predict_proba(X1_a[va1])[:, 1]
    wp_a[va2] = m2.predict_proba(X2_a[va2])[:, 1]
br_rem_after = mb - lbb_a
won_a = i2_m & (cr_a >= tgt); lost_a = i2_m & (~won_a) & ((br_rem_after == 0) | (wr_a == 0)); tie_a = i2_m & (br_rem_after == 0) & (cr_a == (tgt - 1))
wp_a[won_a] = 1.0; wp_a[lost_a] = 0.0; wp_a[tie_a] = 0.5
tf['wpa_bat'] = wp_a[tf.index] - tf['wp_before']
tf['wpa_bowl'] = -tf['wpa_bat']
b_tr = tf[tf['is_batter_ball'] & (tf['year']<=2024)].copy()
b_te = tf[tf['is_batter_ball'] & (tf['year']>=2025)].copy()

tr_b_m = full_bat_window_score(b_tr, min_hp=80)
te_b_m = b_te[b_te['hp_bat']].groupby('striker_id').agg(fut_hp_b=('is_batter_ball','sum'), fut_hp_tva=('tva_bat_opp', lambda x: 100*x.mean())).reset_index()
te_b_m = te_b_m[te_b_m['fut_hp_b']>=80]
bt_eval = tr_b_m.merge(te_b_m, on='striker_id')
print(f"Batters Temporal Cohort (n={len(bt_eval)}, >=80 HP balls in BOTH 2022-24 and 2025-26):")
for col, lbl in [('raw_sr','1. 2022-24 Overall SR'),('hp_sr','2. 2022-24 Raw HP SR'),('naive_clutch','3. 2022-24 Naive Clutch (HP-LP)'),('hp_tva','4. 2022-24 Unshrunk HP TVA'),('blended','5. 2022-24 Blended EB+RAPM HP TVA'),('prs_pred','6. 2022-24 Full RAPM PRS Score')]:
    rp = pearsonr(bt_eval[col], bt_eval['fut_hp_tva'])
    rs = spearmanr(bt_eval[col], bt_eval['fut_hp_tva'])
    print(f"  {lbl:36s}: Pearson r = {rp.statistic:+.3f} (p={rp.pvalue:.4f}) | Spearman rho = {rs.statistic:+.3f}")

bw_tr = tf[tf['year']<=2024].copy(); bw_te = tf[tf['year']>=2025].copy()
def full_bowl_window_score(df_sub, min_hp=80):
    k = 500.0
    d = df_sub.copy()
    d['w'] = (d['PI_bowl'] / 100.0) ** 2
    mw = d.loc[d['hp_bowl'], 'w'].mean()
    hp = d[d['hp_bowl']]; lp = d[~d['hp_bowl']]
    cw = d.groupby('bowler_id').apply(lambda gr: pd.Series({
        'wn': gr['w'].sum() / mw,
        'wtva': 100.0 * (gr['w'].values * gr['tva_bowl_opp'].values).sum() / max(1e-9, gr['w'].sum()),
        'wwpa': 100.0 * (gr['w'].values * gr['wpa_bowl'].values).sum() / max(1e-9, gr['w'].sum()) if 'wpa_bowl' in gr else 0.0
    }), include_groups=False).reset_index()
    tot = d.groupby('bowler_id').agg(leg=('is_legal','sum'), r=('bowler_runs','sum')).assign(neg_econ=lambda q:-6*q.r/q.leg).reset_index()
    ahp = hp.groupby('bowler_id').agg(hp_leg=('is_legal','sum'), hp_m=('match_id','nunique'), hp_r=('bowler_runs','sum'), hp_tva=('tva_bowl_opp', lambda x: 100*x.mean()), hp_wpa=('wpa_bowl', lambda x: 100*x.mean())).assign(neg_hp_econ=lambda q:-6*q.hp_r/q.hp_leg).reset_index()
    alp = lp.groupby('bowler_id').agg(lp_leg=('is_legal','sum'), lp_tva=('tva_bowl_opp', lambda x: 100*x.mean())).reset_index()
    m_hp = hp.groupby(['bowler_id','match_id'])['tva_bowl_opp'].sum().reset_index()
    mc = m_hp.groupby('bowler_id')['tva_bowl_opp'].agg(pos_m=lambda x:(x>0).mean(), m_mean='mean', m_std='std').reset_index()
    r = cw.merge(tot[['bowler_id','neg_econ']], on='bowler_id').merge(ahp, on='bowler_id').merge(alp, on='bowler_id').merge(mc, on='bowler_id')
    r = r[r['hp_leg']>=min_hp].copy()
    r['naive_clutch'] = r['hp_tva'] - r['lp_tva']
    r['blended'] = 0.5*(r['hp_leg']/(r['hp_leg']+k))*r['hp_tva'] + 0.5*(r['wn']/(r['wn']+k))*r['wtva']
    r['lp_eb'] = (r['lp_leg']/(r['lp_leg']+k))*r['lp_tva']
    r['wpa_eb'] = 0.5*(r['hp_leg']/(r['hp_leg']+k))*r['hp_wpa'] + 0.5*(r['wn']/(r['wn']+k))*r['wwpa']
    lr = LinearRegression().fit(r[['lp_eb']], r['blended'])
    r['cr'] = r['blended'] - (lr.intercept_ + lr.coef_[0]*np.maximum(0.0, r['lp_eb']))
    r['tstat'] = r['m_mean'] / (r['m_std'].fillna(10.0) / np.sqrt(r['hp_m']))
    def z(s): return (s - s.mean()) / (s.std() + 1e-9)
    r['prs_pred'] = 0.50*z(r['blended']) + 0.20*z(r['wpa_eb']) + 0.15*z(r['cr']) + 0.15*(0.5*z(r['pos_m']) + 0.5*z(r['tstat']))
    return r
tr_bw_m = full_bowl_window_score(bw_tr, min_hp=80)
te_bw_m = bw_te[bw_te['hp_bowl']].groupby('bowler_id').agg(fut_hp_leg=('is_legal','sum'), fut_hp_tva=('tva_bowl_opp', lambda x: 100*x.mean())).reset_index()
te_bw_m = te_bw_m[te_bw_m['fut_hp_leg']>=80]
bw_eval = tr_bw_m.merge(te_bw_m, on='bowler_id')
print(f"\nBowlers Temporal Cohort (n={len(bw_eval)}, >=80 HP balls in BOTH 2022-24 and 2025-26):")
for col, lbl in [('neg_econ','1. 2022-24 Overall (-Econ)'),('neg_hp_econ','2. 2022-24 Raw HP (-Econ)'),('naive_clutch','3. 2022-24 Naive Clutch (HP-LP)'),('hp_tva','4. 2022-24 Unshrunk HP TVA'),('blended','5. 2022-24 Blended EB+RAPM HP TVA'),('prs_pred','6. 2022-24 Full RAPM PRS Score')]:
    rp = pearsonr(bw_eval[col], bw_eval['fut_hp_tva'])
    rs = spearmanr(bw_eval[col], bw_eval['fut_hp_tva'])
    print(f"  {lbl:36s}: Pearson r = {rp.statistic:+.3f} (p={rp.pvalue:.4f}) | Spearman rho = {rs.statistic:+.3f}")

print("\n============================================================")
print("B. MECHANISMS: TOP-10 vs DIP-10 BATTERS & BOWLERS (LP -> HP)")
print("============================================================")
dip10_b = eb[eb['tot_b']>=500].sort_values('clutch_resid').head(10)
dip10_bw = ebw[ebw['tot_leg']>=500].sort_values('clutch_resid').head(10)
b_tf = tf[tf['is_batter_ball']].copy()
for lbl, ids in [('TOP-10 BATTERS', set(top10_b['striker_id'])), ('DIP-10 BATTERS (>=500b)', set(dip10_b['striker_id']))]:
    sub = b_tf[b_tf['striker_id'].isin(ids)]
    for is_h, hnm in [(False,'LP'), (True,'HP')]:
        s = sub[sub['hp_bat']==is_h]
        bdry = 100*s['batter_runs'].isin([4,6]).mean()
        dot = 100*(s['batter_runs']==0).mean()
        outp = 100*s['is_striker_out'].mean()
        sr = 100*s['batter_runs'].mean()
        tva = 100*s['tva_bat_opp'].mean()
        att_s = 100*s['stroke'].isin(['Straight Drive','Pull shot','Cover Drive','On drive','Flick','Paddle Scoop','Slog Sweep']).mean()
        slog_def = 100*s['stroke'].isin(['Slog','Frontfoot defensive','Backfoot defensive','Push']).mean()
        print(f"  {lbl:25s} [{hnm}]: n={len(s):5d} | SR {sr:5.1f} | Dot {dot:4.1f}% | Bdry {bdry:4.1f}% | Out {outp:4.2f}% | AttStroke {att_s:4.1f}% | Slog/Def {slog_def:4.1f}% | TVA {tva:+5.1f}")

for lbl, ids in [('TOP-10 BOWLERS', set(top10_bw['bowler_id'])), ('DIP-10 BOWLERS (>=500b)', set(dip10_bw['bowler_id']))]:
    sub = tf[tf['bowler_id'].isin(ids)]
    for is_h, hnm in [(False,'LP'), (True,'HP')]:
        s = sub[sub['hp_bowl']==is_h]
        dot = 100*(s['bowler_runs']==0).mean()
        bdry = 100*s['runs'].isin([4,6]).mean()
        wkt = 100*s['is_bowler_wkt'].mean()
        econ = 6*s['bowler_runs'].sum()/s['is_legal'].sum()
        poff = 100*(s.loc[s['pace_or_spin']=='Pace','pace_cat']=='Pace-Off').mean()
        york_sb = 100*s['delivery_style'].isin(['Yorker','Wide Yorker','Slow Bouncer','Off Cutter','Back of Hand','Carrom']).mean()
        freebie = 100*s['delivery_style'].isin(['Full Toss','Straight']).mean()
        around_p = 100*(s['over_or_around']=='Around').mean()
        print(f"  {lbl:25s} [{hnm}]: n={len(s):5d} | Econ {econ:4.2f} | Dot {dot:4.1f}% | Bdry {bdry:4.1f}% | Wkt {wkt:4.2f}% | PaceOff% {poff:4.1f}% | EliteVar% {york_sb:4.1f}% | FullToss/Str% {freebie:4.1f}% | Around% {around_p:4.1f}%")

print("\n============================================================")
print("C. PLAYING ELEVEN CONTEXT: CAPTAIN, WK, IMPACT PLAYER & LHB/RHB")
print("============================================================")
b_p = tf[tf['is_batter_ball']].merge(p11[['match_id','player_id','is_captain','is_wk','is_impact']], left_on=['match_id','striker_id'], right_on=['match_id','player_id'], how='left')
bw_p = tf.merge(p11[['match_id','player_id','is_captain','is_wk','is_impact']], left_on=['match_id','bowler_id'], right_on=['match_id','player_id'], how='left')
print("IPL 2023-26 Batters Impact vs Non-Impact under HP:")
b_ipl23 = b_p[(b_p['match_tier']=='IPL') & (b_p['year']>=2023) & (b_p['hp_bat'])]
print(b_ipl23.groupby('is_impact').agg(n=('tva_bat_opp','size'), sr=('batter_runs',lambda x:100*x.mean()), tva100=('tva_bat_opp',lambda x:100*x.mean())).round(2))
print("IPL 2023-26 Bowlers Impact vs Non-Impact under HP:")
bw_ipl23 = bw_p[(bw_p['match_tier']=='IPL') & (bw_p['year']>=2023) & (bw_p['hp_bowl'])]
print(bw_ipl23.groupby('is_impact').agg(n=('tva_bowl_opp','size'), econ=('bowler_runs',lambda x:6*x.mean()), tva100=('tva_bowl_opp',lambda x:100*x.mean())).round(2))
print("Captain vs Non-Captain HP Batting & Bowling:")
print(b_p[b_p['hp_bat']].groupby('is_captain').agg(n=('tva_bat_opp','size'), tva100=('tva_bat_opp',lambda x:100*x.mean())).round(2))
print(bw_p[bw_p['hp_bowl']].groupby('is_captain').agg(n=('tva_bowl_opp','size'), tva100=('tva_bowl_opp',lambda x:100*x.mean())).round(2))

print("\n============================================================")
print("D. RR HOME VENUES (JAIPUR #6 & GUWAHATI #1241) + LHB/RHB BOWLING SPLITS")
print("============================================================")
for vid, vnm in [(6,'Jaipur (#6)'), (1241,'Guwahati (#1241)')]:
    vb = tf[(tf['venue_id']==vid) & (tf['match_tier']=='IPL')]
    print(f"\n{vnm} - Pace vs Spin under HP vs LP:")
    print(vb.groupby(['hp_bowl','pace_cat']).agg(n=('bowler_runs','size'), econ=('bowler_runs',lambda x:6*x.sum()/max(1,vb.loc[x.index,'is_legal'].sum())), bdry=('runs',lambda x:100*x.isin([4,6]).mean())).round(2))

print("\nTop-10 Bowlers vs LHB & RHB under High Pressure (tva_bowl_opp/100):")
hp_bw_top = tf[tf['hp_bowl'] & tf['bowler_id'].isin(top10_bw['bowler_id'])]
piv = hp_bw_top.groupby(['bowler_name','batting_type']).agg(b=('is_legal','sum'), tva=('tva_bowl_opp',lambda x:100*x.mean())).reset_index()
print(piv.pivot(index='bowler_name', columns='batting_type', values=['b','tva']).round(1))

print("\n============================================================")
print("E. MONEYBALL / UNCAPPED & YOUNG VALUE RADAR (IPL-ONLY + 2024-26)")
print("============================================================")
# Compute IPL-only >= 60 HP balls to spot hidden gems (like Digvesh, Vipraj, Priyansh, Anshul, etc.)
b_ipl_hp = b_tf[(b_tf['match_tier']=='IPL') & (b_tf['hp_bat'])]
gem_b = b_ipl_hp.groupby(['striker_id','striker_name']).agg(hp_b=('is_batter_ball','sum'), hp_m=('match_id','nunique'), hp_sr=('batter_runs',lambda x:100*x.mean()), hp_tva=('tva_bat_opp',lambda x:100*x.mean()), hp_wpa=('wpa_bat',lambda x:100*x.mean())).reset_index()
gem_b = gem_b[(gem_b['hp_b']>=70) & (~gem_b['striker_id'].isin(top10_b['striker_id']))].sort_values('hp_tva', ascending=False).head(15)
print("Hidden Gem Batters (IPL HP >=70b, outside Top 10):")
print(gem_b.round(2).to_string(index=False))

bw_ipl_hp = tf[(tf['match_tier']=='IPL') & (tf['hp_bowl'])]
gem_bw = bw_ipl_hp.groupby(['bowler_id','bowler_name']).agg(hp_leg=('is_legal','sum'), hp_m=('match_id','nunique'), hp_w=('is_bowler_wkt','sum'), hp_econ=('bowler_runs',lambda x:6*x.sum()/max(1,bw_ipl_hp.loc[x.index,'is_legal'].sum())), hp_tva=('tva_bowl_opp',lambda x:100*x.mean()), hp_wpa=('wpa_bowl',lambda x:100*x.mean())).reset_index()
gem_bw = gem_bw[(gem_bw['hp_leg']>=70) & (~gem_bw['bowler_id'].isin(top10_bw['bowler_id']))].sort_values('hp_tva', ascending=False).head(15)
print("\nHidden Gem Bowlers (IPL HP >=70b, outside Top 10):")
print(gem_bw.round(2).to_string(index=False))

# Export Structured Deep-Dive Outputs to CSV for Fig 10, Report v3, & Dashboard
t_rows = []
for col, lbl in [('raw_sr','Overall Baseline (SR / -Econ)'),('hp_sr','Raw HP Rate (HP SR / -HP Econ)'),('naive_clutch','Naive Clutch Delta (HP - LP)'),('hp_tva','Unshrunk HP RAPM TVA'),('blended','Blended EB+RAPM HP TVA'),('prs_pred','Full RAPM PRS Model')]:
    rp = pearsonr(bt_eval[col], bt_eval['fut_hp_tva'])
    rs = spearmanr(bt_eval[col], bt_eval['fut_hp_tva'])
    t_rows.append({'role':'Batters (n=50)', 'metric': lbl, 'pearson_r': rp.statistic, 'p_val': rp.pvalue, 'spearman_rho': rs.statistic})
for col, lbl in [('neg_econ','Overall Baseline (SR / -Econ)'),('neg_hp_econ','Raw HP Rate (HP SR / -HP Econ)'),('naive_clutch','Naive Clutch Delta (HP - LP)'),('hp_tva','Unshrunk HP RAPM TVA'),('blended','Blended EB+RAPM HP TVA'),('prs_pred','Full RAPM PRS Model')]:
    rp = pearsonr(bw_eval[col], bw_eval['fut_hp_tva'])
    rs = spearmanr(bw_eval[col], bw_eval['fut_hp_tva'])
    t_rows.append({'role':'Bowlers (n=54)', 'metric': lbl, 'pearson_r': rp.statistic, 'p_val': rp.pvalue, 'spearman_rho': rs.statistic})
pd.DataFrame(t_rows).to_csv(f'{OUT}/temporal_prediction_benchmark.csv', index=False)

mech_rows = []
for lbl, ids in [('Top-10 Batters', set(top10_b['striker_id'])), ('Dip-10 Batters (>=500b)', set(dip10_b['striker_id']))]:
    sub = b_tf[b_tf['striker_id'].isin(ids)]
    for is_h, hnm in [(False,'LP'), (True,'HP')]:
        s = sub[sub['hp_bat']==is_h]
        mech_rows.append({
            'cohort': lbl, 'regime': hnm, 'balls': len(s),
            'rate': 100*s['batter_runs'].mean(), 'dot_pct': 100*(s['batter_runs']==0).mean(),
            'bdry_pct': 100*s['batter_runs'].isin([4,6]).mean(), 'wkt_pct': 100*s['is_striker_out'].mean(),
            'var_pct': 100*s['stroke'].isin(['Straight Drive','Pull shot','Cover Drive','On drive','Flick','Paddle Scoop','Slog Sweep']).mean(),
            'err_pct': 100*s['stroke'].isin(['Slog','Frontfoot defensive','Backfoot defensive','Push']).mean(),
            'tva100': 100*s['tva_bat_opp'].mean()
        })
for lbl, ids in [('Top-10 Bowlers', set(top10_bw['bowler_id'])), ('Dip-10 Bowlers (>=500b)', set(dip10_bw['bowler_id']))]:
    sub = tf[tf['bowler_id'].isin(ids)]
    for is_h, hnm in [(False,'LP'), (True,'HP')]:
        s = sub[sub['hp_bowl']==is_h]
        mech_rows.append({
            'cohort': lbl, 'regime': hnm, 'balls': len(s),
            'rate': 6*s['bowler_runs'].sum()/s['is_legal'].sum(), 'dot_pct': 100*(s['bowler_runs']==0).mean(),
            'bdry_pct': 100*s['runs'].isin([4,6]).mean(), 'wkt_pct': 100*s['is_bowler_wkt'].mean(),
            'var_pct': 100*s['delivery_style'].isin(['Yorker','Wide Yorker','Slow Bouncer','Off Cutter','Back of Hand','Carrom']).mean(),
            'err_pct': 100*s['delivery_style'].isin(['Full Toss','Straight']).mean(),
            'tva100': 100*s['tva_bowl_opp'].mean()
        })
pd.DataFrame(mech_rows).to_csv(f'{OUT}/clutch_mechanisms_summary.csv', index=False)

piv.to_csv(f'{OUT}/bowler_lhb_rhb_hp.csv', index=False)

v_rows = []
for vid, vnm in [(6,'Jaipur (Sawai Mansingh)'), (1241,'Guwahati (Barsapara)')]:
    vb = tf[(tf['venue_id']==vid) & (tf['match_tier']=='IPL')]
    g_v = vb.groupby(['hp_bowl','pace_cat']).agg(balls=('bowler_runs','size'), legal=('is_legal','sum'), runs=('bowler_runs','sum'), bdry=('runs', lambda x: 100*x.isin([4,6]).mean())).reset_index()
    g_v['venue'] = vnm; g_v['econ'] = 6*g_v['runs']/g_v['legal'].clip(lower=1)
    v_rows.append(g_v)
pd.concat(v_rows).to_csv(f'{OUT}/venue_jaipur_guwahati_hp.csv', index=False)

# Front-Office Trade Audit CSV (all disambiguated by exact IDs!)
trade_ids = [
    ('RR -> LSG: Sam Curran (OUT)', 4409, 'bowl'),
    ('LSG -> RR: Mohammed Shami (IN)', int(ebw[ebw['bowler_name']=='Mohammed Shami']['bowler_id'].iloc[0]), 'bowl'),
    ('LSG -> RR: M. Siddharth (IN)', 6237, 'bowl'),
    ('CSK -> SRH: Nathan Ellis (IN)', int(ebw[ebw['bowler_name']=='Nathan Ellis']['bowler_id'].iloc[0]), 'bowl'),
    ('CSK -> SRH: Khaleel Ahmed (IN)', int(ebw[ebw['bowler_name']=='Khaleel Ahmed']['bowler_id'].iloc[0]), 'bowl'),
    ('MI -> PBKS: Hardik Pandya (OUT)', int(ebw[ebw['bowler_name']=='Hardik Pandya']['bowler_id'].iloc[0]), 'bowl'),
    ('PBKS -> MI: Arshdeep Singh (IN)', int(ebw[ebw['bowler_name']=='Arshdeep Singh']['bowler_id'].iloc[0]), 'bowl'),
]
tr_audit = []
for lbl, pid, role in trade_ids:
    sub_el = ebw[ebw['bowler_id']==pid]
    if len(sub_el):
        r = sub_el.iloc[0]
        tr_audit.append({'move': lbl, 'player': r['bowler_name'], 'rank': int(r['rank']), 'prs': r['PRS_100'], 'hp_leg': int(r['hp_leg']), 'hp_econ': r['hp_econ'], 'hp_tva': r['blended_hp_tva'], 'raw_hp_tva': r['hp_tva100']})
    else:
        sub_raw = tf[tf['bowler_id']==pid]
        hp_s = sub_raw[sub_raw['hp_bowl']]
        tr_audit.append({'move': lbl, 'player': sub_raw['bowler_name'].iloc[0], 'rank': np.nan, 'prs': np.nan, 'hp_leg': int(hp_s['is_legal'].sum()), 'hp_econ': 6*hp_s['bowler_runs'].sum()/max(1,hp_s['is_legal'].sum()), 'hp_tva': np.nan, 'raw_hp_tva': 100*hp_s['tva_bowl_opp'].mean() if len(hp_s) else np.nan})
pd.DataFrame(tr_audit).to_csv(f'{OUT}/front_office_trade_audit.csv', index=False)

# Moneyball Uncapped / Value Radar export
gem_b.to_csv(f'{OUT}/moneyball_batters_radar.csv', index=False)
gem_bw.to_csv(f'{OUT}/moneyball_bowlers_radar.csv', index=False)

p11_rows = [
    {'factor':'IPL 2023-26 Batters: Starting XI', 'n': int((b_ipl23['is_impact']==0).sum()), 'tva100': 100*b_ipl23.loc[b_ipl23['is_impact']==0,'tva_bat_opp'].mean(), 'rate': 100*b_ipl23.loc[b_ipl23['is_impact']==0,'batter_runs'].mean()},
    {'factor':'IPL 2023-26 Batters: Impact Sub', 'n': int((b_ipl23['is_impact']==1).sum()), 'tva100': 100*b_ipl23.loc[b_ipl23['is_impact']==1,'tva_bat_opp'].mean(), 'rate': 100*b_ipl23.loc[b_ipl23['is_impact']==1,'batter_runs'].mean()},
    {'factor':'IPL 2023-26 Bowlers: Starting XI', 'n': int((bw_ipl23['is_impact']==0).sum()), 'tva100': 100*bw_ipl23.loc[bw_ipl23['is_impact']==0,'tva_bowl_opp'].mean(), 'rate': 6*bw_ipl23.loc[bw_ipl23['is_impact']==0,'bowler_runs'].mean()},
    {'factor':'IPL 2023-26 Bowlers: Impact Sub', 'n': int((bw_ipl23['is_impact']==1).sum()), 'tva100': 100*bw_ipl23.loc[bw_ipl23['is_impact']==1,'tva_bowl_opp'].mean(), 'rate': 6*bw_ipl23.loc[bw_ipl23['is_impact']==1,'bowler_runs'].mean()},
    {'factor':'All Top-Flight Batters: Captain', 'n': int((b_p[b_p['hp_bat']]['is_captain']==True).sum()), 'tva100': 100*b_p[b_p['hp_bat']].loc[b_p[b_p['hp_bat']]['is_captain']==True,'tva_bat_opp'].mean(), 'rate': 100*b_p[b_p['hp_bat']].loc[b_p[b_p['hp_bat']]['is_captain']==True,'batter_runs'].mean()},
    {'factor':'All Top-Flight Batters: Non-Captain', 'n': int((b_p[b_p['hp_bat']]['is_captain']==False).sum()), 'tva100': 100*b_p[b_p['hp_bat']].loc[b_p[b_p['hp_bat']]['is_captain']==False,'tva_bat_opp'].mean(), 'rate': 100*b_p[b_p['hp_bat']].loc[b_p[b_p['hp_bat']]['is_captain']==False,'batter_runs'].mean()},
]
pd.DataFrame(p11_rows).to_csv(f'{OUT}/p11_context_effects.csv', index=False)
print("\nExported all deep-dive CSVs to outputs/ cleanly!")
