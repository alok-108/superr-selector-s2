import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

BASE = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\data_pack\Data Pack'
OUT = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\outputs'

print("1. Loading dataset & preparing state features...")
comp = pd.read_csv(f'{BASE}/competitions.csv')
matches = pd.read_csv(f'{BASE}/matches.csv')
bbb = pd.read_csv(f'{BASE}/ball_by_ball.csv', low_memory=False)
ctpm = pd.read_csv(f'{BASE}/competition_team_player_mapping.csv')
p11 = pd.read_csv(f'{BASE}/playing_eleven.csv')

cm = ctpm.merge(comp, on=['comp_id', 'comp_name'])
ipl26_map = cm[cm['comp_name'] == 'IPL 2026'].groupby('player_id')['team_name'].first().to_dict()
ipl_last = cm[cm['comp_name'].str.startswith('IPL')].sort_values('season').groupby('player_id').agg(
    last_ipl_season=('season', 'last'), last_ipl_team=('team_name', 'last')
)
t20i_nat = cm[cm['match_format'] == 'T20I'].groupby('player_id')['team_name'].first().to_dict()

m = matches.merge(comp, on='comp_id', how='left')
full_members = {
    'India','Australia','England','South Africa','New Zealand','Pakistan',
    'West Indies','Sri Lanka','Bangladesh','Afghanistan','Ireland','Zimbabwe'
}
def classify_tier(row):
    if str(row['comp_name']).startswith('IPL'):
        return 'IPL'
    if 'Emerging' in str(row['comp_name']) or 'Rising' in str(row['comp_name']):
        return 'Emerging_A'
    if row['home_team'] in full_members and row['away_team'] in full_members:
        return 'T20I_FM_vs_FM'
    if row['home_team'] in full_members or row['away_team'] in full_members:
        return 'T20I_FM_vs_Assoc'
    return 'T20I_Assoc_vs_Assoc'

m['match_tier'] = m.apply(classify_tier, axis=1)
m['year'] = pd.to_datetime(m['match_date']).dt.year
m['is_knockout'] = ((m['knock_out'] == 1) | (m['stage'].isin(['Final','Qualifier','Eliminator','Semi Final']))).astype(int)

reg = bbb[bbb['innings'].isin([1, 2])].copy()
reg = reg.sort_values(['match_id', 'innings', 'overid', 'over_ball_num', 'innings_ball_num']).reset_index(drop=True)

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

# Classify Pace-On vs Pace-Off vs Spin
pace_off_styles = {'Off Cutter', 'Slow Bouncer', 'Leg Cut', 'Back of Hand', 'Knuckle', 'Slower Ball'}
def classify_bowler_delivery(df):
    is_spin = df['pace_or_spin'] == 'Spin'
    is_poff = (~is_spin) & (df['delivery_style'].isin(pace_off_styles) | ((df['bowl_speed'] > 40) & (df['bowl_speed'] < 123)))
    return np.where(is_spin, 'Spin', np.where(is_poff, 'Pace-Off', 'Pace-On'))

reg['pace_cat'] = classify_bowler_delivery(reg)

i1_totals = reg[reg['innings'] == 1].groupby('match_id')['tot_runs'].sum().rename('inn1_total')
m = m.merge(i1_totals, on='match_id', how='left')
m['max_balls_1'] = m['first_inn_revised_overs'].fillna(120).astype(int)
m['max_balls_2'] = m['second_inn_revised_overs'].fillna(120).astype(int)
m['target'] = np.where(m['revised_target'].notna(), m['revised_target'], m['inn1_total'] + 1)
dls_mid_cut = m['result'].str.contains('D/L|DLS', case=False, na=False) & m['revised_target'].isna()
m['has_valid_outcome'] = m['win_team_id'].notna() & (~dls_mid_cut)

reg = reg.merge(m[['match_id','comp_name','season','year','stage','is_knockout','venue_id','match_tier',
                   'max_balls_1','max_balls_2','target','inn1_total','win_team_id','has_valid_outcome','result',
                   'home_team','away_team','match_date']],
                on='match_id', how='left')

reg['max_balls'] = np.where(reg['innings'] == 1, reg['max_balls_1'], reg['max_balls_2'])
reg['balls_rem'] = (reg['max_balls'] - reg['legal_balls_before']).clip(lower=1)
reg['wickets_rem'] = (10 - reg['cum_wkts_before']).clip(lower=0)
reg['bat_won'] = (reg['batting_team_id'] == reg['win_team_id']).astype(float)
reg['phase'] = np.where(reg['overid'] <= 6, 'PP', np.where(reg['overid'] <= 15, 'Middle', 'Death'))

# Environment par RPO (LOO venue shrunk to tier-season)
match_rpo = reg.groupby(['match_id','match_tier','season','venue_id']).agg(
    m_runs=('tot_runs','sum'), m_balls=('is_legal','sum')
).reset_index()
ts_sum = match_rpo.groupby(['match_tier','season'])[['m_runs','m_balls']].transform('sum')
ts_loo_rpo = 6.0 * (ts_sum['m_runs'] - match_rpo['m_runs']) / (ts_sum['m_balls'] - match_rpo['m_balls']).clip(lower=1)
t_sum = match_rpo.groupby('match_tier')[['m_runs','m_balls']].transform('sum')
t_loo_rpo = 6.0 * (t_sum['m_runs'] - match_rpo['m_runs']) / (t_sum['m_balls'] - match_rpo['m_balls']).clip(lower=1)
ts_loo_rpo = ts_loo_rpo.fillna(t_loo_rpo)
v_sum = match_rpo.groupby('venue_id')[['m_runs','m_balls']].transform('sum')
match_rpo['env_par_rpo'] = 6.0 * ((v_sum['m_runs'] - match_rpo['m_runs']) + 1440.0 * (ts_loo_rpo / 6.0)) / ((v_sum['m_balls'] - match_rpo['m_balls']) + 1440.0)
reg = reg.merge(match_rpo[['match_id','env_par_rpo']], on='match_id', how='left')

# 2. Resource & WP Models (identical validated pipeline)
print("2. Computing Resource, WP, and Pressure Indices...")
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(reg, groups=reg['match_id']))
full_i1_all = reg[(reg['innings']==1)&(reg['max_balls']==120)&(reg['has_valid_outcome'])].copy()
full_i1_all['runs_to_go'] = full_i1_all['inn1_total'] - full_i1_all['cum_runs_before']
mdl_rem = HistGradientBoostingRegressor(max_iter=100, max_depth=4, min_samples_leaf=500, l2_regularization=20.0, monotonic_cst=[1,1,1], random_state=42)
mdl_rem.fit(full_i1_all[['balls_rem','wickets_rem','env_par_rpo']], full_i1_all['runs_to_go'])

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
    return np.column_stack([cs, cs / np.sqrt(np.maximum(1.0, br) + 6.0), np.log((rrr + 1.0)/(env_r + 1.0)),
                            rn, br, wr, env_r, prog, wr * prog])

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
cr_a = cum_r + reg['tot_runs'].values; lbb_a = np.minimum(mb, lbb + reg['is_legal'].astype(int).values)
br_a = np.maximum(1, mb - lbb_a); wr_a = np.maximum(0, wr - reg['is_wicket'].astype(int).values)
er_a = fast_rem_lookup(br_a, wr_a, env_r)
X1_a = build_i1_feats(cr_a, lbb_a, br_a, wr_a, mb, env_r, is_ipl_arr, er_a)
X2_a = build_i2_feats(cr_a, lbb_a, br_a, wr_a, mb, env_r, tgt, er_a)

wp_b = np.zeros(len(reg)); wp_a = np.zeros(len(reg))
wp_0 = np.zeros(len(reg)); wp_1 = np.zeros(len(reg)); wp_4 = np.zeros(len(reg)); wp_6 = np.zeros(len(reg)); wp_w = np.zeros(len(reg))

for tr_idx, va_idx in folds:
    tr_m = np.zeros(len(reg), dtype=bool); tr_m[tr_idx] = True
    va_m = np.zeros(len(reg), dtype=bool); va_m[va_idx] = True
    tr1 = tr_m & (reg['innings'].values==1) & reg['has_valid_outcome'].values
    tr2 = tr_m & (reg['innings'].values==2) & reg['has_valid_outcome'].values
    va1 = va_m & (reg['innings'].values==1); va2 = va_m & (reg['innings'].values==2)
    m1 = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=0.1))]).fit(X1_all[tr1], reg.loc[tr1, 'bat_won'])
    m2 = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=0.1))]).fit(X2_all[tr2], reg.loc[tr2, 'bat_won'])
    for arr, X in [(wp_b, X1_all), (wp_a, X1_a), (wp_0, X1_0), (wp_1, X1_1), (wp_4, X1_4), (wp_6, X1_6), (wp_w, X1_w)]: arr[va1] = m1.predict_proba(X[va1])[:, 1]
    for arr, X in [(wp_b, X2_all), (wp_a, X2_a), (wp_0, X2_0), (wp_1, X2_1), (wp_4, X2_4), (wp_6, X2_6), (wp_w, X2_w)]: arr[va2] = m2.predict_proba(X[va2])[:, 1]

i2_m = (reg['innings'].values == 2); last_b = (br == 1); all_out_w = (wr <= 1)
for arr, add_r, add_w in [(wp_0,0,0),(wp_1,1,0),(wp_4,4,0),(wp_6,6,0),(wp_w,0,1)]:
    won = i2_m & ((cum_r + add_r) >= tgt); lost = i2_m & (~won) & (last_b | ((add_w == 1) & all_out_w)); tie = i2_m & last_b & ((cum_r + add_r) == (tgt - 1))
    arr[won] = 1.0; arr[lost] = 0.0; arr[tie] = 0.5
br_rem_after = mb - lbb_a; won_a = i2_m & (cr_a >= tgt); lost_a = i2_m & (~won_a) & ((br_rem_after == 0) | (wr_a == 0)); tie_a = i2_m & (br_rem_after == 0) & (cr_a == (tgt - 1))
wp_a[won_a] = 1.0; wp_a[lost_a] = 0.0; wp_a[tie_a] = 0.5

reg['wp_before'] = wp_b; reg['wp_after'] = wp_a
reg['wpa_bat'] = wp_a - wp_b; reg['wpa_bowl'] = -reg['wpa_bat']

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

# 3. Context-Adjusted Baselines WITH Batting Order awareness for Bowling!
print("3. Fitting Context Baselines + Out-of-Fold RAPM Player-vs-Player Ridge Quality Model...")
reg['exp_bat_runs'] = np.nan; reg['exp_bat_out'] = np.nan
reg['exp_bowl_runs'] = np.nan; reg['exp_bowl_wkt'] = np.nan

X_ctx_bat = reg[['match_tier','pace_or_spin','innings','env_par_rpo','overid','wickets_rem']].copy()
X_ctx_bowl = reg[['match_tier','pace_or_spin','innings','env_par_rpo','overid','wickets_rem','batting_order']].copy()
for c in ['match_tier','pace_or_spin']:
    X_ctx_bat[c] = X_ctx_bat[c].astype('category')
    X_ctx_bowl[c] = X_ctx_bowl[c].astype('category')

for tr_idx, va_idx in folds:
    tr_b = tr_idx[reg.iloc[tr_idx]['is_batter_ball'].values]; va_b = va_idx[reg.iloc[va_idx]['is_batter_ball'].values]
    mr_bat = HistGradientBoostingRegressor(max_iter=80, max_depth=4, min_samples_leaf=400, categorical_features=['match_tier','pace_or_spin'], random_state=42).fit(X_ctx_bat.iloc[tr_b], reg.iloc[tr_b]['batter_runs'])
    reg.iloc[va_b, reg.columns.get_loc('exp_bat_runs')] = mr_bat.predict(X_ctx_bat.iloc[va_b])
    mw_bat = HistGradientBoostingClassifier(max_iter=80, max_depth=4, min_samples_leaf=400, categorical_features=['match_tier','pace_or_spin'], random_state=42).fit(X_ctx_bat.iloc[tr_b], reg.iloc[tr_b]['is_striker_out'].astype(int))
    reg.iloc[va_b, reg.columns.get_loc('exp_bat_out')] = mw_bat.predict_proba(X_ctx_bat.iloc[va_b])[:, 1]
    
    mr_bowl = HistGradientBoostingRegressor(max_iter=80, max_depth=4, min_samples_leaf=400, categorical_features=['match_tier','pace_or_spin'], random_state=42).fit(X_ctx_bowl.iloc[tr_idx], reg.iloc[tr_idx]['bowler_runs'])
    reg.iloc[va_idx, reg.columns.get_loc('exp_bowl_runs')] = mr_bowl.predict(X_ctx_bowl.iloc[va_idx])
    mw_bowl = HistGradientBoostingClassifier(max_iter=80, max_depth=4, min_samples_leaf=400, categorical_features=['match_tier','pace_or_spin'], random_state=42).fit(X_ctx_bowl.iloc[tr_idx], reg.iloc[tr_idx]['is_bowler_wkt'].astype(int))
    reg.iloc[va_idx, reg.columns.get_loc('exp_bowl_wkt')] = mw_bowl.predict_proba(X_ctx_bowl.iloc[va_idx])[:, 1]

reg['raa_bat'] = reg['batter_runs'] - reg['exp_bat_runs']
reg['waa_bat'] = reg['wicket_value'] * (reg['exp_bat_out'] - reg['is_striker_out'].astype(float))
reg['tva_bat'] = reg['raa_bat'] + reg['waa_bat']

reg['raa_bowl'] = reg['exp_bowl_runs'] - reg['bowler_runs']
reg['waa_bowl'] = reg['wicket_value'] * (reg['is_bowler_wkt'].astype(float) - reg['exp_bowl_wkt'])
reg['tva_bowl'] = reg['raa_bowl'] + reg['waa_bowl']

# OOF Hierarchical Player-vs-Player RAPM Ridge Model!
# Simultaneously solves: tva_bat ~ batter_effect(striker_id) + bowler_effect(bowler_id) + opp_team_effect!
# Using sparse OneHotEncoder + Ridge(alpha=1500) -> extracts pure OOF individual opponent bowler effect & individual batter effect!
ohe_bat = OneHotEncoder(handle_unknown='ignore')
ohe_bowl = OneHotEncoder(handle_unknown='ignore')
ohe_bteam = OneHotEncoder(handle_unknown='ignore')
ohe_bwteam = OneHotEncoder(handle_unknown='ignore')

S_all = ohe_bat.fit_transform(reg[['striker_id']])
B_all = ohe_bowl.fit_transform(reg[['bowler_id']])
BT_all = ohe_bteam.fit_transform(reg[['batting_team_id']])
BWT_all = ohe_bwteam.fit_transform(reg[['bowling_team_id']])

# Combined design matrix for Batter TVA: [Striker | Bowler | BowlingTeam]
X_rapm_bat = hstack([S_all, B_all, BWT_all]).tocsr()
n_s = S_all.shape[1]; n_b = B_all.shape[1]; n_bwt = BWT_all.shape[1]

# Combined design matrix for Bowler TVA: [Bowler | Striker | BattingTeam]
X_rapm_bowl = hstack([B_all, S_all, BT_all]).tocsr()

reg['rapm_opp_bat_adj'] = 0.0   # opponent bowler + bowling team effect subtracted from tva_bat
reg['rapm_opp_bowl_adj'] = 0.0  # opponent batter + batting team effect subtracted from tva_bowl

for tr_idx, va_idx in folds:
    tr_bat_idx = tr_idx[reg.iloc[tr_idx]['is_batter_ball'].values]
    va_bat_idx = va_idx[reg.iloc[va_idx]['is_batter_ball'].values]
    
    rd_bat = Ridge(alpha=1200.0, fit_intercept=True).fit(X_rapm_bat[tr_bat_idx], reg.iloc[tr_bat_idx]['tva_bat'])
    # Opponent effect on batting = Bowler coef + BowlingTeam coef (everything AFTER n_s!)
    opp_coef_bat = rd_bat.coef_[n_s:]
    reg.iloc[va_bat_idx, reg.columns.get_loc('rapm_opp_bat_adj')] = X_rapm_bat[va_bat_idx][:, n_s:].dot(opp_coef_bat)
    
    rd_bowl = Ridge(alpha=1200.0, fit_intercept=True).fit(X_rapm_bowl[tr_idx], reg.iloc[tr_idx]['tva_bowl'])
    # Opponent effect on bowling = Striker coef + BattingTeam coef (everything AFTER n_b!)
    opp_coef_bowl = rd_bowl.coef_[n_b:]
    reg.iloc[va_idx, reg.columns.get_loc('rapm_opp_bowl_adj')] = X_rapm_bowl[va_idx][:, n_b:].dot(opp_coef_bowl)

reg['tva_bat_rapm'] = reg['tva_bat'] - reg['rapm_opp_bat_adj']
reg['tva_bowl_rapm'] = reg['tva_bowl'] - reg['rapm_opp_bowl_adj']

print("RAPM opponent adjustment std (runs/100b): Bat =", round(100*reg['rapm_opp_bat_adj'].std(), 2),
      "| Bowl =", round(100*reg['rapm_opp_bowl_adj'].std(), 2))

# Inspect hardest & easiest bowlers / batters inferred by RAPM!
tf = reg[reg['match_tier'].isin(['IPL', 'T20I_FM_vs_FM'])].copy()
print("Top 10 toughest individual bowlers (most negative rapm_opp_bat_adj faced by batters):")
print((100*tf.groupby('bowler_name')['rapm_opp_bat_adj'].agg(['count','mean'])).query('count>=500').sort_values('mean').head(10).round(2).to_string())
print("Top 10 toughest individual batters (most negative rapm_opp_bowl_adj faced by bowlers):")
print((100*tf.groupby('striker_name')['rapm_opp_bowl_adj'].agg(['count','mean'])).query('count>=500').sort_values('mean').head(10).round(2).to_string())

# 4. Delivery Style & Stroke Execution Playbook Under High Pressure!
q75_bat = tf.loc[tf['is_batter_ball'], 'PI_bat'].quantile(0.75)
q75_bowl = tf['PI_bowl'].quantile(0.75)
tf_hp_bw = tf[tf['PI_bowl'] >= q75_bowl]
print("\n=== BOWLING DELIVERY STYLE PLAYBOOK UNDER HIGH PRESSURE (Death vs Non-Death, min>=150 balls) ===")
ds_death = tf_hp_bw[tf_hp_bw['phase']=='Death'].groupby(['pace_cat','delivery_style']).agg(
    balls=('bowler_runs','size'),
    econ=('bowler_runs', lambda x: 6*x.sum()/max(1, tf_hp_bw.loc[x.index,'is_legal'].sum())),
    wkt_pct=('is_bowler_wkt', lambda x: 100*x.mean()),
    dot_pct=('bowler_runs', lambda x: 100*(x==0).mean()),
    bdry_pct=('runs', lambda x: 100*x.isin([4,6]).mean()),
    tva100=('tva_bowl_rapm', lambda x: 100*x.mean())
).reset_index()
print(ds_death[ds_death['balls']>=150].sort_values('tva100', ascending=False).round(2).to_string(index=False))

tf_hp_b = tf[tf['is_batter_ball'] & (tf['PI_bat'] >= q75_bat)]
print("\n=== BATTING STROKE EXECUTION UNDER HIGH PRESSURE (min>=200 balls) ===")
st_hp = tf_hp_b.groupby('stroke').agg(
    balls=('batter_runs','size'),
    sr=('batter_runs', lambda x: 100*x.mean()),
    out_pct=('is_striker_out', lambda x: 100*x.mean()),
    bdry_pct=('batter_runs', lambda x: 100*x.isin([4,6]).mean()),
    tva100=('tva_bat_rapm', lambda x: 100*x.mean())
).reset_index()
print(st_hp[st_hp['balls']>=200].sort_values('tva100', ascending=False).round(2).to_string(index=False))

# 5. Compare RAPM-Adjusted Rankings vs Previous Rankings!
from sklearn.linear_model import LinearRegression

def score_bat_rapm(df, k=500.0, min_tot=400, min_hp=180, min_hp_m=15):
    b = df[df['is_batter_ball']].copy()
    cut = b['PI_bat'].quantile(0.75)
    b['is_hp'] = b['PI_bat'] >= cut
    b['w_pi'] = (b['PI_bat'] / 100.0) ** 2
    mean_w_hp = b.loc[b['is_hp'], 'w_pi'].mean()
    hp = b[b['is_hp']]; lp = b[~b['is_hp']]
    m_hp = hp.groupby(['striker_id','match_id'])['tva_bat_rapm'].sum().reset_index()
    
    def w_agg(gr):
        w = gr['w_pi'].values; ws = w.sum()
        return pd.Series({'w_n_eff': ws/mean_w_hp, 'w_tva100': 100*np.sum(w*gr['tva_bat_rapm'].values)/max(1e-9,ws), 'w_wpa100': 100*np.sum(w*gr['wpa_bat'].values)/max(1e-9,ws)})
    cw = b.groupby(['striker_id','striker_name']).apply(w_agg, include_groups=False).reset_index()
    tot = b.groupby('striker_id').agg(tot_b=('is_batter_ball','sum'), ipl_b=('match_tier', lambda x: (x=='IPL').sum())).reset_index()
    ahp = hp.groupby('striker_id').agg(
        hp_b=('is_batter_ball','sum'), hp_m=('match_id','nunique'), hp_sr=('batter_runs', lambda x: 100*x.mean()),
        hp_tva100=('tva_bat_rapm', lambda x: 100*x.mean()), hp_wpa100=('wpa_bat', lambda x: 100*x.mean()),
        entry10_tva=('tva_bat_rapm', lambda x: 100*x[hp.loc[x.index,'batter_balls_before']<10].mean()),
        entry10_b=('batter_balls_before', lambda x: (x<10).sum()),
    ).reset_index()
    alp = lp.groupby('striker_id').agg(lp_b=('is_batter_ball','sum'), lp_tva100=('tva_bat_rapm', lambda x: 100*x.mean())).reset_index()
    mc = m_hp.groupby('striker_id')['tva_bat_rapm'].agg(pos_m_rate=lambda x: (x>0).mean(), m_mean='mean', m_std='std').reset_index()
    r = cw.merge(tot, on='striker_id').merge(ahp, on='striker_id').merge(alp, on='striker_id').merge(mc, on='striker_id')
    r = r[(r['tot_b']>=min_tot)&(r['hp_b']>=min_hp)&(r['hp_m']>=min_hp_m)].copy()
    r['hp_tva_eb'] = (r['hp_b']/(r['hp_b']+k))*r['hp_tva100']
    r['cont_tva_eb'] = (r['w_n_eff']/(r['w_n_eff']+k))*r['w_tva100']
    r['blended_hp_tva'] = 0.5*r['hp_tva_eb'] + 0.5*r['cont_tva_eb']
    r['lp_tva_eb'] = (r['lp_b']/(r['lp_b']+k))*r['lp_tva100']
    r['hp_wpa_eb'] = 0.5*(r['hp_b']/(r['hp_b']+k))*r['hp_wpa100'] + 0.5*(r['w_n_eff']/(r['w_n_eff']+k))*r['w_wpa100']
    lr = LinearRegression().fit(r[['lp_tva_eb']], r['blended_hp_tva'])
    r['clutch_resid'] = r['blended_hp_tva'] - (lr.intercept_ + lr.coef_[0]*np.maximum(0.0, r['lp_tva_eb']))
    r['m_tstat'] = r['m_mean'] / (r['m_std'].fillna(10.0)/np.sqrt(r['hp_m']))
    def z(s): return (s - s.mean())/(s.std()+1e-9)
    r['PRS_z'] = 0.50*z(r['blended_hp_tva']) + 0.20*z(r['hp_wpa_eb']) + 0.15*z(r['clutch_resid']) + 0.15*(0.5*z(r['pos_m_rate']) + 0.5*z(r['m_tstat']))
    r['PRS_100'] = 50 + 15*r['PRS_z']
    r = r.sort_values('PRS_z', ascending=False).reset_index(drop=True)
    r['rank'] = np.arange(1, len(r)+1)
    return r

def score_bowl_rapm(df, k=500.0, min_tot=400, min_hp=180, min_hp_m=15):
    bw = df.copy()
    cut = bw['PI_bowl'].quantile(0.75)
    bw['is_hp'] = bw['PI_bowl'] >= cut
    bw['w_pi'] = (bw['PI_bowl'] / 100.0) ** 2
    mean_w_hp = bw.loc[bw['is_hp'], 'w_pi'].mean()
    hp = bw[bw['is_hp']]; lp = bw[~bw['is_hp']]
    m_hp = hp.groupby(['bowler_id','match_id'])['tva_bowl_rapm'].sum().reset_index()
    def w_agg(gr):
        w = gr['w_pi'].values; ws = w.sum()
        return pd.Series({'w_n_eff': ws/mean_w_hp, 'w_tva100': 100*np.sum(w*gr['tva_bowl_rapm'].values)/max(1e-9,ws), 'w_wpa100': 100*np.sum(w*gr['wpa_bowl'].values)/max(1e-9,ws)})
    cw = bw.groupby(['bowler_id','bowler_name']).apply(w_agg, include_groups=False).reset_index()
    tot = bw.groupby('bowler_id').agg(tot_leg=('is_legal','sum'), ipl_b=('match_tier', lambda x: (x=='IPL').sum())).reset_index()
    ahp = hp.groupby('bowler_id').agg(
        hp_deliv=('bowler_runs','size'), hp_leg=('is_legal','sum'), hp_m=('match_id','nunique'),
        hp_runs=('bowler_runs','sum'), hp_wkts=('is_bowler_wkt','sum'),
        hp_tva100=('tva_bowl_rapm', lambda x: 100*x.mean()), hp_wpa100=('wpa_bowl', lambda x: 100*x.mean())
    ).reset_index()
    ahp['hp_econ'] = 6*ahp['hp_runs']/ahp['hp_leg'].clip(lower=1)
    alp = lp.groupby('bowler_id').agg(lp_deliv=('bowler_runs','size'), lp_tva100=('tva_bowl_rapm', lambda x: 100*x.mean())).reset_index()
    mc = m_hp.groupby('bowler_id')['tva_bowl_rapm'].agg(pos_m_rate=lambda x: (x>0).mean(), m_mean='mean', m_std='std').reset_index()
    r = cw.merge(tot, on='bowler_id').merge(ahp, on='bowler_id').merge(alp, on='bowler_id').merge(mc, on='bowler_id')
    r = r[(r['tot_leg']>=min_tot)&(r['hp_leg']>=min_hp)&(r['hp_m']>=min_hp_m)].copy()
    r['hp_tva_eb'] = (r['hp_deliv']/(r['hp_deliv']+k))*r['hp_tva100']
    r['cont_tva_eb'] = (r['w_n_eff']/(r['w_n_eff']+k))*r['w_tva100']
    r['blended_hp_tva'] = 0.5*r['hp_tva_eb'] + 0.5*r['cont_tva_eb']
    r['lp_tva_eb'] = (r['lp_deliv']/(r['lp_deliv']+k))*r['lp_tva100']
    r['hp_wpa_eb'] = 0.5*(r['hp_deliv']/(r['hp_deliv']+k))*r['hp_wpa100'] + 0.5*(r['w_n_eff']/(r['w_n_eff']+k))*r['w_wpa100']
    lr = LinearRegression().fit(r[['lp_tva_eb']], r['blended_hp_tva'])
    r['clutch_resid'] = r['blended_hp_tva'] - (lr.intercept_ + lr.coef_[0]*np.maximum(0.0, r['lp_tva_eb']))
    r['m_tstat'] = r['m_mean'] / (r['m_std'].fillna(10.0)/np.sqrt(r['hp_m']))
    def z(s): return (s - s.mean())/(s.std()+1e-9)
    r['PRS_z'] = 0.50*z(r['blended_hp_tva']) + 0.20*z(r['hp_wpa_eb']) + 0.15*z(r['clutch_resid']) + 0.15*(0.5*z(r['pos_m_rate']) + 0.5*z(r['m_tstat']))
    r['PRS_100'] = 50 + 15*r['PRS_z']
    r = r.sort_values('PRS_z', ascending=False).reset_index(drop=True)
    r['rank'] = np.arange(1, len(r)+1)
    return r

rb_rapm = score_bat_rapm(tf)
rbw_rapm = score_bowl_rapm(tf)
print("\n=== RAPM PLAYER-VS-PLAYER + BATTING-ORDER ADJUSTED TOP 15 BATTERS ===")
print(rb_rapm[['rank','striker_name','hp_b','hp_m','hp_sr','blended_hp_tva','lp_tva_eb','clutch_resid','entry10_b','entry10_tva','PRS_100']].head(15).round(2).to_string(index=False))

print("\n=== RAPM PLAYER-VS-PLAYER + BATTING-ORDER ADJUSTED TOP 15 BOWLERS ===")
print(rbw_rapm[['rank','bowler_name','hp_leg','hp_m','hp_econ','hp_wkts','blended_hp_tva','lp_tva_eb','clutch_resid','PRS_100']].head(15).round(2).to_string(index=False))

# 6. RR IPL 2026 Match-by-Match Wins vs Losses Pressure Autopsy!
rr_tid = 3
rr26 = reg[(reg['comp_name']=='IPL 2026') & ((reg['batting_team_id']==rr_tid) | (reg['bowling_team_id']==rr_tid))].copy()
rr26['rr_role'] = np.where(rr26['batting_team_id']==rr_tid, 'RR_Bat', 'RR_Bowl')
rr26['rr_won'] = (rr26['win_team_id']==rr_tid)
rr26['rr_tva'] = np.where(rr26['rr_role']=='RR_Bat', rr26['tva_bat_rapm'], rr26['tva_bowl_rapm'])
rr26['rr_wpa'] = np.where(rr26['rr_role']=='RR_Bat', rr26['wpa_bat'], rr26['wpa_bowl'])
rr26['is_hp'] = np.where(rr26['rr_role']=='RR_Bat', rr26['PI_bat']>=q75_bat, rr26['PI_bowl']>=q75_bowl)
print("\n=== RR IPL 2026 WINS vs LOSSES BY ROLE & PRESSURE ===")
print(rr26.groupby(['rr_won','rr_role','is_hp']).agg(balls=('rr_tva','size'), tva_sum=('rr_tva','sum'), tva_100=('rr_tva', lambda x: 100*x.mean()), wpa_sum=('rr_wpa','sum')).round(2).to_string())
