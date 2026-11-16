import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, mean_absolute_error, r2_score

BASE = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\data_pack\Data Pack'
OUT = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\outputs'
os.makedirs(OUT, exist_ok=True)

print("1. Loading raw Data Pack CSVs...")
comp = pd.read_csv(f'{BASE}/competitions.csv')
matches = pd.read_csv(f'{BASE}/matches.csv')
bbb = pd.read_csv(f'{BASE}/ball_by_ball.csv', low_memory=False)
teams = pd.read_csv(f'{BASE}/teams.csv')
ctpm = pd.read_csv(f'{BASE}/competition_team_player_mapping.csv')
p11 = pd.read_csv(f'{BASE}/playing_eleven.csv')

# Mappings
cm = ctpm.merge(comp, on=['comp_id', 'comp_name'])
ipl26_map = cm[cm['comp_name'] == 'IPL 2026'].groupby('player_id')['team_name'].first().to_dict()
ipl_last = cm[cm['comp_name'].str.startswith('IPL')].sort_values('season').groupby('player_id').agg(
    last_ipl_season=('season', 'last'), last_ipl_team=('team_name', 'last')
)
t20i_nat = cm[cm['match_format'] == 'T20I'].groupby('player_id')['team_name'].first().to_dict()

# Match tier classification
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

# 2. Clean regular innings 1 & 2 deliveries
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
reg['crr'] = np.where(reg['legal_balls_before'] > 0, 6.0 * reg['cum_runs_before'] / reg['legal_balls_before'], 0.0)
reg['bat_won'] = (reg['batting_team_id'] == reg['win_team_id']).astype(float)
reg['phase'] = np.where(reg['overid'] <= 6, 'PP', np.where(reg['overid'] <= 15, 'Middle', 'Death'))

# Shrunk LOO environment par RPO
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

# Save tier x season summary
tier_season = reg.groupby(['match_tier','season']).agg(
    matches=('match_id','nunique'),
    legal_balls=('is_legal','sum'),
    runs=('tot_runs','sum'),
    wickets=('is_wicket','sum')
).reset_index()
tier_season['rpo'] = 6.0 * tier_season['runs'] / tier_season['legal_balls']
tier_season['bpw'] = tier_season['legal_balls'] / tier_season['wickets']
tier_season.to_csv(f'{OUT}/tier_season_summary.csv', index=False)

# 3. Resource Model & Win Probability Models (with full validation logging)
print("2. Fitting Resource & Out-of-Fold Win Probability Models...")
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(reg, groups=reg['match_id']))
res_feats = ['balls_rem', 'wickets_rem', 'env_par_rpo']

# Chronological validation for Resource Model (<=2025 -> 2026)
full_i1_tr = reg[(reg['innings']==1)&(reg['max_balls']==120)&(reg['has_valid_outcome'])&(reg['year']<=2025)].copy()
full_i1_te = reg[(reg['innings']==1)&(reg['max_balls']==120)&(reg['has_valid_outcome'])&(reg['year']==2026)].copy()
full_i1_tr['runs_to_go'] = full_i1_tr['inn1_total'] - full_i1_tr['cum_runs_before']
full_i1_te['runs_to_go'] = full_i1_te['inn1_total'] - full_i1_te['cum_runs_before']

mdl_rem_tr = HistGradientBoostingRegressor(max_iter=100, max_depth=4, min_samples_leaf=500, l2_regularization=20.0, monotonic_cst=[1,1,1], random_state=42)
mdl_rem_tr.fit(full_i1_tr[res_feats], full_i1_tr['runs_to_go'])
pred_rem_te = np.maximum(0.0, mdl_rem_tr.predict(full_i1_te[res_feats]))
res_mae_26 = float(mean_absolute_error(full_i1_te['runs_to_go'], pred_rem_te))
res_r2_26 = float(r2_score(full_i1_te['runs_to_go'], pred_rem_te))

# Full resource model for fast monotonic lookup grid
full_i1_all = reg[(reg['innings']==1)&(reg['max_balls']==120)&(reg['has_valid_outcome'])].copy()
full_i1_all['runs_to_go'] = full_i1_all['inn1_total'] - full_i1_all['cum_runs_before']
mdl_rem = HistGradientBoostingRegressor(max_iter=100, max_depth=4, min_samples_leaf=500, l2_regularization=20.0, monotonic_cst=[1,1,1], random_state=42)
mdl_rem.fit(full_i1_all[res_feats], full_i1_all['runs_to_go'])

b_grid = np.arange(1, 121)
w_grid = np.arange(0, 11)
r_grid = np.round(np.arange(5.0, 13.0, 0.25), 2)
bb, ww, rr = np.meshgrid(b_grid, w_grid, r_grid, indexing='ij')
grid_df = pd.DataFrame({'balls_rem': bb.ravel(), 'wickets_rem': ww.ravel(), 'env_par_rpo': rr.ravel()})
grid_pred = mdl_rem.predict(grid_df)
grid_pred = np.where(grid_df['wickets_rem'].values == 0, 0.0, np.maximum(0.0, grid_pred))
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
    par_tot = env_r * (mb / 6.0)
    proj_tot = cum_r + exp_rem
    prog = lbb / mb
    ps = proj_tot - par_tot
    return np.column_stack([ps, ps * (0.3 + 0.7 * prog), cum_r, wr, br, prog, env_r, is_ipl])

def build_i2_feats(cum_r, lbb, br, wr, mb, env_r, tgt, exp_rem):
    rn = np.maximum(0.0, tgt - cum_r)
    rrr = 6.0 * rn / np.maximum(1.0, br)
    cs = exp_rem - rn
    prog = lbb / mb
    return np.column_stack([cs, cs / np.sqrt(np.maximum(1.0, br) + 6.0), np.log((rrr + 1.0)/(env_r + 1.0)),
                            rn, br, wr, env_r, prog, wr * prog])

is_ipl_arr = (reg['match_tier'] == 'IPL').astype(float).values
cum_r = reg['cum_runs_before'].values
lbb = reg['legal_balls_before'].values
br = reg['balls_rem'].values
wr = reg['wickets_rem'].values
mb = reg['max_balls'].values
env_r = reg['env_par_rpo'].values
tgt = reg['target'].fillna(160).values
exp_r = reg['exp_rem_runs'].values

X1_all = build_i1_feats(cum_r, lbb, br, wr, mb, env_r, is_ipl_arr, exp_r)
X2_all = build_i2_feats(cum_r, lbb, br, wr, mb, env_r, tgt, exp_r)

def make_cf(add_r, add_w):
    cr_c = cum_r + add_r
    lbb_c = np.minimum(mb, lbb + 1)
    br_c = np.maximum(1, br - 1)
    wr_c = np.maximum(0, wr - add_w)
    er_c = fast_rem_lookup(br_c, wr_c, env_r)
    return (build_i1_feats(cr_c, lbb_c, br_c, wr_c, mb, env_r, is_ipl_arr, er_c),
            build_i2_feats(cr_c, lbb_c, br_c, wr_c, mb, env_r, tgt, er_c))

X1_0, X2_0 = make_cf(0, 0)
X1_1, X2_1 = make_cf(1, 0)
X1_4, X2_4 = make_cf(4, 0)
X1_6, X2_6 = make_cf(6, 0)
X1_w, X2_w = make_cf(0, 1)

cr_a = cum_r + reg['tot_runs'].values
lbb_a = np.minimum(mb, lbb + reg['is_legal'].astype(int).values)
br_a = np.maximum(1, mb - lbb_a)
wr_a = np.maximum(0, wr - reg['is_wicket'].astype(int).values)
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
    va1 = va_m & (reg['innings'].values==1)
    va2 = va_m & (reg['innings'].values==2)
    m1 = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=0.1))])
    m1.fit(X1_all[tr1], reg.loc[tr1, 'bat_won'])
    m2 = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=0.1))])
    m2.fit(X2_all[tr2], reg.loc[tr2, 'bat_won'])
    for arr, X in [(wp_b, X1_all), (wp_a, X1_a), (wp_0, X1_0), (wp_1, X1_1), (wp_4, X1_4), (wp_6, X1_6), (wp_w, X1_w)]:
        arr[va1] = m1.predict_proba(X[va1])[:, 1]
    for arr, X in [(wp_b, X2_all), (wp_a, X2_a), (wp_0, X2_0), (wp_1, X2_1), (wp_4, X2_4), (wp_6, X2_6), (wp_w, X2_w)]:
        arr[va2] = m2.predict_proba(X[va2])[:, 1]

i2_m = (reg['innings'].values == 2)
last_b = (br == 1)
all_out_w = (wr <= 1)
for arr, add_r, add_w in [(wp_0,0,0),(wp_1,1,0),(wp_4,4,0),(wp_6,6,0),(wp_w,0,1)]:
    won = i2_m & ((cum_r + add_r) >= tgt)
    lost = i2_m & (~won) & (last_b | ((add_w == 1) & all_out_w))
    tie = i2_m & last_b & ((cum_r + add_r) == (tgt - 1))
    arr[won] = 1.0; arr[lost] = 0.0; arr[tie] = 0.5

br_rem_after = mb - lbb_a
won_a = i2_m & (cr_a >= tgt)
lost_a = i2_m & (~won_a) & ((br_rem_after == 0) | (wr_a == 0))
tie_a = i2_m & (br_rem_after == 0) & (cr_a == (tgt - 1))
wp_a[won_a] = 1.0; wp_a[lost_a] = 0.0; wp_a[tie_a] = 0.5

reg['wp_before'] = wp_b
reg['wp_after'] = wp_a
reg['wpa_bat'] = wp_a - wp_b
reg['wpa_bowl'] = -reg['wpa_bat']

# Save WP Calibration Metrics + Reliability Curve bins
val_rows = []
for inn_num in [1, 2]:
    sub = reg[(reg['innings']==inn_num) & reg['has_valid_outcome']]
    val_rows.append({
        'model': f'Innings {inn_num} WP (5-Fold Match-Grouped OOF)',
        'n_balls': len(sub),
        'n_matches': sub['match_id'].nunique(),
        'log_loss': log_loss(sub['bat_won'], sub['wp_before']),
        'brier_score': brier_score_loss(sub['bat_won'], sub['wp_before']),
        'roc_auc': roc_auc_score(sub['bat_won'], sub['wp_before']),
    })
pd.DataFrame(val_rows).to_csv(f'{OUT}/wp_validation_metrics.csv', index=False)

cal_sub = reg[(reg['innings']==2) & reg['has_valid_outcome']].copy()
cal_sub['wp_bin'] = pd.cut(cal_sub['wp_before'], np.linspace(0, 1, 11))
cal_df = cal_sub.groupby('wp_bin', observed=False).agg(
    n=('bat_won','size'), mean_pred=('wp_before','mean'), actual_win=('bat_won','mean')
).reset_index()
cal_df['wp_bin'] = cal_df['wp_bin'].astype(str)
cal_df.to_csv(f'{OUT}/wp_calibration_bins.csv', index=False)

# 4. Pressure Index Computation
p_dot = ((reg['tot_runs']==0) & (~reg['is_wicket'])).groupby(reg['phase']).mean()
p_123 = ((reg['tot_runs'].isin([1,2,3])) & (~reg['is_wicket'])).groupby(reg['phase']).mean()
p_4 = ((reg['runs']==4) & (~reg['is_wicket'])).groupby(reg['phase']).mean()
p_6 = ((reg['runs']==6) & (~reg['is_wicket'])).groupby(reg['phase']).mean()
p_w = reg['is_wicket'].groupby(reg['phase']).mean()
w0, w1, w4, w6, ww = [reg['phase'].map(s).values for s in [p_dot, p_123, p_4, p_6, p_w]]
ws = w0 + w1 + w4 + w6 + ww
w0, w1, w4, w6, ww = w0/ws, w1/ws, w4/ws, w6/ws, ww/ws
mu_cf = w0*wp_0 + w1*wp_1 + w4*wp_4 + w6*wp_6 + ww*wp_w
reg['wp_lev'] = np.sqrt(w0*(wp_0-mu_cf)**2 + w1*(wp_1-mu_cf)**2 + w4*(wp_4-mu_cf)**2 + w6*(wp_6-mu_cf)**2 + ww*(wp_w-mu_cf)**2)

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

# Save situation validity table
situations = {
    'I2 Death Crunch (Ov 17-20, 0.2<=WP<=0.8)': (reg['innings']==2)&(reg['overid']>=17)&(reg['wp_before'].between(0.2,0.8)),
    'I1 Death Competitive (Ov 17-20, 0.25<=WP<=0.75)': (reg['innings']==1)&(reg['overid']>=17)&(reg['wp_before'].between(0.25,0.75)),
    'Early Collapse (Ov 1-8, Wkts>=3, WP>=0.08)': (reg['overid']<=8)&(reg['cum_wkts_before']>=3)&(reg['wp_before']>=0.08),
    'Mid-Innings Rebuild (Ov 7-14, Wkts>=4, WP>=0.08)': (reg['overid'].between(7,14))&(reg['cum_wkts_before']>=4)&(reg['wp_before']>=0.08),
    'Knockout / Playoff Matches (All Balls)': (reg['is_knockout']==1),
    'Comfortable Powerplay (Ov 1-6, Wkts<=1, I1)': (reg['innings']==1)&(reg['overid']<=6)&(reg['cum_wkts_before']<=1),
    'Routine Middle Overs (Ov 7-15, Wkts<=2, I1)': (reg['innings']==1)&(reg['overid'].between(7,15))&(reg['cum_wkts_before']<=2),
    'I2 Cruise Chase (Ov 17-20, WP>0.95)': (reg['innings']==2)&(reg['overid']>=17)&(reg['wp_before']>0.95),
    'I2 Dead Chase (Ov 17-20, WP<0.03)': (reg['innings']==2)&(reg['overid']>=17)&(reg['wp_before']<0.03),
}
sit_rows = []
for lbl, msk in situations.items():
    sub = reg[msk]
    sit_rows.append({'situation': lbl, 'deliveries': len(sub), 'mean_PI_bat': sub['PI_bat'].mean(), 'mean_PI_bowl': sub['PI_bowl'].mean()})
pd.DataFrame(sit_rows).to_csv(f'{OUT}/situation_validation.csv', index=False)

# 5. Context-Adjusted xR & xW Baselines (OOF)
print("3. Fitting Context-Adjusted xR & xW Baselines...")
reg['exp_bat_runs'] = np.nan
reg['exp_bat_out'] = np.nan
reg['exp_bowl_runs'] = np.nan
reg['exp_bowl_wkt'] = np.nan

X_ctx = reg[['match_tier','pace_or_spin','innings','env_par_rpo','overid','wickets_rem']].copy()
for c in ['match_tier','pace_or_spin']:
    X_ctx[c] = X_ctx[c].astype('category')

for tr_idx, va_idx in folds:
    tr_b = tr_idx[reg.iloc[tr_idx]['is_batter_ball'].values]
    va_b = va_idx[reg.iloc[va_idx]['is_batter_ball'].values]
    mr_bat = HistGradientBoostingRegressor(max_iter=80, max_depth=4, min_samples_leaf=400, categorical_features=['match_tier','pace_or_spin'], random_state=42)
    mr_bat.fit(X_ctx.iloc[tr_b], reg.iloc[tr_b]['batter_runs'])
    reg.iloc[va_b, reg.columns.get_loc('exp_bat_runs')] = mr_bat.predict(X_ctx.iloc[va_b])
    mw_bat = HistGradientBoostingClassifier(max_iter=80, max_depth=4, min_samples_leaf=400, categorical_features=['match_tier','pace_or_spin'], random_state=42)
    mw_bat.fit(X_ctx.iloc[tr_b], reg.iloc[tr_b]['is_striker_out'].astype(int))
    reg.iloc[va_b, reg.columns.get_loc('exp_bat_out')] = mw_bat.predict_proba(X_ctx.iloc[va_b])[:, 1]
    
    mr_bowl = HistGradientBoostingRegressor(max_iter=80, max_depth=4, min_samples_leaf=400, categorical_features=['match_tier','pace_or_spin'], random_state=42)
    mr_bowl.fit(X_ctx.iloc[tr_idx], reg.iloc[tr_idx]['bowler_runs'])
    reg.iloc[va_idx, reg.columns.get_loc('exp_bowl_runs')] = mr_bowl.predict(X_ctx.iloc[va_idx])
    mw_bowl = HistGradientBoostingClassifier(max_iter=80, max_depth=4, min_samples_leaf=400, categorical_features=['match_tier','pace_or_spin'], random_state=42)
    mw_bowl.fit(X_ctx.iloc[tr_idx], reg.iloc[tr_idx]['is_bowler_wkt'].astype(int))
    reg.iloc[va_idx, reg.columns.get_loc('exp_bowl_wkt')] = mw_bowl.predict_proba(X_ctx.iloc[va_idx])[:, 1]

reg['raa_bat'] = reg['batter_runs'] - reg['exp_bat_runs']
reg['waa_bat'] = reg['wicket_value'] * (reg['exp_bat_out'] - reg['is_striker_out'].astype(float))
reg['tva_bat'] = reg['raa_bat'] + reg['waa_bat']

reg['raa_bowl'] = reg['exp_bowl_runs'] - reg['bowler_runs']
reg['waa_bowl'] = reg['wicket_value'] * (reg['is_bowler_wkt'].astype(float) - reg['exp_bowl_wkt'])
reg['tva_bowl'] = reg['raa_bowl'] + reg['waa_bowl']

# OOF Opponent strength adjustment
reg['opp_bowl_adj'] = 0.0
reg['opp_bat_adj'] = 0.0
for tr_i, va_i in folds:
    tr = reg.iloc[tr_i]; va = reg.iloc[va_i]
    opp_bw = tr[tr['is_batter_ball']].groupby('bowling_team_id')['tva_bat'].agg(['sum','count'])
    reg.iloc[va_i, reg.columns.get_loc('opp_bowl_adj')] = va['bowling_team_id'].map(opp_bw['sum'] / (opp_bw['count'] + 2000.0)).fillna(0.0).values
    opp_bt = tr.groupby('batting_team_id')['tva_bowl'].agg(['sum','count'])
    reg.iloc[va_i, reg.columns.get_loc('opp_bat_adj')] = va['batting_team_id'].map(opp_bt['sum'] / (opp_bt['count'] + 2000.0)).fillna(0.0).values

reg['tva_bat_opp'] = reg['tva_bat'] - reg['opp_bowl_adj']
reg['tva_bowl_opp'] = reg['tva_bowl'] - reg['opp_bat_adj']

# 6. Top-Flight Player Scoring (IPL + Full-Member T20Is)
print("4. Computing Final Batters & Bowlers Rankings, Sensitivity & Bootstrap...")
tf = reg[reg['match_tier'].isin(['IPL', 'T20I_FM_vs_FM'])].copy()
b_tf = tf[tf['is_batter_ball']].copy()
bw_tf = tf.copy()

def compute_batters(df, q_thresh=0.75, k=500.0, min_tot=400, min_hp=180, min_hp_m=15):
    b = df.copy()
    cut = b['PI_bat'].quantile(q_thresh)
    b['is_hp'] = b['PI_bat'] >= cut
    b['w_pi'] = (b['PI_bat'] / 100.0) ** 2
    mean_w_hp = b.loc[b['is_hp'], 'w_pi'].mean()
    
    hp = b[b['is_hp']]; lp = b[~b['is_hp']]
    m_hp = hp.groupby(['striker_id','match_id'])['tva_bat_opp'].sum().reset_index()
    
    def w_agg(gr):
        w = gr['w_pi'].values
        ws = w.sum()
        return pd.Series({
            'w_n_eff': ws / mean_w_hp,
            'w_tva100': 100.0 * np.sum(w * gr['tva_bat_opp'].values) / max(1e-9, ws),
            'w_wpa100': 100.0 * np.sum(w * gr['wpa_bat'].values) / max(1e-9, ws),
        })
    cw = b.groupby(['striker_id','striker_name']).apply(w_agg, include_groups=False).reset_index()
    tot = b.groupby('striker_id').agg(
        tot_b=('is_batter_ball','sum'), tot_runs=('batter_runs','sum'), tot_m=('match_id','nunique'),
        ipl_b=('match_tier', lambda x: (x=='IPL').sum()),
        t20i_b=('match_tier', lambda x: (x=='T20I_FM_vs_FM').sum()),
        bat_type=('batting_type','first'),
        med_order=('batting_order','median'),
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
    
    lr = LinearRegression().fit(r[['lp_tva_eb']], r['blended_hp_tva'])
    r['exp_hp_from_lp'] = lr.intercept_ + lr.coef_[0] * np.maximum(0.0, r['lp_tva_eb'])
    r['clutch_resid'] = r['blended_hp_tva'] - r['exp_hp_from_lp']
    r['m_tstat'] = r['m_mean'] / (r['m_std'].fillna(10.0) / np.sqrt(r['hp_m']))
    
    def z(s): return (s - s.mean()) / (s.std() + 1e-9)
    r['PRS_z'] = 0.50*z(r['blended_hp_tva']) + 0.20*z(r['hp_wpa_eb']) + 0.15*z(r['clutch_resid']) + 0.15*(0.5*z(r['pos_m_rate']) + 0.5*z(r['m_tstat']))
    r['PRS_100'] = 50.0 + 15.0 * r['PRS_z']
    r = r.sort_values('PRS_z', ascending=False).reset_index(drop=True)
    r['rank'] = np.arange(1, len(r)+1)
    r['ipl_2026_team'] = r['striker_id'].map(ipl26_map).fillna('Non-IPL26')
    r['last_ipl_season'] = r['striker_id'].map(ipl_last['last_ipl_season'])
    r['last_ipl_team'] = r['striker_id'].map(ipl_last['last_ipl_team']).fillna('None')
    r['t20i_team'] = r['striker_id'].map(t20i_nat).fillna('Unmapped')
    return r, cut, lr.coef_[0], lr.intercept_

def compute_bowlers(df, q_thresh=0.75, k=500.0, min_tot=400, min_hp=180, min_hp_m=15):
    bw = df.copy()
    cut = bw['PI_bowl'].quantile(q_thresh)
    bw['is_hp'] = bw['PI_bowl'] >= cut
    bw['w_pi'] = (bw['PI_bowl'] / 100.0) ** 2
    mean_w_hp = bw.loc[bw['is_hp'], 'w_pi'].mean()
    
    hp = bw[bw['is_hp']]; lp = bw[~bw['is_hp']]
    m_hp = hp.groupby(['bowler_id','match_id'])['tva_bowl_opp'].sum().reset_index()
    
    def w_agg(gr):
        w = gr['w_pi'].values
        ws = w.sum()
        return pd.Series({
            'w_n_eff': ws / mean_w_hp,
            'w_tva100': 100.0 * np.sum(w * gr['tva_bowl_opp'].values) / max(1e-9, ws),
            'w_wpa100': 100.0 * np.sum(w * gr['wpa_bowl'].values) / max(1e-9, ws),
        })
    cw = bw.groupby(['bowler_id','bowler_name']).apply(w_agg, include_groups=False).reset_index()
    tot = bw.groupby('bowler_id').agg(
        tot_leg=('is_legal','sum'), tot_wkts=('is_bowler_wkt','sum'), tot_m=('match_id','nunique'),
        ipl_b=('match_tier', lambda x: (x=='IPL').sum()),
        t20i_b=('match_tier', lambda x: (x=='T20I_FM_vs_FM').sum()),
        bowl_style=('bowling_type','first'),
        pace_spin=('pace_or_spin','first'),
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
        lp_bdry=('runs', lambda x: 100*x.isin([4,6]).mean()),
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
    r['exp_hp_from_lp'] = lr.intercept_ + lr.coef_[0] * np.maximum(0.0, r['lp_tva_eb'])
    r['clutch_resid'] = r['blended_hp_tva'] - r['exp_hp_from_lp']
    r['m_tstat'] = r['m_mean'] / (r['m_std'].fillna(10.0) / np.sqrt(r['hp_m']))
    
    def z(s): return (s - s.mean()) / (s.std() + 1e-9)
    r['PRS_z'] = 0.50*z(r['blended_hp_tva']) + 0.20*z(r['hp_wpa_eb']) + 0.15*z(r['clutch_resid']) + 0.15*(0.5*z(r['pos_m_rate']) + 0.5*z(r['m_tstat']))
    r['PRS_100'] = 50.0 + 15.0 * r['PRS_z']
    r = r.sort_values('PRS_z', ascending=False).reset_index(drop=True)
    r['rank'] = np.arange(1, len(r)+1)
    r['ipl_2026_team'] = r['bowler_id'].map(ipl26_map).fillna('Non-IPL26')
    r['last_ipl_season'] = r['bowler_id'].map(ipl_last['last_ipl_season'])
    r['last_ipl_team'] = r['bowler_id'].map(ipl_last['last_ipl_team']).fillna('None')
    r['t20i_team'] = r['bowler_id'].map(t20i_nat).fillna('Unmapped')
    return r, cut, lr.coef_[0], lr.intercept_

rb, cut_b, beta_b, int_b = compute_batters(b_tf)
rbw, cut_bw, beta_bw, int_bw = compute_bowlers(bw_tf)

# Sensitivity specs
rb_q70, _, _, _ = compute_batters(b_tf, q_thresh=0.70)
rb_q80, _, _, _ = compute_batters(b_tf, q_thresh=0.80, min_hp=140, min_hp_m=12)
rb_k300, _, _, _ = compute_batters(b_tf, k=300.0)
rb_k800, _, _, _ = compute_batters(b_tf, k=800.0)
rb_ipl, _, _, _ = compute_batters(b_tf[b_tf['match_tier']=='IPL'], min_tot=250, min_hp=100, min_hp_m=10)

rbw_q70, _, _, _ = compute_bowlers(bw_tf, q_thresh=0.70)
rbw_q80, _, _, _ = compute_bowlers(bw_tf, q_thresh=0.80, min_hp=140, min_hp_m=12)
rbw_k300, _, _, _ = compute_bowlers(bw_tf, k=300.0)
rbw_k800, _, _, _ = compute_bowlers(bw_tf, k=800.0)
rbw_ipl, _, _, _ = compute_bowlers(bw_tf[bw_tf['match_tier']=='IPL'], min_tot=250, min_hp=100, min_hp_m=10)

# Bootstrap 1000-iter Match-Clustered CIs for Top 15
rng = np.random.default_rng(42)
b_tf['is_hp'] = b_tf['PI_bat'] >= cut_b
bw_tf['is_hp'] = bw_tf['PI_bowl'] >= cut_bw

def boot_ci(df_hp, id_col, val_col, ids):
    rows = []
    for pid in ids:
        sub = df_hp[df_hp[id_col] == pid]
        m_arr = sub.groupby('match_id')[val_col].agg(['sum','count']).values
        n_m = len(m_arr)
        bs = np.zeros(1000)
        for i in range(1000):
            s = m_arr[rng.integers(0, n_m, size=n_m)]
            bs[i] = 100.0 * s[:, 0].sum() / max(1.0, s[:, 1].sum())
        rows.append({
            id_col: pid,
            'boot_mean': bs.mean(), 'ci_low_95': np.percentile(bs, 2.5),
            'ci_high_95': np.percentile(bs, 97.5), 'p_gt_zero': (bs > 0).mean()
        })
    return pd.DataFrame(rows)

boot_b = boot_ci(b_tf[b_tf['is_hp']], 'striker_id', 'tva_bat_opp', rb.head(15)['striker_id'])
boot_bw = boot_ci(bw_tf[bw_tf['is_hp']], 'bowler_id', 'tva_bowl_opp', rbw.head(15)['bowler_id'])

rb = rb.merge(boot_b, on='striker_id', how='left')
rbw = rbw.merge(boot_bw, on='bowler_id', how='left')

rb.to_csv(f'{OUT}/eligible_batters.csv', index=False)
rbw.to_csv(f'{OUT}/eligible_bowlers.csv', index=False)
rb.head(10).to_csv(f'{OUT}/top10_batters.csv', index=False)
rbw.head(10).to_csv(f'{OUT}/top10_bowlers.csv', index=False)

# Sensitivity summary
sens_b = rb[['striker_id','striker_name','rank']].head(12).rename(columns={'rank':'Primary_Q75_K500'})
for lbl, df_s in [('Q70_K500', rb_q70), ('Q80_K500', rb_q80), ('Q75_K300', rb_k300), ('Q75_K800', rb_k800), ('IPL_Only', rb_ipl)]:
    sens_b = sens_b.merge(df_s[['striker_id','rank']].rename(columns={'rank':lbl}), on='striker_id', how='left')
sens_b.to_csv(f'{OUT}/sensitivity_batters.csv', index=False)

sens_bw = rbw[['bowler_id','bowler_name','rank']].head(12).rename(columns={'rank':'Primary_Q75_K500'})
for lbl, df_s in [('Q70_K500', rbw_q70), ('Q80_K500', rbw_q80), ('Q75_K300', rbw_k300), ('Q75_K800', rbw_k800), ('IPL_Only', rbw_ipl)]:
    sens_bw = sens_bw.merge(df_s[['bowler_id','rank']].rename(columns={'rank':lbl}), on='bowler_id', how='left')
sens_bw.to_csv(f'{OUT}/sensitivity_bowlers.csv', index=False)

# Micro-tactical execution shifts for Top 10 Batters & Top 10 Bowlers
top10_bat_ids = list(rb.head(10)['striker_id'])
top10_bowl_ids = list(rbw.head(10)['bowler_id'])

bat_micro = []
for pid in top10_bat_ids:
    sub = b_tf[b_tf['striker_id'] == pid]
    hp = sub[sub['is_hp']]; lp = sub[~sub['is_hp']]
    p_hp = hp[hp['pace_or_spin']=='Pace']; s_hp = hp[hp['pace_or_spin']=='Spin']
    bat_micro.append({
        'striker_id': pid,
        'striker_name': sub['striker_name'].iloc[0],
        'hp_pace_b': len(p_hp), 'hp_pace_sr': 100*p_hp['batter_runs'].mean() if len(p_hp) else np.nan,
        'hp_pace_tva100': 100*p_hp['tva_bat_opp'].mean() if len(p_hp) else np.nan,
        'hp_spin_b': len(s_hp), 'hp_spin_sr': 100*s_hp['batter_runs'].mean() if len(s_hp) else np.nan,
        'hp_spin_tva100': 100*s_hp['tva_bat_opp'].mean() if len(s_hp) else np.nan,
        'hp_i1_b': (hp['innings']==1).sum(), 'hp_i1_tva100': 100*hp.loc[hp['innings']==1, 'tva_bat_opp'].mean(),
        'hp_i2_b': (hp['innings']==2).sum(), 'hp_i2_tva100': 100*hp.loc[hp['innings']==2, 'tva_bat_opp'].mean(),
        'hp_bdry_pct': 100*hp['batter_runs'].isin([4,6]).mean(),
        'lp_bdry_pct': 100*lp['batter_runs'].isin([4,6]).mean(),
        'hp_dot_pct': 100*(hp['batter_runs']==0).mean(),
        'lp_dot_pct': 100*(lp['batter_runs']==0).mean(),
        'top_hp_stroke': hp['stroke'].value_counts().index[0] if hp['stroke'].notna().any() else 'N/A',
        'ipl_hp_tva100': 100*hp.loc[hp['match_tier']=='IPL', 'tva_bat_opp'].mean() if (hp['match_tier']=='IPL').sum()>0 else np.nan,
        't20i_hp_tva100': 100*hp.loc[hp['match_tier']=='T20I_FM_vs_FM', 'tva_bat_opp'].mean() if (hp['match_tier']=='T20I_FM_vs_FM').sum()>0 else np.nan,
    })
pd.DataFrame(bat_micro).to_csv(f'{OUT}/micro_tactics_batters.csv', index=False)

bowl_micro = []
for pid in top10_bowl_ids:
    sub = bw_tf[bw_tf['bowler_id'] == pid]
    hp = sub[sub['is_hp']]; lp = sub[~sub['is_hp']]
    spd_hp = hp.loc[hp['bowl_speed'].between(65, 160), 'bowl_speed']
    spd_lp = lp.loc[lp['bowl_speed'].between(65, 160), 'bowl_speed']
    # Top 2 delivery styles in HP vs LP
    ds_hp = hp['delivery_style'].value_counts(normalize=True).head(3)
    ds_str = ', '.join([f"{k} ({100*v:.0f}%)" for k, v in ds_hp.items()])
    bowl_micro.append({
        'bowler_id': pid,
        'bowler_name': sub['bowler_name'].iloc[0],
        'pace_spin': sub['pace_or_spin'].iloc[0],
        'bowl_style': sub['bowling_type'].iloc[0],
        'hp_spd_mean': spd_hp.mean() if len(spd_hp)>=20 else np.nan,
        'lp_spd_mean': spd_lp.mean() if len(spd_lp)>=20 else np.nan,
        'hp_spd_std': spd_hp.std() if len(spd_hp)>=20 else np.nan,
        'hp_dot_pct': 100*(hp['bowler_runs']==0).mean(),
        'lp_dot_pct': 100*(lp['bowler_runs']==0).mean(),
        'hp_bdry_pct': 100*hp['runs'].isin([4,6]).mean(),
        'lp_bdry_pct': 100*lp['runs'].isin([4,6]).mean(),
        'hp_death_leg': hp.loc[hp['phase']=='Death', 'is_legal'].sum(),
        'hp_death_tva100': 100*hp.loc[hp['phase']=='Death', 'tva_bowl_opp'].mean() if (hp['phase']=='Death').sum()>0 else np.nan,
        'hp_i2_leg': hp.loc[hp['innings']==2, 'is_legal'].sum(),
        'hp_i2_tva100': 100*hp.loc[hp['innings']==2, 'tva_bowl_opp'].mean() if (hp['innings']==2).sum()>0 else np.nan,
        'top_hp_styles': ds_str,
        'ipl_hp_tva100': 100*hp.loc[hp['match_tier']=='IPL', 'tva_bowl_opp'].mean() if (hp['match_tier']=='IPL').sum()>0 else np.nan,
        't20i_hp_tva100': 100*hp.loc[hp['match_tier']=='T20I_FM_vs_FM', 'tva_bowl_opp'].mean() if (hp['match_tier']=='T20I_FM_vs_FM').sum()>0 else np.nan,
    })
pd.DataFrame(bowl_micro).to_csv(f'{OUT}/micro_tactics_bowlers.csv', index=False)

# Signature HP performances for Top 10 Batters & Top 10 Bowlers
sig_bat = b_tf[b_tf['striker_id'].isin(top10_bat_ids) & b_tf['is_hp']].groupby(
    ['striker_id','striker_name','match_id','match_date','comp_name','home_team','away_team','result']
).agg(hp_b=('is_batter_ball','sum'), hp_r=('batter_runs','sum'), hp_tva=('tva_bat_opp','sum'), hp_wpa=('wpa_bat','sum')).reset_index()
sig_bat.sort_values('hp_tva', ascending=False).groupby('striker_id').head(2).to_csv(f'{OUT}/signature_bat_innings.csv', index=False)

sig_bowl = bw_tf[bw_tf['bowler_id'].isin(top10_bowl_ids) & bw_tf['is_hp']].groupby(
    ['bowler_id','bowler_name','match_id','match_date','comp_name','home_team','away_team','result']
).agg(hp_leg=('is_legal','sum'), hp_r=('bowler_runs','sum'), hp_w=('is_bowler_wkt','sum'), hp_tva=('tva_bowl_opp','sum'), hp_wpa=('wpa_bowl','sum')).reset_index()
sig_bowl.sort_values('hp_tva', ascending=False).groupby('bowler_id').head(2).to_csv(f'{OUT}/signature_bowl_spells.csv', index=False)

# RR IPL 2026 full roster pressure diagnostics
rr_ids = set(ctpm[(ctpm['comp_name']=='IPL 2026') & (ctpm['team_name']=='Rajasthan Royals')]['player_id'])
rr_bat = b_tf[b_tf['striker_id'].isin(rr_ids)].groupby(['striker_id','striker_name']).agg(
    tot_b=('is_batter_ball','sum'),
    hp_b=('is_hp','sum'),
    hp_m=('match_id', lambda x: b_tf.loc[x.index][b_tf.loc[x.index,'is_hp']]['match_id'].nunique()),
    hp_sr=('batter_runs', lambda x: 100*x[b_tf.loc[x.index,'is_hp']].mean() if b_tf.loc[x.index,'is_hp'].sum()>0 else np.nan),
    hp_tva100=('tva_bat_opp', lambda x: 100*x[b_tf.loc[x.index,'is_hp']].mean() if b_tf.loc[x.index,'is_hp'].sum()>0 else np.nan),
    lp_tva100=('tva_bat_opp', lambda x: 100*x[~b_tf.loc[x.index,'is_hp']].mean() if (~b_tf.loc[x.index,'is_hp']).sum()>0 else np.nan),
).reset_index()
rr_bat.to_csv(f'{OUT}/rr_roster_bat.csv', index=False)

rr_bowl = bw_tf[bw_tf['bowler_id'].isin(rr_ids)].groupby(['bowler_id','bowler_name']).agg(
    tot_leg=('is_legal','sum'),
    hp_leg=('is_legal', lambda x: x[bw_tf.loc[x.index,'is_hp']].sum()),
    hp_m=('match_id', lambda x: bw_tf.loc[x.index][bw_tf.loc[x.index,'is_hp']]['match_id'].nunique()),
    hp_econ=('bowler_runs', lambda x: 6*x[bw_tf.loc[x.index,'is_hp']].sum()/max(1, bw_tf.loc[x.index][bw_tf.loc[x.index,'is_hp']]['is_legal'].sum())),
    hp_tva100=('tva_bowl_opp', lambda x: 100*x[bw_tf.loc[x.index,'is_hp']].mean() if bw_tf.loc[x.index,'is_hp'].sum()>0 else np.nan),
    lp_tva100=('tva_bowl_opp', lambda x: 100*x[~bw_tf.loc[x.index,'is_hp']].mean() if (~bw_tf.loc[x.index,'is_hp']).sum()>0 else np.nan),
).reset_index()
rr_bowl.to_csv(f'{OUT}/rr_roster_bowl.csv', index=False)

# IPL-only emerging / non-top-10 auction gems (e.g. Ashutosh Sharma, Harpreet Brar, Lungi Ngidi, etc.)
rb_ipl.head(15).to_csv(f'{OUT}/ipl_only_top15_batters.csv', index=False)
rbw_ipl.head(15).to_csv(f'{OUT}/ipl_only_top15_bowlers.csv', index=False)

# Negative clutch contrast list (high-volume >=600 balls)
rb[rb['tot_b']>=600].sort_values('clutch_resid').head(10).to_csv(f'{OUT}/batters_biggest_dip.csv', index=False)
rbw[rbw['tot_leg']>=600].sort_values('clutch_resid').head(10).to_csv(f'{OUT}/bowlers_biggest_dip.csv', index=False)

# Save summary metadata JSON
meta = {
    'total_deliveries': int(len(bbb)),
    'regular_deliveries': int(len(reg)),
    'super_over_deliveries': int(len(bbb[bbb['innings'].isin([3,4])])),
    'total_matches_metadata': int(len(matches)),
    'matches_with_bbb': int(bbb['match_id'].nunique()),
    'valid_outcome_matches': int(m['has_valid_outcome'].sum()),
    'top_flight_deliveries': int(len(tf)),
    'top_flight_matches': int(tf['match_id'].nunique()),
    'eligible_batters': int(len(rb)),
    'eligible_bowlers': int(len(rbw)),
    'q75_pi_bat': float(cut_b),
    'q75_pi_bowl': float(cut_bw),
    'bat_hp_lp_slope': float(beta_b),
    'bat_hp_lp_intercept': float(int_b),
    'bowl_hp_lp_slope': float(beta_bw),
    'bowl_hp_lp_intercept': float(int_bw),
    'res_mae_2026': res_mae_26,
    'res_r2_2026': res_r2_26,
    'spearman_bat_q70': float(spearmanr(rb.merge(rb_q70, on='striker_id')['PRS_z_x'], rb.merge(rb_q70, on='striker_id')['PRS_z_y']).statistic),
    'spearman_bat_k300': float(spearmanr(rb.merge(rb_k300, on='striker_id')['PRS_z_x'], rb.merge(rb_k300, on='striker_id')['PRS_z_y']).statistic),
    'spearman_bat_k800': float(spearmanr(rb.merge(rb_k800, on='striker_id')['PRS_z_x'], rb.merge(rb_k800, on='striker_id')['PRS_z_y']).statistic),
    'spearman_bowl_q70': float(spearmanr(rbw.merge(rbw_q70, on='bowler_id')['PRS_z_x'], rbw.merge(rbw_q70, on='bowler_id')['PRS_z_y']).statistic),
    'spearman_bowl_k300': float(spearmanr(rbw.merge(rbw_k300, on='bowler_id')['PRS_z_x'], rbw.merge(rbw_k300, on='bowler_id')['PRS_z_y']).statistic),
    'spearman_bowl_k800': float(spearmanr(rbw.merge(rbw_k800, on='bowler_id')['PRS_z_x'], rbw.merge(rbw_k800, on='bowler_id')['PRS_z_y']).statistic),
}
with open(f'{OUT}/summary_metrics.json', 'w') as f:
    json.dump(meta, f, indent=2)

# Also save lightweight delivery-level sample / phase distributions for plotting (instead of 167MB pickle!)
sample_pi = reg[['innings','phase','match_tier','PI_bat','PI_bowl','wp_before','wp_lev','wicket_value']].sample(n=40000, random_state=42)
sample_pi.to_csv(f'{OUT}/pi_sample_for_plots.csv', index=False)

print("5. All outputs saved successfully! Summary meta:")
print(json.dumps(meta, indent=2))
