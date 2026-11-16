import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

BASE = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\data_pack\Data Pack'
OUT = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\outputs'

comp = pd.read_csv(f'{BASE}/competitions.csv')
matches = pd.read_csv(f'{BASE}/matches.csv')
bbb = pd.read_csv(f'{BASE}/ball_by_ball.csv', low_memory=False)
teams = pd.read_csv(f'{BASE}/teams.csv')

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

# Regular innings 1 and 2
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
reg['is_nonstriker_out'] = (reg['is_wicket']) & (reg['dismissed_player_id'] == reg['non_striker_id'])

g = reg.groupby(['match_id', 'innings'], sort=False)
reg['cum_runs_before'] = g['tot_runs'].cumsum() - reg['tot_runs']
reg['cum_wkts_before'] = g['is_wicket'].cumsum() - reg['is_wicket'].astype(int)
reg['legal_balls_before'] = g['is_legal'].cumsum() - reg['is_legal'].astype(int)
reg['runs_last18'] = reg['cum_runs_before'] - g['cum_runs_before'].shift(18).fillna(0)
reg['wkts_last18'] = reg['cum_wkts_before'] - g['cum_wkts_before'].shift(18).fillna(0)

# Batter balls & runs so far in innings (before current ball)
gb = reg.groupby(['match_id', 'innings', 'striker_id'], sort=False)
reg['batter_balls_before'] = gb['is_batter_ball'].cumsum() - reg['is_batter_ball'].astype(int)
reg['batter_runs_before'] = gb['batter_runs'].cumsum() - reg['batter_runs']

i1_totals = reg[reg['innings'] == 1].groupby('match_id')['tot_runs'].sum().rename('inn1_total')
m = m.merge(i1_totals, on='match_id', how='left')
m['max_balls_1'] = m['first_inn_revised_overs'].fillna(120).astype(int)
m['max_balls_2'] = m['second_inn_revised_overs'].fillna(120).astype(int)
m['target'] = np.where(m['revised_target'].notna(), m['revised_target'], m['inn1_total'] + 1)

dls_mid_cut = m['result'].str.contains('D/L|DLS', case=False, na=False) & m['revised_target'].isna()
m['has_valid_outcome'] = m['win_team_id'].notna() & (~dls_mid_cut)

reg = reg.merge(m[['match_id','comp_name','season','year','stage','is_knockout','venue_id','match_tier',
                   'max_balls_1','max_balls_2','target','inn1_total','win_team_id','has_valid_outcome','result',
                   'home_team','away_team']],
                on='match_id', how='left')

reg['max_balls'] = np.where(reg['innings'] == 1, reg['max_balls_1'], reg['max_balls_2'])
reg['balls_rem'] = (reg['max_balls'] - reg['legal_balls_before']).clip(lower=1)
reg['wickets_rem'] = (10 - reg['cum_wkts_before']).clip(lower=0)
reg['crr'] = np.where(reg['legal_balls_before'] > 0, 6.0 * reg['cum_runs_before'] / reg['legal_balls_before'], 0.0)
reg['bat_won'] = (reg['batting_team_id'] == reg['win_team_id']).astype(float)

# Shrunk leave-one-match-out environment par RPO
match_rpo = reg.groupby(['match_id','match_tier','season','venue_id']).agg(
    m_runs=('tot_runs','sum'), m_balls=('is_legal','sum')
).reset_index()
ts_sum = match_rpo.groupby(['match_tier','season'])[['m_runs','m_balls']].transform('sum')
ts_loo_rpo = 6.0 * (ts_sum['m_runs'] - match_rpo['m_runs']) / (ts_sum['m_balls'] - match_rpo['m_balls']).clip(lower=1)
t_sum = match_rpo.groupby('match_tier')[['m_runs','m_balls']].transform('sum')
t_loo_rpo = 6.0 * (t_sum['m_runs'] - match_rpo['m_runs']) / (t_sum['m_balls'] - match_rpo['m_balls']).clip(lower=1)
ts_loo_rpo = ts_loo_rpo.fillna(t_loo_rpo)
v_sum = match_rpo.groupby('venue_id')[['m_runs','m_balls']].transform('sum')
v_runs_loo = v_sum['m_runs'] - match_rpo['m_runs']
v_balls_loo = v_sum['m_balls'] - match_rpo['m_balls']
match_rpo['env_par_rpo'] = 6.0 * (v_runs_loo + 1440.0 * (ts_loo_rpo / 6.0)) / (v_balls_loo + 1440.0)
reg = reg.merge(match_rpo[['match_id','env_par_rpo']], on='match_id', how='left')

# 1. Fit Out-of-Fold Resource Model (Expected Remaining Runs) on full 20-over Innings 1
gkf = GroupKFold(n_splits=5)
full_i1_mask = (reg['innings'] == 1) & (reg['max_balls'] == 120) & (reg['has_valid_outcome'])
full_i1 = reg[full_i1_mask].copy()
full_i1['runs_to_go'] = full_i1['inn1_total'] - full_i1['cum_runs_before']
res_feats = ['balls_rem', 'wickets_rem', 'env_par_rpo']

# Also evaluate OOF predictions for full_i1, and fit final resource model on all full_i1 (since it's strictly monotonic & regularized)
# Even better: use OOF resource model per fold!
reg['exp_rem_runs'] = np.nan
folds = list(gkf.split(reg, groups=reg['match_id']))
for fold_i, (tr_idx, va_idx) in enumerate(folds):
    tr_sub = reg.iloc[tr_idx]
    tr_i1 = tr_sub[(tr_sub['innings'] == 1) & (tr_sub['max_balls'] == 120) & (tr_sub['has_valid_outcome'])].copy()
    tr_i1['runs_to_go'] = tr_i1['inn1_total'] - tr_i1['cum_runs_before']
    mdl_rem = HistGradientBoostingRegressor(
        max_iter=100, max_depth=4, min_samples_leaf=500, l2_regularization=20.0,
        monotonic_cst=[1, 1, 1], random_state=42
    )
    mdl_rem.fit(tr_i1[res_feats], tr_i1['runs_to_go'])
    reg.iloc[va_idx, reg.columns.get_loc('exp_rem_runs')] = mdl_rem.predict(reg.iloc[va_idx][res_feats])

reg.loc[reg['wickets_rem'] == 0, 'exp_rem_runs'] = 0.0
reg['exp_rem_runs'] = reg['exp_rem_runs'].clip(lower=0.0)

# Helper to compute features for WP1 & WP2 given (cum_runs, balls_rem, wickets_rem, ...)
def build_i1_feats(df, exp_rem):
    out = pd.DataFrame(index=df.index)
    par_total = df['env_par_rpo'] * (df['max_balls'] / 6.0)
    proj_total = df['cum_runs_before'] + exp_rem
    prog = df['legal_balls_before'] / df['max_balls']
    out['proj_surplus'] = proj_total - par_total
    out['proj_surplus_x_prog'] = out['proj_surplus'] * (0.3 + 0.7 * prog)
    out['cum_runs_before'] = df['cum_runs_before']
    out['wickets_rem'] = df['wickets_rem']
    out['balls_rem'] = df['balls_rem']
    out['progress'] = prog
    out['env_par_rpo'] = df['env_par_rpo']
    out['is_ipl'] = (df['match_tier'] == 'IPL').astype(int)
    return out

def build_i2_feats(df, exp_rem):
    out = pd.DataFrame(index=df.index)
    runs_needed = (df['target'] - df['cum_runs_before']).clip(lower=0)
    rrr = 6.0 * runs_needed / df['balls_rem'].clip(lower=1)
    chase_surplus = exp_rem - runs_needed
    prog = df['legal_balls_before'] / df['max_balls']
    out['chase_surplus'] = chase_surplus
    out['surplus_std'] = chase_surplus / np.sqrt(df['balls_rem'].clip(lower=1) + 6.0)
    out['log_rrr_ratio'] = np.log((rrr + 1.0) / (df['env_par_rpo'] + 1.0))
    out['runs_needed'] = runs_needed
    out['balls_rem'] = df['balls_rem']
    out['wickets_rem'] = df['wickets_rem']
    out['env_par_rpo'] = df['env_par_rpo']
    out['progress'] = prog
    out['wkt_x_balls'] = df['wickets_rem'] * prog
    return out

# Fit OOF WP1 & WP2 models (and keep fold models for counterfactual Leverage!)
# Actually let's fit a full resource model too just for counterfactual grid, or use each match's OOF fold model!
# Let's check coefficients and OOF performance of LogReg_Eng_C0.1 first
X1_all = build_i1_feats(reg, reg['exp_rem_runs'])
X2_all = build_i2_feats(reg, reg['exp_rem_runs'])

reg['wp_before'] = np.nan
# Store counterfactual resource predictions in one vectorized batch via full-data regularized models for counterfactuals,
# while checking OOF match accuracy! Wait: using fold OOF models for counterfactuals guarantees ZERO leakage even for WP deltas!
# Let's compute OOF wp_before AND counterfactuals (dot, +1, +4, +6, wicket) per fold!
mdl_rem_full = HistGradientBoostingRegressor(
    max_iter=100, max_depth=4, min_samples_leaf=500, l2_regularization=20.0,
    monotonic_cst=[1, 1, 1], random_state=42
)
valid_i1_full = reg[(reg['innings']==1)&(reg['max_balls']==120)&(reg['has_valid_outcome'])].copy()
valid_i1_full['runs_to_go'] = valid_i1_full['inn1_total'] - valid_i1_full['cum_runs_before']
mdl_rem_full.fit(valid_i1_full[res_feats], valid_i1_full['runs_to_go'])

# Counterfactual states (after 1 legal ball):
def cf_state(df, add_runs, add_wkt):
    cf = df[['cum_runs_before','legal_balls_before','balls_rem','wickets_rem','max_balls','env_par_rpo','target','match_tier']].copy()
    cf['cum_runs_before'] = cf['cum_runs_before'] + add_runs
    cf['legal_balls_before'] = (cf['legal_balls_before'] + 1).clip(upper=cf['max_balls'])
    cf['balls_rem'] = (cf['balls_rem'] - 1).clip(lower=1)
    cf['wickets_rem'] = (cf['wickets_rem'] - add_wkt).clip(lower=0)
    exp_rem_cf = mdl_rem_full.predict(cf[['balls_rem','wickets_rem','env_par_rpo']])
    exp_rem_cf = np.where(cf['wickets_rem'] == 0, 0.0, np.maximum(0.0, exp_rem_cf))
    # Edge: if innings 2 and target reached -> WP = 1.0; if innings 2 and (wickets_rem == 0 or balls_rem_after == 0) & target not reached -> WP = 0.0
    return cf, exp_rem_cf

wp_before = np.zeros(len(reg))
wp_dot = np.zeros(len(reg))
wp_1 = np.zeros(len(reg))
wp_4 = np.zeros(len(reg))
wp_6 = np.zeros(len(reg))
wp_w = np.zeros(len(reg))

# Also compute actual wp_after for WPA!
reg['balls_rem_after'] = (reg['max_balls'] - (reg['legal_balls_before'] + reg['is_legal'].astype(int))).clip(lower=0)
reg['wickets_rem_after'] = (10 - (reg['cum_wkts_before'] + reg['is_wicket'].astype(int))).clip(lower=0)
reg['cum_runs_after'] = reg['cum_runs_before'] + reg['tot_runs']

cf_actual_df = pd.DataFrame({
    'cum_runs_before': reg['cum_runs_after'],
    'legal_balls_before': reg['legal_balls_before'] + reg['is_legal'].astype(int),
    'balls_rem': reg['balls_rem_after'].clip(lower=1),
    'wickets_rem': reg['wickets_rem_after'],
    'max_balls': reg['max_balls'],
    'env_par_rpo': reg['env_par_rpo'],
    'target': reg['target'],
    'match_tier': reg['match_tier']
})
exp_rem_actual = np.where(cf_actual_df['wickets_rem']==0, 0.0, np.maximum(0.0, mdl_rem_full.predict(cf_actual_df[['balls_rem','wickets_rem','env_par_rpo']])))

cf0, er0 = cf_state(reg, 0, 0)
cf1, er1 = cf_state(reg, 1, 0)
cf4, er4 = cf_state(reg, 4, 0)
cf6, er6 = cf_state(reg, 6, 0)
cfw, erw = cf_state(reg, 0, 1)

X1_0, X1_1, X1_4, X1_6, X1_w, X1_act = [build_i1_feats(d, e) for d, e in [(cf0,er0),(cf1,er1),(cf4,er4),(cf6,er6),(cfw,erw),(cf_actual_df,exp_rem_actual)]]
X2_0, X2_1, X2_4, X2_6, X2_w, X2_act = [build_i2_feats(d, e) for d, e in [(cf0,er0),(cf1,er1),(cf4,er4),(cf6,er6),(cfw,erw),(cf_actual_df,exp_rem_actual)]]

wp_after = np.zeros(len(reg))

for fold_i, (tr_idx, va_idx) in enumerate(folds):
    tr_mask_1 = np.zeros(len(reg), dtype=bool); tr_mask_1[tr_idx] = True
    tr1 = tr_mask_1 & (reg['innings']==1) & (reg['has_valid_outcome'])
    tr2 = tr_mask_1 & (reg['innings']==2) & (reg['has_valid_outcome'])
    
    va_mask_1 = np.zeros(len(reg), dtype=bool); va_mask_1[va_idx] = True
    va1 = va_mask_1 & (reg['innings']==1)
    va2 = va_mask_1 & (reg['innings']==2)
    
    m1 = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=0.1))])
    m1.fit(X1_all[tr1], reg.loc[tr1, 'bat_won'])
    
    m2 = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=0.1))])
    m2.fit(X2_all[tr2], reg.loc[tr2, 'bat_won'])
    
    wp_before[va1] = m1.predict_proba(X1_all[va1])[:, 1]
    wp_dot[va1]    = m1.predict_proba(X1_0[va1])[:, 1]
    wp_1[va1]      = m1.predict_proba(X1_1[va1])[:, 1]
    wp_4[va1]      = m1.predict_proba(X1_4[va1])[:, 1]
    wp_6[va1]      = m1.predict_proba(X1_6[va1])[:, 1]
    wp_w[va1]      = m1.predict_proba(X1_w[va1])[:, 1]
    wp_after[va1]  = m1.predict_proba(X1_act[va1])[:, 1]
    
    wp_before[va2] = m2.predict_proba(X2_all[va2])[:, 1]
    wp_dot[va2]    = m2.predict_proba(X2_0[va2])[:, 1]
    wp_1[va2]      = m2.predict_proba(X2_1[va2])[:, 1]
    wp_4[va2]      = m2.predict_proba(X2_4[va2])[:, 1]
    wp_6[va2]      = m2.predict_proba(X2_6[va2])[:, 1]
    wp_w[va2]      = m2.predict_proba(X2_w[va2])[:, 1]
    wp_after[va2]  = m2.predict_proba(X2_act[va2])[:, 1]

# Exact terminal boundary corrections for Innings 2 counterfactuals & wp_after
i2_m = (reg['innings'] == 2).values
last_ball = (reg['balls_rem'] == 1).values
# For cf_w when wickets_rem == 1 -> all out!
all_out_w = (reg['wickets_rem'] <= 1).values
for arr, add_r, add_w in [(wp_dot,0,0),(wp_1,1,0),(wp_4,4,0),(wp_6,6,0),(wp_w,0,1)]:
    won_cf = i2_m & ((reg['cum_runs_before'].values + add_r) >= reg['target'].values)
    lost_cf = i2_m & (~won_cf) & (last_ball | ((add_w == 1) & all_out_w))
    # wait: if tied at last ball (runs == target - 1), WP = 0.5
    tie_cf = i2_m & last_ball & ((reg['cum_runs_before'].values + add_r) == (reg['target'].values - 1))
    arr[won_cf] = 1.0
    arr[lost_cf] = 0.0
    arr[tie_cf] = 0.5

# Also for actual wp_after in Innings 2:
won_act = i2_m & (reg['cum_runs_after'].values >= reg['target'].values)
lost_act = i2_m & (~won_act) & ((reg['balls_rem_after'].values == 0) | (reg['wickets_rem_after'].values == 0))
tie_act = i2_m & (reg['balls_rem_after'].values == 0) & (reg['cum_runs_after'].values == (reg['target'].values - 1))
wp_after[won_act] = 1.0
wp_after[lost_act] = 0.0
wp_after[tie_act] = 0.5
# Also if all out in Innings 1, projected remaining = 0 so X1_act already handles wickets_rem_after==0!

reg['wp_before'] = wp_before
reg['wp_after'] = wp_after
reg['wpa_bat'] = reg['wp_after'] - reg['wp_before']
reg['wpa_bowl'] = -reg['wpa_bat']

# Empirical outcome probabilities by phase (dot, 1/2/3, 4, 6, wicket) for Leverage expectation
# Rough T20 empirical weights: ~0.35 dot, ~0.40 1/2/3, ~0.12 4, ~0.07 6, ~0.06 wicket
# Better: compute exact phase weights from data!
reg['phase'] = np.where(reg['overid'] <= 6, 'PP', np.where(reg['overid'] <= 15, 'Middle', 'Death'))
p_dot = ((reg['tot_runs']==0) & (~reg['is_wicket'])).groupby(reg['phase']).mean()
p_123 = ((reg['tot_runs'].isin([1,2,3])) & (~reg['is_wicket'])).groupby(reg['phase']).mean()
p_4 = ((reg['runs']==4) & (~reg['is_wicket'])).groupby(reg['phase']).mean()
p_6 = ((reg['runs']==6) & (~reg['is_wicket'])).groupby(reg['phase']).mean()
p_w = reg['is_wicket'].groupby(reg['phase']).mean()
print('Phase outcome probabilities:')
for ph in ['PP','Middle','Death']:
    print(f'  {ph:6s}: dot={p_dot[ph]:.3f}, 1-3={p_123[ph]:.3f}, 4={p_4[ph]:.3f}, 6={p_6[ph]:.3f}, W={p_w[ph]:.3f}')

w0 = reg['phase'].map(p_dot).values
w1 = reg['phase'].map(p_123).values
w4 = reg['phase'].map(p_4).values
w6 = reg['phase'].map(p_6).values
ww = reg['phase'].map(p_w).values
w_sum = w0 + w1 + w4 + w6 + ww
w0, w1, w4, w6, ww = w0/w_sum, w1/w_sum, w4/w_sum, w6/w_sum, ww/w_sum

mu_cf = w0*wp_dot + w1*wp_1 + w4*wp_4 + w6*wp_6 + ww*wp_w
leverage_rms = np.sqrt(
    w0*(wp_dot - mu_cf)**2 + w1*(wp_1 - mu_cf)**2 + w4*(wp_4 - mu_cf)**2 +
    w6*(wp_6 - mu_cf)**2 + ww*(wp_w - mu_cf)**2
)
reg['wp_leverage'] = leverage_rms
reg['wp_uncertainty'] = 4.0 * reg['wp_before'] * (1.0 - reg['wp_before'])

print('\n=== WP Leverage summary by Innings & Phase ===')
print(reg.groupby(['innings','phase'])['wp_leverage'].describe(percentiles=[0.5, 0.75, 0.9]).round(4).to_string())

# Print sample extreme leverage balls and dead-game balls to verify!
print('\n=== Top 10 highest WP Leverage balls in Innings 2 ===')
top_lev = reg[reg['innings']==2].sort_values('wp_leverage', ascending=False).head(10)
print(top_lev[['match_id','comp_name','overid','over_ball_num','cum_runs_before','cum_wkts_before','target','balls_rem','wp_before','wp_leverage','striker_name','bowler_name','runs','is_wicket']].to_string())

print('\n=== Sample high-RRR dead chase balls (balls_rem<=12, needed>=45, wkts_rem<=3) ===')
dead = reg[(reg['innings']==2) & (reg['balls_rem']<=12) & ((reg['target']-reg['cum_runs_before'])>=45) & (reg['wickets_rem']<=3)].head(5)
print(dead[['match_id','overid','cum_runs_before','cum_wkts_before','target','balls_rem','wp_before','wp_leverage']].to_string())

print('\n=== Top 10 highest WP Leverage balls in Innings 1 ===')
top_lev1 = reg[reg['innings']==1].sort_values('wp_leverage', ascending=False).head(10)
print(top_lev1[['match_id','comp_name','overid','over_ball_num','cum_runs_before','cum_wkts_before','balls_rem','wp_before','wp_leverage','striker_name','bowler_name']].to_string())
