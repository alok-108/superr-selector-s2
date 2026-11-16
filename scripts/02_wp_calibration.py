import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\data_pack\Data Pack'

comp = pd.read_csv(f'{BASE}/competitions.csv')
matches = pd.read_csv(f'{BASE}/matches.csv')
bbb = pd.read_csv(f'{BASE}/ball_by_ball.csv', low_memory=False)

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

reg = bbb[bbb['innings'].isin([1, 2])].copy()
reg = reg.sort_values(['match_id', 'innings', 'overid', 'over_ball_num', 'innings_ball_num']).reset_index(drop=True)

reg['is_wide'] = reg['extras_type'] == 'wide'
reg['is_noball'] = reg['extras_type'].isin(['Overstep', 'other'])
reg['is_legal'] = ~(reg['is_wide'] | reg['is_noball'])
reg['tot_runs'] = reg['runs'] + reg['extras1_runs'] + reg['extras_2_runs']

# Vectorized cumulative & rolling stats
g = reg.groupby(['match_id', 'innings'], sort=False)
cum_runs = g['tot_runs'].cumsum()
cum_wkts = g['is_wicket'].cumsum()
cum_legal = g['is_legal'].cumsum()

reg['cum_runs_before'] = cum_runs - reg['tot_runs']
reg['cum_wkts_before'] = cum_wkts - reg['is_wicket'].astype(int)
reg['legal_balls_before'] = cum_legal - reg['is_legal'].astype(int)

# Fast vectorized last-18-deliveries rolling using shifted cumsum within group
lag18_runs = g['tot_runs'].shift(18, fill_value=0)
# Wait: shift(18) within group gives value 18 rows earlier; cum_runs_before - cum_runs_18_ago
cum_runs_before_s = reg['cum_runs_before']
cum_runs_18_ago = g['cum_runs_before'].shift(18).fillna(0)
reg['runs_last18'] = cum_runs_before_s - cum_runs_18_ago

cum_wkts_18_ago = g['cum_wkts_before'].shift(18).fillna(0)
reg['wkts_last18'] = reg['cum_wkts_before'] - cum_wkts_18_ago

i1_totals = reg[reg['innings'] == 1].groupby('match_id')['tot_runs'].sum().rename('inn1_total')
m = m.merge(i1_totals, on='match_id', how='left')
m['max_balls_1'] = m['first_inn_revised_overs'].fillna(120).astype(int)
m['max_balls_2'] = m['second_inn_revised_overs'].fillna(120).astype(int)
m['target'] = np.where(m['revised_target'].notna(), m['revised_target'], m['inn1_total'] + 1)

# Clean outcome mask
dls_mid_cut = m['result'].str.contains('D/L|DLS', case=False, na=False) & m['revised_target'].isna()
m['has_valid_outcome'] = m['win_team_id'].notna() & (~dls_mid_cut)

reg = reg.merge(m[['match_id','comp_name','season','year','stage','knock_out','venue_id','match_tier',
                   'max_balls_1','max_balls_2','target','inn1_total','win_team_id','has_valid_outcome','result']],
                on='match_id', how='left')

reg['max_balls'] = np.where(reg['innings'] == 1, reg['max_balls_1'], reg['max_balls_2'])
reg['balls_rem'] = (reg['max_balls'] - reg['legal_balls_before']).clip(lower=1)
reg['wickets_rem'] = (10 - reg['cum_wkts_before']).clip(lower=0)
reg['crr'] = np.where(reg['legal_balls_before'] > 0, 6.0 * reg['cum_runs_before'] / reg['legal_balls_before'], 0.0)
reg['bat_won'] = (reg['batting_team_id'] == reg['win_team_id']).astype(float)

# Shrunk leave-one-match-out environment RPO
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

# 1. Empirical T20 Acceleration & Resource Model (fit ONLY on train<=2025 full 20-over matches first to inspect)
full_i1 = reg[(reg['innings'] == 1) & (reg['max_balls'] == 120) & (reg['year'] <= 2025) & (reg['has_valid_outcome'])].copy()
full_i1['runs_to_go'] = full_i1['inn1_total'] - full_i1['cum_runs_before']
# Compute empirical expected runs per ball remaining as function of over & wickets_rem
full_i1['over_bin'] = ((full_i1['legal_balls_before'] // 6) + 1).clip(1, 20)
over_rpo = full_i1.groupby('over_bin').apply(lambda g: 6.0 * g['tot_runs'].sum() / max(g['is_legal'].sum(), 1))
print('Empirical RPO by Over (1..20) across <=2025 I1:')
print(over_rpo.round(2).to_dict())

# Also fit a clean Ridge / Monotonic HGBR for expected remaining runs in I1: E[runs_to_go | balls_rem, wickets_rem, env_par_rpo, cum_runs_before]
# Wait: in a pure resource model, runs_to_go depends on balls_rem, wickets_rem, env_par_rpo (and slightly current pace / team quality via crr!)
res_feats = ['balls_rem', 'wickets_rem', 'env_par_rpo']
# Note: monotonic_cst in HistGradientBoostingRegressor: +1 for balls_rem, +1 for wickets_rem, +1 for env_par_rpo!
hgb_rem = HistGradientBoostingRegressor(
    max_iter=120, max_depth=4, min_samples_leaf=500, l2_regularization=20.0,
    monotonic_cst=[1, 1, 1], random_state=42
)
hgb_rem.fit(full_i1[res_feats], full_i1['runs_to_go'])

# Evaluate expected remaining runs on 2026 full I1
full_i1_26 = reg[(reg['innings'] == 1) & (reg['max_balls'] == 120) & (reg['year'] == 2026) & (reg['has_valid_outcome'])].copy()
full_i1_26['runs_to_go'] = full_i1_26['inn1_total'] - full_i1_26['cum_runs_before']
pred_rem_26 = hgb_rem.predict(full_i1_26[res_feats])
print(f'I1 Remaining Runs MAE (2026 test): {mean_absolute_error(full_i1_26["runs_to_go"], pred_rem_26):.2f} runs, R2: {r2_score(full_i1_26["runs_to_go"], pred_rem_26):.3f}')

# Compute projected total & chase surplus across ALL valid balls!
for df_sub in [reg]:
    df_sub['exp_rem_runs'] = hgb_rem.predict(df_sub[['balls_rem', 'wickets_rem', 'env_par_rpo']].fillna(0))
    # Ensure when wickets_rem == 0, exp_rem_runs = 0
    df_sub.loc[df_sub['wickets_rem'] == 0, 'exp_rem_runs'] = 0.0

# --- INNINGS 2 WP MODELS ---
i2 = reg[(reg['innings'] == 2) & reg['has_valid_outcome']].copy()
i2['runs_needed'] = (i2['target'] - i2['cum_runs_before']).clip(lower=0)
i2['rrr'] = 6.0 * i2['runs_needed'] / i2['balls_rem']
i2['chase_surplus'] = i2['exp_rem_runs'] - i2['runs_needed']
i2['log_rrr_ratio'] = np.log((i2['rrr'] + 1.0) / (i2['env_par_rpo'] + 1.0))
i2['progress'] = i2['legal_balls_before'] / i2['max_balls']
i2['balls_sqrt'] = np.sqrt(i2['balls_rem'])
i2['req_per_wkt'] = i2['runs_needed'] / i2['wickets_rem'].clip(lower=0.5)
i2['wkt_x_balls'] = i2['wickets_rem'] * i2['progress']
i2['surplus_std'] = i2['chase_surplus'] / np.sqrt(i2['balls_rem'] + 6.0)

feat2_eng = ['chase_surplus', 'surplus_std', 'log_rrr_ratio', 'runs_needed', 'balls_rem',
             'wickets_rem', 'env_par_rpo', 'progress', 'wkt_x_balls']
# Monotonic signs for HGB2: chase_surplus(+), surplus_std(+), log_rrr_ratio(-), runs_needed(-), balls_rem(0), wickets_rem(+), env_par_rpo(0), progress(0), wkt_x_balls(0)
mono2 = [1, 1, -1, -1, 0, 1, 0, 0, 0]

tr2 = i2[i2['year'] <= 2025]
te2 = i2[i2['year'] == 2026]

models2 = {
    'LogReg_Eng_C0.1': Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=0.1))]),
    'LogReg_Eng_C1.0': Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=1.0))]),
    'HGB_Mono_D3_Leaf1000': HistGradientBoostingClassifier(max_iter=120, max_depth=3, min_samples_leaf=1000, l2_regularization=50.0, monotonic_cst=mono2, random_state=42),
    'HGB_Mono_D2_Leaf1500': HistGradientBoostingClassifier(max_iter=120, max_depth=2, min_samples_leaf=1500, l2_regularization=100.0, monotonic_cst=mono2, random_state=42),
}

print('\n=== INNINGS 2 WIN PROBABILITY COMPARISON (Chronological 2026 Test + 5-Fold GroupKFold) ===')
gkf = GroupKFold(n_splits=5)
for name, mdl in models2.items():
    # Chronological 2026 test
    mdl.fit(tr2[feat2_eng], tr2['bat_won'])
    p26 = mdl.predict_proba(te2[feat2_eng])[:, 1]
    # 5-fold GroupKFold across all i2
    oof = np.zeros(len(i2))
    for tr_idx, va_idx in gkf.split(i2, i2['bat_won'], groups=i2['match_id']):
        mdl.fit(i2.iloc[tr_idx][feat2_eng], i2.iloc[tr_idx]['bat_won'])
        oof[va_idx] = mdl.predict_proba(i2.iloc[va_idx][feat2_eng])[:, 1]
    print(f'{name:22s} | 2026 Test: LL={log_loss(te2["bat_won"], p26):.4f}, Brier={brier_score_loss(te2["bat_won"], p26):.4f}, AUC={roc_auc_score(te2["bat_won"], p26):.4f} '
          f'| 5-Fold GKF: LL={log_loss(i2["bat_won"], oof):.4f}, Brier={brier_score_loss(i2["bat_won"], oof):.4f}, AUC={roc_auc_score(i2["bat_won"], oof):.4f}')

# --- INNINGS 1 WP MODELS ---
i1 = reg[(reg['innings'] == 1) & reg['has_valid_outcome']].copy()
i1['proj_total'] = i1['cum_runs_before'] + i1['exp_rem_runs']
# Expected match 1st-innings par total = env_par_rpo * (max_balls / 6)
i1['par_total'] = i1['env_par_rpo'] * (i1['max_balls'] / 6.0)
i1['proj_surplus'] = i1['proj_total'] - i1['par_total']
i1['progress'] = i1['legal_balls_before'] / i1['max_balls']
i1['proj_surplus_x_prog'] = i1['proj_surplus'] * (0.3 + 0.7 * i1['progress'])
i1['is_ipl'] = (i1['match_tier'] == 'IPL').astype(int)

feat1_eng = ['proj_surplus', 'proj_surplus_x_prog', 'cum_runs_before', 'wickets_rem', 'balls_rem', 'progress', 'env_par_rpo', 'is_ipl']
mono1 = [1, 1, 1, 1, 0, 0, 0, 0]

tr1 = i1[i1['year'] <= 2025]
te1 = i1[i1['year'] == 2026]

models1 = {
    'LogReg_Eng_C0.1': Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=0.1))]),
    'LogReg_Eng_C1.0': Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=1.0))]),
    'HGB_Mono_D2_Leaf1500': HistGradientBoostingClassifier(max_iter=100, max_depth=2, min_samples_leaf=1500, l2_regularization=100.0, monotonic_cst=mono1, random_state=42),
}

print('\n=== INNINGS 1 WIN PROBABILITY COMPARISON (Chronological 2026 Test + 5-Fold GroupKFold) ===')
for name, mdl in models1.items():
    mdl.fit(tr1[feat1_eng], tr1['bat_won'])
    p26 = mdl.predict_proba(te1[feat1_eng])[:, 1]
    oof = np.zeros(len(i1))
    for tr_idx, va_idx in gkf.split(i1, i1['bat_won'], groups=i1['match_id']):
        mdl.fit(i1.iloc[tr_idx][feat1_eng], i1.iloc[tr_idx]['bat_won'])
        oof[va_idx] = mdl.predict_proba(i1.iloc[va_idx][feat1_eng])[:, 1]
    print(f'{name:22s} | 2026 Test: LL={log_loss(te1["bat_won"], p26):.4f}, Brier={brier_score_loss(te1["bat_won"], p26):.4f}, AUC={roc_auc_score(te1["bat_won"], p26):.4f} '
          f'| 5-Fold GKF: LL={log_loss(i1["bat_won"], oof):.4f}, Brier={brier_score_loss(i1["bat_won"], oof):.4f}, AUC={roc_auc_score(i1["bat_won"], oof):.4f}')
