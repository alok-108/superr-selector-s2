import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\data_pack\Data Pack'
OUT = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\outputs'

comp = pd.read_csv(f'{BASE}/competitions.csv')
matches = pd.read_csv(f'{BASE}/matches.csv')
bbb = pd.read_csv(f'{BASE}/ball_by_ball.csv', low_memory=False)
teams = pd.read_csv(f'{BASE}/teams.csv')
p11 = pd.read_csv(f'{BASE}/playing_eleven.csv')

# 1. Match metadata & tier classification
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
m['match_date_dt'] = pd.to_datetime(m['match_date'])
m['year'] = m['match_date_dt'].dt.year

# 2. Clean ball-by-ball regular innings (1 and 2)
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
reg['is_bowler_wkt'] = reg['is_wicket'] & (~reg['dismissal_type'].isin(non_bowler_dismissals))

# Cumulative BEFORE delivery
g = reg.groupby(['match_id', 'innings'], sort=False)
reg['cum_runs_before'] = g['tot_runs'].cumsum() - reg['tot_runs']
reg['cum_wkts_before'] = g['is_wicket'].cumsum() - reg['is_wicket'].astype(int)
reg['legal_balls_before'] = g['is_legal'].cumsum() - reg['is_legal'].astype(int)

# Recent 18-ball momentum (approx last 18 deliveries in row order, or exact 18 rows ~ 3 overs)
reg['runs_last18'] = g['tot_runs'].transform(lambda s: s.shift(1, fill_value=0).rolling(18, min_periods=1).sum())
reg['wkts_last18'] = g['is_wicket'].transform(lambda s: s.astype(int).shift(1, fill_value=0).rolling(18, min_periods=1).sum())

# Max balls & target per match
i1_totals = reg[reg['innings'] == 1].groupby('match_id')['tot_runs'].sum().rename('inn1_total')
i1_balls = reg[reg['innings'] == 1].groupby('match_id')['is_legal'].sum().rename('inn1_legal_balls')
m = m.merge(i1_totals, on='match_id', how='left').merge(i1_balls, on='match_id', how='left')

m['max_balls_1'] = m['first_inn_revised_overs'].fillna(120).astype(int)
m['max_balls_2'] = m['second_inn_revised_overs'].fillna(120).astype(int)
m['target'] = np.where(m['revised_target'].notna(), m['revised_target'], m['inn1_total'] + 1)

# Merge match info onto reg
reg = reg.merge(m[['match_id','comp_id','comp_name','match_format','season','match_date','match_date_dt','year',
                   'stage','knock_out','venue_id','venue_country','match_tier',
                   'max_balls_1','max_balls_2','target','inn1_total','win_team_id','result']],
                on='match_id', how='left')

reg['max_balls'] = np.where(reg['innings'] == 1, reg['max_balls_1'], reg['max_balls_2'])
reg['balls_rem'] = (reg['max_balls'] - reg['legal_balls_before']).clip(lower=1)
reg['wickets_rem'] = (10 - reg['cum_wkts_before']).clip(lower=0)
reg['crr'] = np.where(reg['legal_balls_before'] > 0, 6.0 * reg['cum_runs_before'] / reg['legal_balls_before'], 0.0)

# Out-of-fold / shrunk environment par RPO per match (using only OTHER matches to avoid target leakage!)
match_rpo = reg.groupby(['match_id','match_tier','season','venue_id']).agg(
    m_runs=('tot_runs','sum'), m_balls=('is_legal','sum')
).reset_index()
match_rpo['m_rpo'] = 6.0 * match_rpo['m_runs'] / match_rpo['m_balls'].clip(lower=1)

# Tier-season baseline (leave-one-match-out)
ts_sum = match_rpo.groupby(['match_tier','season'])[['m_runs','m_balls']].transform('sum')
ts_loo_rpo = 6.0 * (ts_sum['m_runs'] - match_rpo['m_runs']) / (ts_sum['m_balls'] - match_rpo['m_balls']).clip(lower=1)
# Fallback to tier overall if single match in tier-season
t_sum = match_rpo.groupby('match_tier')[['m_runs','m_balls']].transform('sum')
t_loo_rpo = 6.0 * (t_sum['m_runs'] - match_rpo['m_runs']) / (t_sum['m_balls'] - match_rpo['m_balls']).clip(lower=1)
ts_loo_rpo = ts_loo_rpo.fillna(t_loo_rpo)

# Venue leave-one-match-out shrunk toward tier-season baseline (prior weight = 6 matches ~ 1440 balls)
v_sum = match_rpo.groupby('venue_id')[['m_runs','m_balls']].transform('sum')
v_runs_loo = v_sum['m_runs'] - match_rpo['m_runs']
v_balls_loo = v_sum['m_balls'] - match_rpo['m_balls']
prior_balls = 1440.0
match_rpo['env_par_rpo'] = 6.0 * (v_runs_loo + prior_balls * (ts_loo_rpo / 6.0)) / (v_balls_loo + prior_balls)
reg = reg.merge(match_rpo[['match_id','env_par_rpo']], on='match_id', how='left')

# Target label: did batting team win?
# Exclude matches without win_team_id or DLS-curtailed mid-innings without revised_target (the 4 ambiguous DLS games)
dls_no_rev = reg['result'].str.contains('D/L|DLS', case=False, na=False) & m.set_index('match_id').loc[reg['match_id'], 'revised_target'].isna().values
# Wait simpler: merge revised_target flag
reg['has_valid_outcome'] = reg['win_team_id'].notna() & (~(reg['result'].str.contains('D/L|DLS', case=False, na=False) & reg['match_id'].isin(m[m['revised_target'].isna()]['match_id'])))
reg['bat_won'] = (reg['batting_team_id'] == reg['win_team_id']).astype(float)
# For tied super-over matches, at end of regulation innings scores tied -> keep actual win_team_id for WP or 0.5?
is_so = reg['result'].str.contains('Super Over|Tied', case=False, na=False)
print('Super over / tied balls:', is_so.sum(), 'Valid outcome balls:', reg['has_valid_outcome'].sum())

# 3. Train & Compare Win Probability Models separately for Innings 1 and Innings 2
# First check Innings 2
i2 = reg[(reg['innings'] == 2) & reg['has_valid_outcome']].copy()
i2['runs_needed'] = (i2['target'] - i2['cum_runs_before']).clip(lower=0)
i2['rrr'] = 6.0 * i2['runs_needed'] / i2['balls_rem']
i2['rrr_vs_par'] = i2['rrr'] - i2['env_par_rpo']
i2['rrr_vs_crr'] = i2['rrr'] - i2['crr']
i2['progress'] = i2['legal_balls_before'] / i2['max_balls']
i2['runs_per_wkt'] = i2['runs_needed'] / i2['wickets_rem'].clip(lower=1)
i2['target_vs_par'] = (i2['target'] / (i2['max_balls'] / 6.0)) - i2['env_par_rpo']

feat2 = ['runs_needed','balls_rem','wickets_rem','rrr','crr','rrr_vs_par','rrr_vs_crr',
         'progress','runs_per_wkt','env_par_rpo','target_vs_par','runs_last18','wkts_last18']

# Chronological split: Train <= 2025, Test = 2026, PLUS 5-fold GroupKFold by match_id
train2 = i2[i2['year'] <= 2025]
test2 = i2[i2['year'] == 2026]
print(f'\nInnings 2 rows: total={len(i2)}, train<=2025={len(train2)} ({train2["match_id"].nunique()} matches), test2026={len(test2)} ({test2["match_id"].nunique()} matches)')

lr2 = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=1.0))])
hgb2 = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05, min_samples_leaf=200, l2_regularization=5.0, random_state=42)

for name, model in [('LogReg_Baseline', lr2), ('HistGB_Depth4', hgb2)]:
    model.fit(train2[feat2], train2['bat_won'])
    p_tr = model.predict_proba(train2[feat2])[:, 1]
    p_te = model.predict_proba(test2[feat2])[:, 1]
    print(f'I2 {name:15s} | Train LogLoss={log_loss(train2["bat_won"], p_tr):.4f}, AUC={roc_auc_score(train2["bat_won"], p_tr):.4f} | '
          f'2026 Test LogLoss={log_loss(test2["bat_won"], p_te):.4f}, Brier={brier_score_loss(test2["bat_won"], p_te):.4f}, AUC={roc_auc_score(test2["bat_won"], p_te):.4f}')

# Innings 1 features
i1 = reg[(reg['innings'] == 1) & reg['has_valid_outcome']].copy()
i1['progress'] = i1['legal_balls_before'] / i1['max_balls']
i1['par_runs_so_far'] = i1['env_par_rpo'] * (i1['legal_balls_before'] / 6.0)
i1['runs_vs_par'] = i1['cum_runs_before'] - i1['par_runs_so_far']
i1['rpo_vs_par'] = np.where(i1['legal_balls_before'] >= 6, i1['crr'] - i1['env_par_rpo'], 0.0)
i1['is_ipl'] = (i1['match_tier'] == 'IPL').astype(int)

feat1 = ['cum_runs_before','balls_rem','wickets_rem','crr','progress','env_par_rpo',
         'runs_vs_par','rpo_vs_par','runs_last18','wkts_last18','is_ipl']

train1 = i1[i1['year'] <= 2025]
test1 = i1[i1['year'] == 2026]
print(f'\nInnings 1 rows: total={len(i1)}, train<=2025={len(train1)} ({train1["match_id"].nunique()} matches), test2026={len(test1)} ({test1["match_id"].nunique()} matches)')

lr1 = Pipeline([('sc', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, C=1.0))])
hgb1 = HistGradientBoostingClassifier(max_iter=150, max_depth=3, learning_rate=0.05, min_samples_leaf=300, l2_regularization=10.0, random_state=42)

for name, model in [('LogReg_Baseline', lr1), ('HistGB_Depth3', hgb1)]:
    model.fit(train1[feat1], train1['bat_won'])
    p_tr = model.predict_proba(train1[feat1])[:, 1]
    p_te = model.predict_proba(test1[feat1])[:, 1]
    print(f'I1 {name:15s} | Train LogLoss={log_loss(train1["bat_won"], p_tr):.4f}, AUC={roc_auc_score(train1["bat_won"], p_tr):.4f} | '
          f'2026 Test LogLoss={log_loss(test1["bat_won"], p_te):.4f}, Brier={brier_score_loss(test1["bat_won"], p_te):.4f}, AUC={roc_auc_score(test1["bat_won"], p_te):.4f}')
