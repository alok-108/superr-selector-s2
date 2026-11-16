import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\data_pack\Data Pack'
OUT = r'C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\outputs'

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

# Batter balls faced in this innings prior to delivery (for entry/set-ness diagnostics)
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

# Resource model + fast LUT
gkf = GroupKFold(n_splits=5)
res_feats = ['balls_rem', 'wickets_rem', 'env_par_rpo']
valid_i1_full = reg[(reg['innings']==1)&(reg['max_balls']==120)&(reg['has_valid_outcome'])].copy()
valid_i1_full['runs_to_go'] = valid_i1_full['inn1_total'] - valid_i1_full['cum_runs_before']
mdl_rem = HistGradientBoostingRegressor(
    max_iter=100, max_depth=4, min_samples_leaf=500, l2_regularization=20.0,
    monotonic_cst=[1, 1, 1], random_state=42
)
mdl_rem.fit(valid_i1_full[res_feats], valid_i1_full['runs_to_go'])

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
            build_i2_feats(cr_c, lbb_c, br_c, wr_c, mb, env_r, tgt, er_c), er_c)

X1_0, X2_0, er_0 = make_cf(0, 0)
X1_1, X2_1, er_1 = make_cf(1, 0)
X1_4, X2_4, er_4 = make_cf(4, 0)
X1_6, X2_6, er_6 = make_cf(6, 0)
X1_w, X2_w, er_w = make_cf(0, 1)

cr_a = cum_r + reg['tot_runs'].values
lbb_a = np.minimum(mb, lbb + reg['is_legal'].astype(int).values)
br_a = np.maximum(1, mb - lbb_a)
wr_a = np.maximum(0, wr - reg['is_wicket'].astype(int).values)
er_a = fast_rem_lookup(br_a, wr_a, env_r)
X1_a = build_i1_feats(cr_a, lbb_a, br_a, wr_a, mb, env_r, is_ipl_arr, er_a)
X2_a = build_i2_feats(cr_a, lbb_a, br_a, wr_a, mb, env_r, tgt, er_a)

wp_b = np.zeros(len(reg)); wp_a = np.zeros(len(reg))
wp_0 = np.zeros(len(reg)); wp_1 = np.zeros(len(reg)); wp_4 = np.zeros(len(reg)); wp_6 = np.zeros(len(reg)); wp_w = np.zeros(len(reg))

folds = list(gkf.split(reg, groups=reg['match_id']))
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

proj_0 = cum_r + er_0; proj_1 = cum_r + 1 + er_1; proj_4 = cum_r + 4 + er_4; proj_6 = cum_r + 6 + er_6; proj_w = cum_r + er_w
mu_proj = w0*proj_0 + w1*proj_1 + w4*proj_4 + w6*proj_6 + ww*proj_w
reg['proj_lev'] = np.sqrt(w0*(proj_0-mu_proj)**2 + w1*(proj_1-mu_proj)**2 + w4*(proj_4-mu_proj)**2 + w6*(proj_6-mu_proj)**2 + ww*(proj_w-mu_proj)**2)

# --- REFINED, CRICKET-CALIBRATED PRESSURE INDEX ---
# Contest Liveness gate:
# - In Innings 2: 0 if dead loss (wp < 0.04) or easy win (wp > 0.96); rises steeply once wp is in [0.06, 0.94]
# - In Innings 1: match is NEVER dead early, only mildly modulated if massive blowout (wp < 0.10 or > 0.90)
reg['u_live'] = 4.0 * reg['wp_before'] * (1.0 - reg['wp_before'])
live_gate = np.where(
    reg['innings'] == 2,
    np.clip((np.minimum(reg['wp_before'], 1.0 - reg['wp_before']) - 0.03) / 0.15, 0.0, 1.0),
    np.clip((np.minimum(reg['wp_before'], 1.0 - reg['wp_before']) - 0.05) / 0.15, 0.25, 1.0)
)

# Component 1: Win-Probability Leverage (0..1)
# In I2: wp_lev ~ 0.025 normal, 0.050 tense, 0.080+ cliffhanger
# In I1: combine wp_lev and proj_lev*live_gate so routine over-1 doesn't score higher than death overs!
# Specifically, in I1 over 1-6 comfortable: wp_lev ~ 0.023, proj_lev ~ 3.1; over 17-20: proj_lev ~ 2.3 (Wait! why was proj_lev 3.1 in PP vs 2.3 in Death? Because losing a wicket in PP costs 20 projected runs, whereas in over 19 it costs 5!)
# AHA! Look closely: in I1, proj_lev is highest when wicket value is highest (PP), while wp_lev in I1 death is slightly lower ONLY because we didn't include endgame scoring pressure!
# So for I1, let's separate:
# - Normal I1 base leverage (calibrated smoothly around ~0.25-0.35 in ordinary overs)
# - Death-overs push (over 16-20)
# - Collapse shock (excess wickets fallen relative to expected pace!)
lev_score = np.where(
    reg['innings'] == 2,
    np.tanh(reg['wp_lev'] / 0.052),
    0.35 * np.tanh(reg['wp_lev'] / 0.028) * live_gate
)

# Component 2: Collapse / Rebuild Stress (Batting crisis!)
# Expected wickets at legal_balls_before ~ 0.048 * legal_balls_before (plus 0.3 baseline)
exp_wkts = 0.045 * reg['legal_balls_before']
collapse_excess = np.maximum(0.0, reg['cum_wkts_before'] - exp_wkts)
# Require >= 2 wickets down AND live_gate > 0 (not dead chase at 70/8 needing 140 off 12!)
collapse_stress = np.where(reg['cum_wkts_before'] >= 2, np.tanh(collapse_excess / 1.8), 0.0) * live_gate

# Component 3: Required Run Rate / Chase Escalation Stress (Innings 2 only, live gate!)
reg['rrr'] = np.where(reg['innings']==2, 6.0 * np.maximum(0, reg['target'] - reg['cum_runs_before']) / reg['balls_rem'], 0.0)
rrr_excess = np.maximum(0.0, reg['rrr'] - reg['env_par_rpo'])
rrr_stress = np.where(reg['innings']==2, np.tanh(rrr_excess / 4.5) * live_gate, 0.0)

# Component 4: Death / Endgame Urgency (overs 16-20, scaled smoothly from over 16=0.55 to over 20=1.0, gated by live_gate)
death_ramp = np.where(reg['overid'] >= 16, 0.50 + 0.50 * (reg['overid'] - 16) / 4.0, 0.0) * live_gate

# Component 5: Recent Momentum Shock (bowling under assault: runs_last18 >= 30 in live game; batting after twin-wicket shock: wkts_last18 >= 2)
bat_shock = np.tanh(np.maximum(0.0, reg['wkts_last18'] - 1) / 1.5) * live_gate
bowl_surge = np.tanh(np.maximum(0.0, reg['runs_last18'] - 26.0) / 16.0) * live_gate

# Component 6: Knockout / Playoff Stakes (+10 pts when live)
ko_boost = np.where(reg['is_knockout'] == 1, 0.12, 0.0) * np.maximum(0.3, live_gate)

# Build PI_bat and PI_bowl and PI_shared cleanly on 0..100 scale!
# Note: Use max/soft-union or weighted sum so ANY true crisis (e.g. I1 collapse 25/3 OR I2 death thriller OR I1 competitive death) crosses 60+!
# Wait: look at a weighted sum + crisis floor:
pi_bat_raw = (
    0.45 * lev_score +
    0.32 * np.maximum(collapse_stress, 0.5 * bat_shock) +
    0.25 * rrr_stress +
    0.22 * death_ramp +
    ko_boost
)
# Bowling pressure: defending in live I2 (especially death or when opponent has >=30% WP or surging), or I1 death / surge containment
bowl_defense_risk = np.where(
    reg['innings'] == 2,
    np.clip((reg['wp_before'] - 0.12) / 0.55, 0.0, 1.0) * live_gate,
    0.25 * live_gate
)
pi_bowl_raw = (
    0.48 * lev_score +
    0.30 * death_ramp +
    0.20 * bowl_defense_risk +
    0.15 * bowl_surge +
    ko_boost
)
pi_shared_raw = 0.5 * pi_bat_raw + 0.5 * pi_bowl_raw

reg['PI_bat'] = 100.0 * np.clip(pi_bat_raw, 0.0, 1.0)
reg['PI_bowl'] = 100.0 * np.clip(pi_bowl_raw, 0.0, 1.0)
reg['PI_shared'] = 100.0 * np.clip(pi_shared_raw, 0.0, 1.0)

print('=== Refined PI_bat & PI_bowl quantiles by Innings & Phase ===')
for col in ['PI_bat', 'PI_bowl']:
    print(f'-- {col} --')
    print(reg.groupby(['innings','phase'])[col].describe(percentiles=[0.5, 0.75, 0.85, 0.9]).round(1).to_string())

print('\n=== Situation Validity Check (Refined PI) ===')
situations = {
    '1. I2 Death Close (ov17-20, 0.2<=WP<=0.8)': (reg['innings']==2)&(reg['overid']>=17)&(reg['wp_before'].between(0.2,0.8)),
    '2. I2 Dead Chase (ov17-20, WP<0.03)': (reg['innings']==2)&(reg['overid']>=17)&(reg['wp_before']<0.03),
    '3. I2 Cruise Chase (ov17-20, WP>0.95)': (reg['innings']==2)&(reg['overid']>=17)&(reg['wp_before']>0.95),
    '4. Early Collapse (ov1-8, wkts>=3, WP>=0.08)': (reg['overid']<=8)&(reg['cum_wkts_before']>=3)&(reg['wp_before']>=0.08),
    '5. Comfortable PP (ov1-6, wkts<=1, I1)': (reg['innings']==1)&(reg['overid']<=6)&(reg['cum_wkts_before']<=1),
    '6. Routine Middle (ov7-15, wkts<=2, I1)': (reg['innings']==1)&(reg['overid'].between(7,15))&(reg['cum_wkts_before']<=2),
    '7. I1 Death Competitive (ov17-20, 0.25<=WP<=0.75)': (reg['innings']==1)&(reg['overid']>=17)&(reg['wp_before'].between(0.25,0.75)),
    '8. Knockout Playoffs (all balls)': (reg['is_knockout']==1),
}
for label, mask in situations.items():
    sub = reg[mask]
    print(f'{label:48s} | n={len(sub):6d} | PI_bat={sub["PI_bat"].mean():4.1f} | PI_bowl={sub["PI_bowl"].mean():4.1f}')

# Context-Adjusted Expected Runs (xR) & Expected Wickets (xW) for BOTH Batters & Bowlers
print('\n=== Fitting OOF Context-Adjusted xR & xW Baselines ===')
reg['exp_bat_runs'] = np.nan
reg['exp_bat_out'] = np.nan
reg['exp_bowl_runs'] = np.nan
reg['exp_bowl_wkt'] = np.nan

feat_cols = ['match_tier','pace_or_spin','innings','env_par_rpo','overid','wickets_rem']
X_ctx = reg[feat_cols].copy()
for c in ['match_tier','pace_or_spin']:
    X_ctx[c] = X_ctx[c].astype('category')

for tr_idx, va_idx in folds:
    # Batting baseline trained on is_batter_ball
    tr_b = tr_idx[reg.iloc[tr_idx]['is_batter_ball'].values]
    va_b = va_idx[reg.iloc[va_idx]['is_batter_ball'].values]
    
    mr_bat = HistGradientBoostingRegressor(max_iter=80, max_depth=4, min_samples_leaf=400, categorical_features=['match_tier','pace_or_spin'], random_state=42)
    mr_bat.fit(X_ctx.iloc[tr_b], reg.iloc[tr_b]['batter_runs'])
    reg.iloc[va_b, reg.columns.get_loc('exp_bat_runs')] = mr_bat.predict(X_ctx.iloc[va_b])
    
    mw_bat = HistGradientBoostingClassifier(max_iter=80, max_depth=4, min_samples_leaf=400, categorical_features=['match_tier','pace_or_spin'], random_state=42)
    mw_bat.fit(X_ctx.iloc[tr_b], reg.iloc[tr_b]['is_striker_out'].astype(int))
    reg.iloc[va_b, reg.columns.get_loc('exp_bat_out')] = mw_bat.predict_proba(X_ctx.iloc[va_b])[:, 1]
    
    # Bowling baseline trained on ALL deliveries (so wides/no-balls are part of expected bowler runs per delivery!)
    mr_bowl = HistGradientBoostingRegressor(max_iter=80, max_depth=4, min_samples_leaf=400, categorical_features=['match_tier','pace_or_spin'], random_state=42)
    mr_bowl.fit(X_ctx.iloc[tr_idx], reg.iloc[tr_idx]['bowler_runs'])
    reg.iloc[va_idx, reg.columns.get_loc('exp_bowl_runs')] = mr_bowl.predict(X_ctx.iloc[va_idx])
    
    mw_bowl = HistGradientBoostingClassifier(max_iter=80, max_depth=4, min_samples_leaf=400, categorical_features=['match_tier','pace_or_spin'], random_state=42)
    mw_bowl.fit(X_ctx.iloc[tr_idx], reg.iloc[tr_idx]['is_bowler_wkt'].astype(int))
    reg.iloc[va_idx, reg.columns.get_loc('exp_bowl_wkt')] = mw_bowl.predict_proba(X_ctx.iloc[va_idx])[:, 1]

b_sub = reg[reg['is_batter_ball']]
print(f"Bat OOF xR={b_sub['exp_bat_runs'].mean():.4f} (actual={b_sub['batter_runs'].mean():.4f}), xOut={b_sub['exp_bat_out'].mean():.4f} (actual={b_sub['is_striker_out'].mean():.4f})")
print(f"Bowl OOF xR={reg['exp_bowl_runs'].mean():.4f} (actual={reg['bowler_runs'].mean():.4f}), xWkt={reg['exp_bowl_wkt'].mean():.4f} (actual={reg['is_bowler_wkt'].mean():.4f})")

reg.to_pickle(f'{OUT}/enriched_balls.pkl')
print('Saved outputs/enriched_balls.pkl successfully!')
