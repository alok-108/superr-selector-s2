import os
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
ipl_b = pd.read_csv(f'{OUT}/ipl_only_top15_batters.csv')
ipl_bw = pd.read_csv(f'{OUT}/ipl_only_top15_bowlers.csv')

print("=== 1. TRADE & KEY PLAYER CHECK IN FULL DATA PACK ===")
check_names = [
    'Mohammed Shami', 'Siddharth', 'Sam Curran', 'Nathan Ellis', 'Khaleel Ahmed',
    'Matheesha Pathirana', 'Ravi Bishnoi', 'Jofra Archer', 'Nandre Burger',
    'Sandeep Sharma', 'Tushar Deshpande', 'Harpreet Brar', 'Ashutosh Sharma',
    'Hardik Pandya', 'Arshdeep Singh', 'Varun Chakravarthy', 'Abhishek Sharma',
    'Naman Dhir', 'Digvesh', 'Vipraj', 'Rinku Singh', 'Tristan Stubbs', 'Shimron Hetmyer'
]
for nm in check_names:
    s1 = ebw[ebw['bowler_name'].str.contains(nm, case=False, na=False)]
    s2 = eb[eb['striker_name'].str.contains(nm, case=False, na=False)]
    if len(s1):
        for _, r in s1.iterrows():
            print(f"[BOWL Rank #{int(r['rank']):2d}] {r['bowler_name']:22s} ({r['ipl_2026_team']}) | {int(r['hp_leg']):3d} HPb, {r['hp_econ']:5.2f} Econ, {r['blended_hp_tva']:+5.1f} HP TVA, {r['lp_tva_eb']:+5.1f} LP, {r['clutch_resid']:+5.1f} CR, PRS {r['PRS_100']:.1f}")
    if len(s2):
        for _, r in s2.iterrows():
            print(f"[BAT  Rank #{int(r['rank']):2d}] {r['striker_name']:22s} ({r['ipl_2026_team']}) | {int(r['hp_b']):3d} HPb, {r['hp_sr']:5.1f} SR,   {r['blended_hp_tva']:+5.1f} HP TVA, {r['lp_tva_eb']:+5.1f} LP, {r['clutch_resid']:+5.1f} CR, PRS {r['PRS_100']:.1f}")
    if len(s1)==0 and len(s2)==0:
        rb = bbb[bbb['striker_name'].str.contains(nm, case=False, na=False)]
        rbw = bbb[bbb['bowler_name'].str.contains(nm, case=False, na=False)]
        print(f"[RAW ONLY] {nm:22s} | bat_balls={len(rb)}, bowl_balls={len(rbw)}")

print("\n=== 2. BATTERS RANK #11-#25 ===")
print(eb.iloc[10:25][['rank','striker_name','ipl_2026_team','hp_b','hp_m','hp_sr','blended_hp_tva','lp_tva_eb','clutch_resid','PRS_100']].round(2).to_string(index=False))

print("\n=== 3. BOWLERS RANK #11-#25 ===")
print(ebw.iloc[10:25][['rank','bowler_name','ipl_2026_team','hp_leg','hp_m','hp_econ','hp_wkts','blended_hp_tva','lp_tva_eb','clutch_resid','PRS_100']].round(2).to_string(index=False))

print("\n=== 4. RR HOME VENUES & IPL VENUE ANALYSIS ===")
m = matches.merge(comp, on='comp_id', how='left')
ipl_m = m[m['comp_name'].str.startswith('IPL')].copy()
rr_home = ipl_m[ipl_m['home_team_id']==3]
print("RR home matches by venue_id x season:")
print(rr_home.groupby(['venue_id','season']).size().unstack(fill_value=0))
for vid in rr_home['venue_id'].unique():
    sub_v = ipl_m[ipl_m['venue_id']==vid]
    print(f"Venue {vid}: total IPL matches={len(sub_v)}, dates sample={list(sub_v['match_date'].head(3))}, home teams={sub_v['home_team'].value_counts().to_dict()}")
    vb = bbb[bbb['match_id'].isin(sub_v['match_id']) & bbb['innings'].isin([1,2])].copy()
    vb['leg'] = ~vb['extras_type'].isin(['wide','Overstep','other'])
    vb['tot_r'] = vb['runs'] + vb['extras1_runs'] + vb['extras_2_runs']
    ps = vb.groupby('pace_or_spin').agg(balls=('runs','size'), leg=('leg','sum'), r=('tot_r','sum'), w=('is_wicket','sum')).assign(rpo=lambda d:6*d.r/d.leg, bpw=lambda d:d.leg/d.w)
    print(ps[['balls','rpo','bpw']].round(2).to_string())
    ch_w = (sub_v['win_team_id'] == sub_v['away_team_id']).mean()
    # Wait: proper chasing win check
    i1t = vb[vb['innings']==1].groupby('match_id')['batting_team_id'].first()
    sub_v = sub_v[sub_v['win_team_id'].notna()].copy()
    sub_v['bat1'] = sub_v['match_id'].map(i1t)
    print(f"  Venue {vid} Chase Win % = {100*(sub_v['win_team_id']!=sub_v['bat1']).mean():.1f}% ({len(sub_v)} valid IPL games)")

print("\n=== 5. PLAYING ELEVEN ROLE & IMPACT PLAYER / CAPTAIN AUDIT ===")
print(p11[['is_captain','is_wk','is_impact','is_impact_off']].sum())
