import pandas as pd

OUT = r"C:\Users\valok\.gemini\antigravity\scratch\superr_selector_s2\outputs"

eb = pd.read_csv(f"{OUT}/eligible_batters.csv")
ebw = pd.read_csv(f"{OUT}/eligible_bowlers.csv")

m_b = eb[
    ["striker_id", "striker_name", "rank", "PRS_100", "hp_b", "hp_m", "blended_hp_tva", "hp_wpa_eb", "ipl_2026_team"]
].rename(
    columns={
        "striker_id": "pid",
        "striker_name": "name",
        "rank": "rank_bat",
        "PRS_100": "prs_bat",
        "blended_hp_tva": "tva_bat",
        "hp_wpa_eb": "wpa_bat",
        "hp_m": "m_bat",
    }
)

m_bw = ebw[
    ["bowler_id", "bowler_name", "rank", "PRS_100", "hp_leg", "hp_m", "blended_hp_tva", "hp_wpa_eb", "ipl_2026_team"]
].rename(
    columns={
        "bowler_id": "pid",
        "bowler_name": "name",
        "rank": "rank_bowl",
        "PRS_100": "prs_bowl",
        "blended_hp_tva": "tva_bowl",
        "hp_wpa_eb": "wpa_bowl",
        "hp_m": "m_bowl",
    }
)

comb = m_b.merge(m_bw, on="pid", how="outer", suffixes=("_b", "_bw"))
comb["name"] = comb["name_b"].fillna(comb["name_bw"])
comb["team"] = comb["ipl_2026_team_b"].fillna(comb["ipl_2026_team_bw"])
comb["best_prs"] = comb[["prs_bat", "prs_bowl"]].max(axis=1)
comb["primary_role"] = comb.apply(
    lambda r: (
        f"Dual (#{int(r['rank_bat'])}B / #{int(r['rank_bowl'])}Bw)"
        if pd.notna(r["rank_bat"]) and pd.notna(r["rank_bowl"])
        else (f"Batter (#{int(r['rank_bat'])})" if pd.notna(r["rank_bat"]) else f"Bowler (#{int(r['rank_bowl'])})")
    ),
    axis=1,
)
comb["net_hp_tva_per_match"] = (
    comb["tva_bat"].fillna(0) / 100.0 * (comb["hp_b"].fillna(0) / comb["m_bat"].fillna(1))
    + comb["tva_bowl"].fillna(0) / 100.0 * (comb["hp_leg"].fillna(0) / comb["m_bowl"].fillna(1))
)

unified = comb.sort_values("best_prs", ascending=False)[
    ["pid", "name", "team", "primary_role", "best_prs", "rank_bat", "prs_bat", "tva_bat", "rank_bowl", "prs_bowl", "tva_bowl", "net_hp_tva_per_match"]
].head(20).reset_index(drop=True)
unified.insert(0, "unified_rank", range(1, len(unified) + 1))
unified.to_csv(f"{OUT}/unified_top20_prs.csv", index=False)

dual = comb[comb["rank_bat"].notna() & comb["rank_bowl"].notna()].sort_values(
    "net_hp_tva_per_match", ascending=False
)[["pid", "name", "team", "rank_bat", "prs_bat", "tva_bat", "rank_bowl", "prs_bowl", "tva_bowl", "net_hp_tva_per_match"]].reset_index(drop=True)
dual.insert(0, "net_rank", range(1, len(dual) + 1))
dual.to_csv(f"{OUT}/dual_allrounder_audit.csv", index=False)

print("Saved unified_top20_prs.csv (rows=%d) & dual_allrounder_audit.csv (rows=%d)" % (len(unified), len(dual)))
