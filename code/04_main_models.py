"""Основные модели: спецификационный путь, DiD-оценки санкционных эффектов,
гетерогенность, сдвиги градиента расстояния, партнёр-специфические эффекты.

Все модели оцениваются на гармонизированной панели data/gravity_panel.csv
(496 пар × 25 лет). Результаты сохраняются в results/.

Запуск:  python code/04_main_models.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import ppml
from gravity_common import COALITION_2014, RESULTS, SUBGROUP_OF
from gravity_models import (fit_model, fmt_cell, load_panel, results_to_wide, save_table, stars)

LABELS = {
    "lgdp_sum": "ln(ВВП_i) + ln(ВВП_j)", "lpcdiff": "|ln(ВВП п/д_i) − ln(ВВП п/д_j)|", "ldist": "ln расстояние",
    "contig": "Общая граница", "comlang_off": "Общий офиц. язык", "smctry": "Бывшая единая страна",
    "eu_ij": "ЕС (обе страны)", "wto_ij": "ВТО (обе страны)", "eaeu_ij": "ЕАЭС/ТС (обе страны)",
    "nato_ij": "НАТО (обе страны)", "brics_ij": "БРИКС (обе страны)",
    "rus_pair": "Пара с Россией", "coal_rus_ij": "Россия × коалиция (уровень)", "ukr_rus_ij": "Россия × Украина (уровень)",
    "sanc14_ij": "SANC14: Россия × коалиция × (t ≥ 2014)", "sanc22_ij": "SANC22: Россия × коалиция × (t ≥ 2022)",
    "kor22_ij": "Россия × Респ. Корея × (t ≥ 2022)", "ukr14_ij": "Россия × Украина × (t ≥ 2014)",
    "ukr22_ij": "Россия × Украина × (t ≥ 2022)",
}

SANC_VARS = ["sanc14_ij", "sanc22_ij", "kor22_ij", "ukr14_ij", "ukr22_ij"]
POLICY = ["eu_ij", "wto_ij", "eaeu_ij", "nato_ij", "brics_ij"]
GEO = ["ldist", "contig", "comlang_off", "smctry"]


def long_rows(name: str, r: ppml.FitResult, extra: dict | None = None) -> pd.DataFrame:
    t = r.table().reset_index().rename(columns={"index": "variable"})
    t.insert(0, "model", name)
    t["nobs"], t["pseudo_r2"], t["clusters"], t["n_params"] = r.nobs, r.pseudo_r2, r.n_clusters, r.n_params_total
    t["dropped_collinear"] = ",".join(r.dropped)
    for k, v in (extra or {}).items():
        t[k] = v
    return t


def contrast(r: ppml.FitResult, weights: dict[str, float]) -> tuple[float, float]:
    """Линейная комбинация коэффициентов и её кластерная ст. ошибка."""
    idx = {n: k for k, n in enumerate(r.names)}
    w = np.zeros(len(r.names))
    for n, v in weights.items():
        w[idx[n]] = v
    return float(w @ r.params), float(np.sqrt(w @ r.vcov @ w))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    d = load_panel()
    all_long = []
    summary: dict = {}

    # ------------------------------------------------------------------
    # Таблица 1. Спецификационный путь
    # ------------------------------------------------------------------
    rep = pd.read_csv(RESULTS / "replication_vkr_ols.csv")
    vkr4 = rep[rep.model == 4].set_index("variable")

    trad = ["lgdp_sum", "lpcdiff"] + GEO + POLICY + ["rus_pair", "coal_rus_ij", "ukr_rus_ij"] + SANC_VARS
    struct_nopair = GEO + POLICY + ["coal_rus_ij", "ukr_rus_ij"] + SANC_VARS
    within = POLICY + SANC_VARS
    models1 = {
        "(2) МНК, ln T, год. ЭФ": fit_model(d, trad, "year", kind="ols"),
        "(3) PPML, год. ЭФ": fit_model(d, trad, "year"),
        "(4) PPML, страновые + год. ЭФ": fit_model(d, trad, "country+year"),
        "(5) PPML, страна×год ЭФ": fit_model(d, struct_nopair, "cy"),
        "(6) PPML, страна×год + парные ЭФ": fit_model(d, within, "cy+pair", dyadic=True),
    }
    for k, r in models1.items():
        all_long.append(long_rows("T1 " + k, r))
    order1 = ["lgdp_sum", "lpcdiff", "ldist", "contig", "comlang_off", "smctry", "eu_ij", "wto_ij", "eaeu_ij",
              "nato_ij", "brics_ij", "rus_pair", "coal_rus_ij", "ukr_rus_ij"] + SANC_VARS
    wide1 = results_to_wide(models1, LABELS, order1)
    # столбец (1) — ВКР, модель 4
    vk_map = {"ldist": "l_dist", "contig": "contig", "eu_ij": "eu_ij", "nato_ij": "nato_ij", "brics_ij": "brics_ij",
              "sanc14_ij": "post2014", "sanc22_ij": "post2022"}
    col1 = []
    for v in wide1["variable"]:
        key = next((k for k, lab in LABELS.items() if lab == v), None)
        if key in vk_map:
            vr = vkr4.loc[vk_map[key]]
            from scipy import stats
            p = 2 * stats.norm.sf(abs(vr.coef / vr.se))
            col1.append(fmt_cell(vr.coef, vr.se, p))
        else:
            col1.append("")
    wide1.insert(1, "(1) ВКР: МНК (модель 4)", col1)
    # добавим строки ВВП на душу населения из ВКР
    extra = pd.DataFrame([
        {"variable": "ln ВВП п/д страны i (ВКР)", "(1) ВКР: МНК (модель 4)": fmt_cell(*[vkr4.loc["l_gdppc_i", c] for c in ("coef", "se")], 0.0)},
        {"variable": "ln ВВП п/д страны j (ВКР)", "(1) ВКР: МНК (модель 4)": fmt_cell(*[vkr4.loc["l_gdppc_j", c] for c in ("coef", "se")], 0.0)},
    ]).fillna("")
    wide1 = pd.concat([wide1.iloc[:2], extra.reindex(columns=wide1.columns, fill_value=""), wide1.iloc[2:]],
                      ignore_index=True)
    wide1.loc[wide1.variable == "N", "(1) ВКР: МНК (модель 4)"] = "12400"
    wide1.loc[wide1.variable == "pseudo R2", "(1) ВКР: МНК (модель 4)"] = "0.267"
    wide1.loc[wide1.variable == "clusters", "(1) ВКР: МНК (модель 4)"] = "—"
    save_table(wide1, "table1_specification_path")

    # ------------------------------------------------------------------
    # Таблица 2. Основные DiD-оценки (PPML, страна×год + парные ЭФ)
    # ------------------------------------------------------------------
    d09 = d[d.year >= 2009]
    clean = d[(d.flag_carry == 0) & ~d.iso3_i.isin(["ARE", "BLR"]) & ~d.iso3_j.isin(["ARE", "BLR"])]
    d["core14_2014"] = d["core14_rus_ij"] * (d["year"] == 2014)
    d["sanc14_alt"] = d["core14_rus_ij"] * (d["year"] >= 2015)
    d["coal_pretrend"] = d["core14_rus_ij"] * (np.minimum(d["year"], 2013) - 2013)   # линейный пре-тренд до 2013 г.
    d09 = d[d.year >= 2009]
    clean = d[(d.flag_carry == 0) & ~d.iso3_i.isin(["ARE", "BLR"]) & ~d.iso3_j.isin(["ARE", "BLR"]) & (d.year >= 2009)]
    base_covs = SANC_VARS + POLICY
    m2 = {
        "(1) 2000–2024": fit_model(d, base_covs, "cy+pair", dyadic=True),
        "(2) 2009–2024 (основная)": fit_model(d09, base_covs, "cy+pair", dyadic=True),
        "(3) 2009–2021": fit_model(d09[d09.year <= 2021], base_covs, "cy+pair", dyadic=True),
        "(4) 2014 — переходный год": fit_model(d09, ["core14_2014", "sanc14_alt"] + [c for c in base_covs if c != "sanc14_ij"],
                                               "cy+pair", dyadic=True),
        "(5) «чистая» выборка": fit_model(clean, base_covs, "cy+pair", dyadic=True),
        "(6) 2000–2024, пре-тренд": fit_model(d, base_covs + ["coal_pretrend"], "cy+pair", dyadic=True),
        "(7) МНК ln T": fit_model(d09, base_covs, "cy+pair", kind="ols"),
    }
    for k, r in m2.items():
        all_long.append(long_rows("T2 " + k, r))
    labels2 = dict(LABELS, core14_2014="Россия × коалиция × (t = 2014)", sanc14_alt="SANC14': Россия × коалиция × (t ≥ 2015)",
                   coal_pretrend="Пре-тренд коалиции, min(t, 2013) − 2013")
    order2 = ["sanc14_ij", "core14_2014", "sanc14_alt", "sanc22_ij", "kor22_ij", "ukr14_ij", "ukr22_ij", "eu_ij",
              "wto_ij", "eaeu_ij", "nato_ij", "brics_ij", "coal_pretrend"]
    wide2 = results_to_wide({k: v for k, v in m2.items()}, labels2, order2)
    save_table(wide2, "table2_main_did")
    # дополнительная таблица: диадно-робастные ошибки для основных моделей
    rows = []
    for k, r in m2.items():
        if r.se_dyadic is None:
            continue
        t = r.table()
        for v in ("sanc14_ij", "sanc22_ij", "kor22_ij", "ukr14_ij", "ukr22_ij"):
            if v in t.index:
                rows.append({"model": k, "variable": v, "coef": t.loc[v, "coef"], "se_pair": t.loc[v, "se"],
                             "se_dyadic": t.loc[v, "se_dyadic"]})
    pd.DataFrame(rows).to_csv(RESULTS / "table2b_dyadic_se.csv", index=False)

    # ключевые числа
    t = m2["(2) 2009–2024 (основная)"].table()
    for v in SANC_VARS + POLICY:
        summary[f"main_{v}"] = {k: float(t.loc[v, k]) for k in ("coef", "se", "p", "pct_effect")}
    summary["main_total_effect_2022_core"] = float(t.loc["sanc14_ij", "coef"] + t.loc["sanc22_ij", "coef"])
    summary["main_nobs"] = m2["(2) 2009–2024 (основная)"].nobs
    c, s = contrast(m2["(2) 2009–2024 (основная)"], {"sanc14_ij": 1, "sanc22_ij": 1})
    summary["main_total_core_post22"] = {"coef": c, "se": s, "pct": 100 * (np.exp(c) - 1)}

    # ------------------------------------------------------------------
    # Таблица 3. Гетерогенность
    # ------------------------------------------------------------------
    eu8 = {"EST", "FIN", "FRA", "DEU", "ITA", "LVA", "LTU", "POL"}
    anglo = {"USA", "CAN", "AUS", "GBR"}
    other_core = {"JPN", "NOR"}
    dh = d09.copy()
    for nm, grp in [("eu", eu8), ("anglo", anglo), ("jpnor", other_core)]:
        m = (dh.rus_pair == 1) & dh.partner.isin(grp)
        dh[f"sanc14_{nm}"] = (m & (dh.year >= 2014)).astype(int)
        dh[f"sanc22_{nm}"] = (m & (dh.year >= 2022)).astype(int)
    covsA = [f"sanc14_{n}" for n in ("eu", "anglo", "jpnor")] + [f"sanc22_{n}" for n in ("eu", "anglo", "jpnor")] + \
            ["kor22_ij", "ukr14_ij", "ukr22_ij"] + POLICY
    rA = fit_model(dh, covsA, "cy+pair")
    # подгруппы «прочих» партнёров (база сравнения — OTHER)
    sub_order = ["CHN", "IND", "TUR", "EAEU_CIS", "GULF_EGY"]
    for sg in sub_order:
        m = (dh.rus_pair == 1) & (dh.partner.map(SUBGROUP_OF) == sg)
        dh[f"piv14_{sg}"] = (m & (dh.year >= 2014)).astype(int)
        dh[f"piv22_{sg}"] = (m & (dh.year >= 2022)).astype(int)
    covsB = SANC_VARS + [f"piv14_{s}" for s in sub_order] + [f"piv22_{s}" for s in sub_order] + POLICY
    rB = fit_model(dh, covsB, "cy+pair")
    all_long += [long_rows("T3 coalition subgroups", rA), long_rows("T3 pivot subgroups", rB)]
    labA = {"sanc14_eu": "SANC14 × ЕС-8", "sanc14_anglo": "SANC14 × США, Канада, Австралия, Великобритания",
            "sanc14_jpnor": "SANC14 × Япония, Норвегия", "sanc22_eu": "SANC22 × ЕС-8",
            "sanc22_anglo": "SANC22 × США, Канада, Австралия, Великобритания", "sanc22_jpnor": "SANC22 × Япония, Норвегия",
            **LABELS}
    wideA = results_to_wide({"(A) подгруппы коалиции": rA}, labA,
                            [f"sanc14_{n}" for n in ("eu", "anglo", "jpnor")] + [f"sanc22_{n}" for n in ("eu", "anglo", "jpnor")] +
                            ["kor22_ij", "ukr14_ij", "ukr22_ij"])
    labB = {**LABELS}
    sg_names = {"CHN": "Китай", "IND": "Индия", "TUR": "Турция", "EAEU_CIS": "ЕАЭС и Кавказ", "GULF_EGY": "Залив и Египет"}
    for s in sub_order:
        labB[f"piv14_{s}"] = f"{sg_names[s]} × (t ≥ 2014)"
        labB[f"piv22_{s}"] = f"{sg_names[s]} × (t ≥ 2022)"
    wideB = results_to_wide({"(B) «прочие» партнёры (база — ARG, BRA, IDN, MEX, ZAF)": rB}, labB,
                            SANC_VARS + [f"piv14_{s}" for s in sub_order] + [f"piv22_{s}" for s in sub_order])
    save_table(wideA, "table3a_coalition_subgroups")
    save_table(wideB, "table3b_other_subgroups")

    # ------------------------------------------------------------------
    # Таблица 4. Градиент расстояния российской торговли
    # ------------------------------------------------------------------
    cov_geo = ["ldist", "contig", "comlang_off", "smctry"] + POLICY
    dgr = d.copy()
    r_d1 = fit_model(dgr, cov_geo + ["ldist_rus_p1", "ldist_rus_p2", "ldist_rus_p3"] + ["coal_rus_ij", "ukr_rus_ij"], "cy")
    r_d1b = fit_model(dgr, cov_geo + ["ldist_rus_p1", "ldist_rus_p2", "ldist_rus_p3"] + ["coal_rus_ij", "ukr_rus_ij"] + SANC_VARS, "cy")
    r_d2 = fit_model(d09, POLICY + ["dist14", "dist22"], "cy+pair")
    r_d3 = fit_model(d09, POLICY + SANC_VARS + ["dist14", "dist22"], "cy+pair")
    for nm, r in [("T4 D1 gradient by period, no sanctions", r_d1), ("T4 D1b gradient by period, with sanctions", r_d1b),
                  ("T4 D2 pair FE, shifts, no sanctions", r_d2), ("T4 D3 pair FE, shifts, with sanctions", r_d3)]:
        all_long.append(long_rows(nm, r))
    grad_rows = []
    for tag, r in [("без санкционных индикаторов", r_d1), ("с санкционными индикаторами", r_d1b)]:
        for p, lab in [("p1", "2000–2013"), ("p2", "2014–2021"), ("p3", "2022–2024")]:
            c, s = contrast(r, {"ldist": 1, f"ldist_rus_{p}": 1})
            grad_rows.append({"spec": tag, "period": lab, "elasticity_russia": c, "se": s,
                              "lo95": c - 1.96 * s, "hi95": c + 1.96 * s})
        c, s = contrast(r, {"ldist_rus_p3": 1, "ldist_rus_p1": -1})
        grad_rows.append({"spec": tag, "period": "Δ (2022–24) − (2000–13)", "elasticity_russia": c, "se": s,
                          "lo95": c - 1.96 * s, "hi95": c + 1.96 * s})
        c0, s0 = contrast(r, {"ldist": 1})
        grad_rows.append({"spec": tag, "period": "прочие пары (база)", "elasticity_russia": c0, "se": s0,
                          "lo95": c0 - 1.96 * s0, "hi95": c0 + 1.96 * s0})
    pd.DataFrame(grad_rows).to_csv(RESULTS / "table4_distance_gradient.csv", index=False)
    labD = {**LABELS, "ldist_rus_p1": "ln расстояние × Россия × 2000–2013", "ldist_rus_p2": "ln расстояние × Россия × 2014–2021",
            "ldist_rus_p3": "ln расстояние × Россия × 2022–2024", "dist14": "ln расстояние (центр.) × Россия × (t ≥ 2014)",
            "dist22": "ln расстояние (центр.) × Россия × (t ≥ 2022)"}
    wide4 = results_to_wide({"(D1) страна×год ЭФ": r_d1, "(D2) + санкц. индикаторы": r_d1b, "(D3) + парные ЭФ": r_d2,
                             "(D4) + парные ЭФ, санкц.": r_d3}, labD,
                            ["ldist", "ldist_rus_p1", "ldist_rus_p2", "ldist_rus_p3", "dist14", "dist22", "contig", "comlang_off",
                             "smctry", "coal_rus_ij", "ukr_rus_ij"] + SANC_VARS + POLICY)
    save_table(wide4, "table4_distance")

    # ------------------------------------------------------------------
    # Партнёр-специфические сдвиги (отн. среднего российского партнёра)
    # ------------------------------------------------------------------
    partners = sorted(d09.loc[d09.rus_pair == 1, "partner"].unique())
    ref = "ZAF"  # технический референс; затем центрируем по среднему
    dp = d09.copy()
    cols = []
    for pcode in partners:
        if pcode == ref:
            continue
        m = (dp.rus_pair == 1) & (dp.partner == pcode)
        dp[f"s2_{pcode}"] = (m & (dp.year.between(2014, 2021))).astype(int)
        dp[f"s3_{pcode}"] = (m & (dp.year >= 2022)).astype(int)
        cols += [f"s2_{pcode}", f"s3_{pcode}"]
    rP = fit_model(dp, cols + POLICY, "cy+pair")
    all_long.append(long_rows("partner-specific shifts", rP))
    rows = []
    for per, pre in [("2014–2021", "s2_"), ("2022–2024", "s3_")]:
        names = [pre + p for p in partners if p != ref]
        n = len(partners)
        for pcode in partners:
            w = {nm: -1.0 / n for nm in names}
            if pcode != ref:
                w[pre + pcode] = w.get(pre + pcode, 0.0) + 1.0
            c, s = contrast(rP, w)
            rows.append({"partner": pcode, "period": per, "coef": c, "se": s, "lo95": c - 1.96 * s, "hi95": c + 1.96 * s,
                         "pct": 100 * (np.exp(c) - 1)})
    pd.DataFrame(rows).to_csv(RESULTS / "partner_specific_shifts.csv", index=False)

    pd.concat(all_long, ignore_index=True).to_csv(RESULTS / "all_models_long.csv", index=False)
    json.dump(summary, open(RESULTS / "summary_main.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    for nm in ("table1_specification_path", "table2_main_did", "table3a_coalition_subgroups", "table3b_other_subgroups",
               "table4_distance"):
        print("\n=====", nm)
        print(pd.read_csv(RESULTS / f"{nm}.csv").fillna("").to_string(index=False))
    print("\n===== gradient\n", pd.read_csv(RESULTS / "table4_distance_gradient.csv").round(3).to_string(index=False))
    print("\n===== partner shifts\n", pd.read_csv(RESULTS / "partner_specific_shifts.csv").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
