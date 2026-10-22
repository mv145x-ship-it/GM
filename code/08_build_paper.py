"""Сборка рукописи: шаблон (paper/article_template.md, Jinja2) + результаты → Markdown → DOCX.

Все числа в тексте и таблицах подставляются из results/*.csv|json, поэтому после
обновления данных и повторного запуска конвейера (см. README) рукопись
пересобирается автоматически (интерпретационные выводы требуют ручной проверки).

Запуск:  python code/08_build_paper.py [--no-docx]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

import jinja2
import numpy as np
import pandas as pd

from gravity_common import COALITION_2014, COALITION_2022, COUNTRIES, DATA, FIGS, PAPER, RESULTS, RU_NAME, SUBGROUP_OF
from gravity_models import load_panel

NBSP = " "
MINUS = "−"


# --------------------------------------------------------------- форматирование
def rn(x: float, d: int = 2, sign: bool = False) -> str:
    s = f"{abs(x):.{d}f}".replace(".", ",")
    if x < 0 and float(s.replace(",", ".")) != 0:
        return MINUS + s
    return ("+" + s) if sign and x > 0 else s


def pct(b: float, d: int = 0, sign: bool = False) -> str:
    """Процентный эффект exp(b) − 1."""
    v = 100 * (np.exp(b) - 1)
    return rn(v, d, sign) + NBSP + "%"


def drop(b: float, d: int = 0) -> str:
    """Относительное снижение 1 − exp(b), в процентах (положительное число)."""
    return rn(-100 * (np.exp(b) - 1), d) + NBSP + "%"


def en(s) -> str:
    """Английский формат числа: точка, обычный минус, без неразрывных пробелов."""
    return str(s).replace(NBSP, " ").replace(",", ".").replace(MINUS, "-").replace(" %", "%")


def pc(s) -> str:
    """Добавить неразрывный пробел и знак процента к числовой строке."""
    return f"{s}{NBSP}%"


def ru_cell(s: str) -> str:
    """Десятичная запятая и типографский минус в ячейках таблиц."""
    s = re.sub(r"(?<=\d)\.(?=\d)", ",", str(s))
    s = re.sub(r"(?<![\w)])-(?=\d)", MINUS, s)
    return s


def md_table(df: pd.DataFrame, widths: list[float] | None = None, align: str | None = None) -> str:
    cols = list(df.columns)
    n = len(cols)
    widths = widths or [1.0] * n
    tot = sum(widths)
    seps = []
    for k, w in enumerate(widths):
        dashes = "-" * max(3, int(round(100 * w / tot)))
        a = (align or "l" + "c" * (n - 1))[k]
        seps.append({"l": ":" + dashes, "c": ":" + dashes + ":", "r": dashes + ":"}[a])
    lines = ["| " + " | ".join(ru_cell(c) for c in cols) + " |", "|" + "|".join(seps) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(ru_cell(r[c]).replace("|", "\\|") if pd.notna(r[c]) else "" for c in cols) + " |")
    return "\n".join(lines)


def read_wide(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"{name}.csv", dtype=str).fillna("")


# --------------------------------------------------------------- числа
def load_numbers() -> dict:
    v: dict = {}
    sm = json.load(open(RESULTS / "summary_main.json", encoding="utf-8"))
    se_ = json.load(open(RESULTS / "summary_event_cf.json", encoding="utf-8"))
    si = json.load(open(RESULTS / "summary_inference.json", encoding="utf-8"))
    long = pd.read_csv(RESULTS / "all_models_long.csv")

    def coef(model: str, var: str, col: str = "coef") -> float:
        row = long[(long.model == model) & (long.variable == var)]
        return float(row.iloc[0][col])

    main_model = "T2 (2) 2009–2024 (основная)"
    v["main"] = {k.replace("main_", ""): vv for k, vv in sm.items() if k.startswith("main_") and isinstance(vv, dict)}
    for var in ["sanc14_ij", "sanc22_ij", "kor22_ij", "ukr14_ij", "ukr22_ij", "eu_ij", "wto_ij", "eaeu_ij", "nato_ij", "brics_ij"]:
        v[var] = dict(b=coef(main_model, var), se=coef(main_model, var, "se"), p=coef(main_model, var, "p"))
        v[var]["bs"] = rn(v[var]["b"], 2)
        v[var]["bs3"] = rn(v[var]["b"], 3)
        v[var]["ses"] = rn(v[var]["se"], 3)
        v[var]["pct"] = pct(v[var]["b"])
    tot = sm["main_total_core_post22"]
    v["tot"] = dict(b=tot["coef"], se=tot["se"], bs=rn(tot["coef"], 2), ses=rn(tot["se"], 2), pct=pct(tot["coef"]),
                    lo=pct(tot["coef"] - 1.96 * tot["se"]), hi=pct(tot["coef"] + 1.96 * tot["se"]))
    v["loo14"] = [rn(x, 2) for x in sm["loo_sanc14_range"]]
    v["loo22"] = [rn(x, 2) for x in sm["loo_sanc22_range"]]

    def wald_s(w: dict) -> str:
        return f"χ²({w['df']}) = {rn(w['chi2'], 1 if w['chi2'] >= 10 else 2)}, p {'< 0,001' if w['p'] < 0.001 else '= ' + rn(w['p'], 3)}"
    v["sgt14"], v["sgt22"] = wald_s(sm["subgroup_tests"]["eq14"]), wald_s(sm["subgroup_tests"]["eq22"])
    v["nobs_main"] = f"{sm['main_nobs']:,}".replace(",", NBSP)

    # другие спецификации Таблицы 2
    for key, model in {"w00": "T2 (1) 2000–2024", "w09_21": "T2 (3) 2009–2021", "trans": "T2 (4) 2014 — переходный год",
                       "clean": "T2 (5) «чистая» выборка", "pret": "T2 (6) 2000–2024, пре-тренд", "ols": "T2 (7) МНК ln T"}.items():
        v[key] = {}
        for var in ["sanc14_ij", "sanc22_ij", "kor22_ij"]:
            rows = long[(long.model == model) & (long.variable == var)]
            if len(rows):
                v[key][var] = rn(float(rows.iloc[0].coef), 2)
        rows = long[(long.model == model) & (long.variable == "coal_pretrend")]
        if len(rows):
            v[key]["pretrend"] = rn(float(rows.iloc[0].coef), 3)
    v["trans"]["core14_2014"] = rn(coef("T2 (4) 2014 — переходный год", "core14_2014"), 2)
    v["trans"]["sanc14_alt"] = rn(coef("T2 (4) 2014 — переходный год", "sanc14_alt"), 2)

    # Таблица 1: спецификационный путь
    for k, model in {"s2": "T1 (2) МНК, ln T, год. ЭФ", "s3": "T1 (3) PPML, год. ЭФ", "s4": "T1 (4) PPML, страновые + год. ЭФ",
                     "s5": "T1 (5) PPML, страна×год ЭФ", "s6": "T1 (6) PPML, страна×год + парные ЭФ"}.items():
        v[k] = {}
        for var in ["lgdp_sum", "lpcdiff", "ldist", "contig", "comlang_off", "eu_ij", "nato_ij", "brics_ij", "wto_ij", "eaeu_ij",
                    "sanc14_ij", "sanc22_ij"]:
            rows = long[(long.model == model) & (long.variable == var)]
            if len(rows):
                v[k][var] = dict(b=float(rows.iloc[0].coef), s=rn(float(rows.iloc[0].coef), 2), s3=rn(float(rows.iloc[0].coef), 3),
                                 se=rn(float(rows.iloc[0].se), 3), p=float(rows.iloc[0].p), pct=pct(float(rows.iloc[0].coef)))
    # диапазон санкционных оценок по столбцам (2)–(6) табл. 1: (ближайшая к нулю, наибольшая по модулю)
    for yy in ("14", "22"):
        vals = [v[k][f"sanc{yy}_ij"]["b"] for k in ("s2", "s3", "s4", "s5", "s6")]
        v[f"rng{yy}"] = (rn(max(vals), 2), rn(min(vals), 2))
    rep = pd.read_csv(RESULTS / "replication_vkr_ols.csv")
    r4 = rep[rep.model == 4].set_index("variable")
    v["vkr"] = {k: dict(b=rn(r4.loc[k, "coef"], 3), s=rn(r4.loc[k, "coef"], 2), se=rn(r4.loc[k, "se"], 3), pct=pct(r4.loc[k, "coef"]),
                        x=rn(float(np.exp(r4.loc[k, "coef"])), 1), p=float(r4.loc[k, "pval"])) for k in r4.index}
    r1 = rep[rep.model == 1].set_index("variable")
    v["vkr1"] = {k: dict(b=rn(r1.loc[k, "coef"], 3), s=rn(r1.loc[k, "coef"], 2)) for k in r1.index}
    v["vkr_r2_1"] = rn(float(rep[rep.model == 1].r2.iloc[0]), 3)
    v["vkr_r2_5"] = rn(float(rep[rep.model == 5].r2.iloc[0]), 3)
    v["vkr_lag"] = rn(float(rep[(rep.model == 5) & (rep.variable == "l_trade_lag1")].coef.iloc[0]), 3)

    # event study
    ev = pd.read_csv(RESULTS / "event_study_main.csv")
    e00 = ev[(ev.window == "2000-2024") & ev.group.str.contains("коалиция")].set_index("year")
    v["ev"] = {int(y): dict(b=rn(r.coef, 2), pct=pct(r.coef), raw=r.coef) for y, r in e00.iterrows()}
    v["ev_pre_mean_2000_03"] = rn(e00.loc[2000:2003, "coef"].mean(), 2)
    v["pretrend_full"] = se_["pretrend_wald_2000-2024"]
    v["pretrend_09"] = se_["pretrend_wald_2009_2012"]
    _pf = se_["pretrend_wald_2000-2024"]
    v["pretrend_full_s"] = f"χ²({_pf['df']}) = {rn(_pf['chi2'], 1)}, p {'< 0,001' if _pf['p'] < 0.001 else '= ' + rn(_pf['p'], 3)}"
    v["pretrend_09_s"] = f"χ²({se_['pretrend_wald_2009_2012']['df']}) = {rn(se_['pretrend_wald_2009_2012']['chi2'], 2)}, p = {rn(se_['pretrend_wald_2009_2012']['p'], 2)}"
    eu_ = ev[(ev.window == "2000-2024") & ev.group.str.contains("Украина")].set_index("year")
    v["evu"] = {int(y): dict(b=rn(r.coef, 2), pct=pct(r.coef)) for y, r in eu_.iterrows()}

    # подгруппы коалиции (T3 coalition subgroups)
    for k, var in {"eu14": "sanc14_eu", "an14": "sanc14_anglo", "jn14": "sanc14_jpnor", "eu22": "sanc22_eu", "an22": "sanc22_anglo",
                   "jn22": "sanc22_jpnor"}.items():
        v[k] = dict(s=rn(coef("T3 coalition subgroups", var), 2), pct=pct(coef("T3 coalition subgroups", var)))
    for k, var in {"chn14": "piv14_CHN", "ind14": "piv14_IND", "tur14": "piv14_TUR", "eac14": "piv14_EAEU_CIS", "gulf14": "piv14_GULF_EGY",
                   "chn22": "piv22_CHN", "ind22": "piv22_IND", "tur22": "piv22_TUR", "eac22": "piv22_EAEU_CIS", "gulf22": "piv22_GULF_EGY"}.items():
        v[k] = dict(s=rn(coef("T3 pivot subgroups", var), 2), se=rn(coef("T3 pivot subgroups", var, "se"), 2),
                    p=coef("T3 pivot subgroups", var, "p"))
    for k, var in {"b14": "brics_p14", "b22": "brics_p22"}.items():
        v[k] = dict(s=rn(coef("T3c brics partners", var), 2), se=rn(coef("T3c brics partners", var, "se"), 2),
                    p=rn(coef("T3c brics partners", var, "p"), 3))
    for k, var in {"h14": "hub_p14", "h22": "hub_p22"}.items():
        v[k] = dict(s=rn(coef("T3c hubs", var), 2), se=rn(coef("T3c hubs", var, "se"), 2), p=rn(coef("T3c hubs", var, "p"), 2))

    # градиент расстояния
    g = pd.read_csv(RESULTS / "table4_distance_gradient.csv")
    for tag, key in [("без санкционных индикаторов", "dg0"), ("с санкционными индикаторами", "dg1")]:
        sub = g[g.spec == tag].set_index("period")
        v[key] = {p: dict(b=rn(sub.loc[p, "elasticity_russia"], 2), se=rn(sub.loc[p, "se"], 2), lo=rn(sub.loc[p, "lo95"], 2),
                          hi=rn(sub.loc[p, "hi95"], 2)) for p in sub.index}
    v["dist_base"] = v["dg0"]["прочие пары (база)"]["b"]
    dlt = "Δ (2022–24) − (2000–13)"
    d0 = float(g[(g.spec == "без санкционных индикаторов") & (g.period == dlt)].elasticity_russia.iloc[0])
    d1 = float(g[(g.spec == "с санкционными индикаторами") & (g.period == dlt)].elasticity_russia.iloc[0])
    v["dg_share"] = pc(rn(100 * (1 - d1 / d0), 0))
    v["dshift14"] = dict(s=rn(coef("T4 D2 pair FE, shifts, no sanctions", "dist14"), 2), se=rn(coef("T4 D2 pair FE, shifts, no sanctions", "dist14", "se"), 2),
                         s_c=rn(coef("T4 D3 pair FE, shifts, with sanctions", "dist14"), 2), se_c=rn(coef("T4 D3 pair FE, shifts, with sanctions", "dist14", "se"), 2))
    v["dshift22"] = dict(s=rn(coef("T4 D2 pair FE, shifts, no sanctions", "dist22"), 2), se=rn(coef("T4 D2 pair FE, shifts, no sanctions", "dist22", "se"), 2),
                         s_c=rn(coef("T4 D3 pair FE, shifts, with sanctions", "dist22"), 2), se_c=rn(coef("T4 D3 pair FE, shifts, with sanctions", "dist22", "se"), 2))

    v["dshift_sum"] = dict(
        s=rn(coef("T4 D2 pair FE, shifts, no sanctions", "dist14") + coef("T4 D2 pair FE, shifts, no sanctions", "dist22"), 2),
        s_c=rn(coef("T4 D3 pair FE, shifts, with sanctions", "dist14") + coef("T4 D3 pair FE, shifts, with sanctions", "dist22"), 2))

    # партнёрские сдвиги
    ps = pd.read_csv(RESULTS / "partner_specific_shifts.csv")
    v["ps"] = {(r.partner, r.period): dict(b=rn(r.coef, 2), se=rn(r.se, 2)) for r in ps.itertuples()}

    # контрфактуал
    agg = pd.read_csv(RESULTS / "counterfactual_aggregates.csv")
    v["cf"] = {(r.group, r.period): dict(actual=rn(r.actual_bn, 0), pred=rn(r.predicted_bn, 0), gap=rn(r.gap_bn, 0), ratio=rn(r.ratio, 2),
                                         gap_abs=rn(abs(r.gap_bn), 0)) for r in agg.itertuples()}
    cf = pd.read_csv(RESULTS / "counterfactual_group_year.csv")
    main = cf[cf.spec.str.startswith("основной")]
    v["cf_ratio"] = {(r.group, int(r.year)): rn(r.ratio, 2) for r in main.itertuples()}
    alts = cf[cf.spec.str.startswith(("основной", "альт"))]
    rng = {}
    for grp in ("coalition", "other", "ukraine"):
        for yr in (2021, 2024):
            x = alts[(alts.group == grp) & (alts.year == yr)].ratio
            rng[(grp, yr)] = (rn(x.min(), 2), rn(x.max(), 2))
    v["cf_range"] = rng
    # суммарные величины и доля компенсации
    tot_act = {p: sum(float(agg[(agg.group == g_) & (agg.period == p)].actual_bn.iloc[0]) for g_ in ("coalition", "other", "ukraine")) for p in ("2014–2021", "2022–2024")}
    tot_pred = {p: sum(float(agg[(agg.group == g_) & (agg.period == p)].predicted_bn.iloc[0]) for g_ in ("coalition", "other", "ukraine")) for p in ("2014–2021", "2022–2024")}
    v["cf_total"] = {p: dict(actual=rn(tot_act[p], 0), pred=rn(tot_pred[p], 0), gap=rn(tot_act[p] - tot_pred[p], 0),
                             gap_abs=rn(abs(tot_act[p] - tot_pred[p]), 0), share=rn(100 * (tot_act[p] / tot_pred[p] - 1), 0)) for p in tot_act}
    comp = {}
    for p in ("2014–2021", "2022–2024"):
        loss = -(float(agg[(agg.group == "coalition") & (agg.period == p)].gap_bn.iloc[0]) + float(agg[(agg.group == "ukraine") & (agg.period == p)].gap_bn.iloc[0]))
        gain = float(agg[(agg.group == "other") & (agg.period == p)].gap_bn.iloc[0])
        comp[p] = dict(loss=rn(loss, 0), gain=rn(gain, 0), share=rn(100 * gain / loss, 0))
    v["cf_comp"] = comp
    pl = pd.read_csv(RESULTS / "counterfactual_placebo_accuracy.csv")
    v["placebo"] = {(r.spec[:9], r.group): dict(mean_ratio=rn(r.mean_ratio, 2), mae=rn(r.mean_abs_log_err, 2)) for r in pl.itertuples()}

    # инференс
    pm = si["permutation"]
    v["perm"] = dict(n=pm["n_perm"], p14=rn(pm["perm_p_two_sided_sanc14"], 3), p22=rn(pm["perm_p_two_sided_sanc22"], 3),
                     sd14=rn(pm["perm_sd_sanc14"], 2), sd22=rn(pm["perm_sd_sanc22"], 2),
                     q14=(rn(pm["perm_q05_sanc14"], 2), rn(pm["perm_q95_sanc14"], 2)), q22=(rn(pm["perm_q05_sanc22"], 2), rn(pm["perm_q95_sanc22"], 2)))
    bb = pd.read_csv(RESULTS / "bootstrap_bias_check.csv").set_index("variable")
    v["boot"] = {k: dict(bias=rn(bb.loc[k, "bias"], 3), bias_se=rn(bb.loc[k, "bias_over_se"], 2), sd=rn(bb.loc[k, "boot_sd"], 3),
                         se=rn(bb.loc[k, "se_analytic"], 3), cov=rn(100 * bb.loc[k, "coverage95"], 0)) for k in bb.index}
    v["n_boot"] = si["bootstrap"]["n_boot"]
    # во сколько раз надо увеличить ст. ошибки, чтобы эмпирическое покрытие стало номинальным (округление до 5 %: 200 повторений)
    from scipy.stats import norm
    infl = [100 * (1.96 / norm.ppf((1 + bb.loc[k, "coverage95"]) / 2) - 1) for k in ("sanc14_ij", "sanc22_ij")]
    v["boot_infl"] = f"{5 * round(float(np.mean(infl)) / 5):.0f}{NBSP}%"
    dy = pd.read_csv(RESULTS / "table2b_dyadic_se.csv")
    d2 = dy[dy.model == "(2) 2009–2024 (основная)"].set_index("variable")
    v["dyad"] = {k: dict(se=rn(d2.loc[k, "se_pair"], 3), sed=rn(d2.loc[k, "se_dyadic"], 3)) for k in d2.index}

    # описательные числа
    d = load_panel()
    r = d[d.rus_pair == 1].copy()
    r["grp"] = r.partner.map(lambda p: "coalition" if p in COALITION_2022 else ("ukraine" if p == "UKR" else "other"))
    byy = r.groupby(["year", "grp"]).trade.sum().unstack().fillna(0) / 1e3
    tot_y = byy.sum(axis=1)
    sh = 100 * byy["coalition"] / tot_y
    wd = r.groupby("year").apply(lambda x: np.average(x.dist, weights=x.trade), include_groups=False) / 1e3
    v["desc"] = dict(tot={int(y): rn(x, 0) for y, x in tot_y.items()}, share={int(y): rn(x, 0) for y, x in sh.items()},
                     wd={int(y): rn(x, 1) for y, x in wd.items()},
                     coal={int(y): rn(x, 0) for y, x in byy["coalition"].items()}, other={int(y): rn(x, 0) for y, x in byy["other"].items()})
    v["desc"]["wd_delta"] = rn(wd.loc[2024] - wd.loc[2000], 1)
    v["n_pairs"], v["n_countries"], v["n_years"] = d.pair_id.nunique(), len(COUNTRIES), d.year.nunique()
    v["n_obs"] = f"{len(d):,}".replace(",", NBSP)
    # аудит данных
    qr = pd.read_csv(DATA / "data_quality_report.csv")
    v["q_dist_n"] = int(qr[qr.variable == "dist_vkr"].n_obs_or_pairs.iloc[0])
    v["q_jump"] = qr[qr.issue.str.contains("скачки")].details.iloc[0]
    m_ = re.search(r"([\d.]+)% пар против ([\d.]+)%", v["q_jump"])
    v["q22_share"], v["q19_share"] = rn(float(m_.group(1)), 1), rn(float(m_.group(2)), 1)
    v["q_carry24"] = int(qr[(qr.issue.str.contains("протянуто")) & (qr.details == "год 2024")].n_obs_or_pairs.iloc[0])
    v["q_carry23"] = int(qr[(qr.issue.str.contains("протянуто")) & (qr.details == "год 2023")].n_obs_or_pairs.iloc[0])
    return v


# --------------------------------------------------------------- таблицы
def build_tables(v: dict) -> dict:
    T: dict = {}
    # Таблица 1
    t1 = read_wide("table1_specification_path")
    t1 = t1.rename(columns={"variable": "Переменная"})
    T["t1"] = md_table(t1, widths=[2.6, 1.3, 1.3, 1.3, 1.5, 1.4, 1.6])
    # Таблица 2
    t2 = read_wide("table2_main_did").rename(columns={"variable": "Переменная"})
    T["t2"] = md_table(t2, widths=[2.4, 1.2, 1.3, 1.1, 1.4, 1.2, 1.4, 1.2])
    # Таблица 3
    a = read_wide("table3a_coalition_subgroups").rename(columns={"variable": "Переменная"})
    b = read_wide("table3b_other_subgroups").rename(columns={"variable": "Переменная"})
    c = read_wide("table3c_brics_hubs").rename(columns={"variable": "Переменная"})
    T["t3a"] = md_table(a, widths=[3.2, 1.6])
    T["t3b"] = md_table(b, widths=[3.2, 1.6])
    T["t3c"] = md_table(c, widths=[3.4, 1.4, 1.4])
    # Таблица 4
    t4 = read_wide("table4_distance").rename(columns={"variable": "Переменная"})
    T["t4"] = md_table(t4, widths=[2.8, 1.3, 1.3, 1.3, 1.3])
    # Таблица 5: контрфактуал
    agg = pd.read_csv(RESULTS / "counterfactual_aggregates.csv")
    name = {"coalition": "Страны коалиции", "other": "Прочие партнёры", "ukraine": "Украина"}
    rows = []
    for r in agg.itertuples():
        rows.append({"Группа": name[r.group], "Период": r.period, "Факт, млрд долл.": rn(r.actual_bn, 0), "Ориентир, млрд долл.": rn(r.predicted_bn, 0),
                     "Разрыв, млрд долл.": rn(r.gap_bn, 0, sign=True), "Факт / ориентир": rn(r.ratio, 2)})
    for p in ("2014–2021", "2022–2024"):
        tt = v["cf_total"][p]
        rows.append({"Группа": "Всего (31 партнёр)", "Период": p, "Факт, млрд долл.": tt["actual"], "Ориентир, млрд долл.": tt["pred"],
                     "Разрыв, млрд долл.": tt["gap"], "Факт / ориентир": rn(float(tt["actual"].replace(MINUS, "-").replace(",", ".").replace(NBSP, "")) /
                                                                       float(tt["pred"].replace(MINUS, "-").replace(",", ".").replace(NBSP, "")), 2)})
    T["t5"] = md_table(pd.DataFrame(rows), widths=[2.0, 1.1, 1.3, 1.5, 1.5, 1.2])
    # Таблица 6: устойчивость вывода
    pm = json.load(open(RESULTS / "summary_inference.json", encoding="utf-8"))["permutation"]
    rows = []
    for var, lab, ps_ in [("sanc14_ij", "SANC14", "sanc14"), ("sanc22_ij", "SANC22", "sanc22")]:
        bb = v["boot"][var]
        rows.append({"Коэф.": lab, "Оценка": rn(v[var]["b"], 3), "СО (пары)": v["dyad"][var]["se"],
                     "СО (диады)": v["dyad"][var]["sed"], "СО (перест.)": rn(pm[f"perm_sd_{ps_}"], 3),
                     "p (перест.)": rn(pm[f"perm_p_two_sided_{ps_}"], 3), "Смещение (бутстрэп)": bb["bias"],
                     "Покрытие 95 % ДИ, %": bb["cov"]})
    T["t6"] = md_table(pd.DataFrame(rows), widths=[0.9, 0.9, 0.9, 0.9, 1.0, 0.9, 1.2, 1.3])
    # Приложение: аудит данных
    qr = pd.read_csv(DATA / "data_quality_report.csv")
    qa = qr[["issue", "variable", "details", "n_obs_or_pairs"]].copy()
    qa.columns = ["Дефект", "Переменная", "Описание", "Кол-во"]
    T["ta1"] = md_table(qa, widths=[2.4, 1.0, 3.4, 0.6], align="llll")
    # Приложение: состав групп
    rows = []
    for grp, codes in [("Коалиция 2014 г. («ядро», 14 стран)", sorted(COALITION_2014)), ("Присоединилась в 2022 г.", ["KOR"]),
                       ("Украина", ["UKR"]),
                       ("Прочие партнёры (15 стран)", sorted(set(COUNTRIES) - COALITION_2022 - {"RUS", "UKR"}))]:
        rows.append({"Группа": grp, "Страны": ", ".join(f"{RU_NAME[c]} ({c})" for c in codes)})
    T["ta2"] = md_table(pd.DataFrame(rows), widths=[1.4, 4.0], align="ll")
    # Приложение: партнёрская таблица
    d = load_panel()
    r = d[d.rus_pair == 1]
    piv = r.pivot_table(index="partner", columns="year", values="trade", aggfunc="sum") / 1e3
    ps = pd.read_csv(RESULTS / "partner_specific_shifts.csv")
    cfp = pd.read_csv(RESULTS / "counterfactual_partner_year.csv")
    cfp = cfp[cfp.spec.str.startswith("основной")]
    rows = []
    for p in sorted(piv.index, key=lambda x: RU_NAME[x]):
        grp = "коалиция" if p in COALITION_2022 else ("Украина" if p == "UKR" else "прочие")
        s2 = ps[(ps.partner == p) & (ps.period == "2014–2021")].iloc[0]
        s3 = ps[(ps.partner == p) & (ps.period == "2022–2024")].iloc[0]
        r21 = cfp[(cfp.partner == p) & (cfp.year == 2021)].ratio.iloc[0]
        r24 = cfp[(cfp.partner == p) & (cfp.year == 2024)].ratio.iloc[0]
        rows.append({"Партнёр": RU_NAME[p], "Группа": grp, "2013": rn(piv.loc[p, 2013], 1), "2021": rn(piv.loc[p, 2021], 1), "2024": rn(piv.loc[p, 2024], 1),
                     "Сдвиг 2014–21": rn(s2.coef, 2), "Сдвиг 2022–24": rn(s3.coef, 2), "Факт/ор. 2021": rn(r21, 2), "Факт/ор. 2024": rn(r24, 2)})
    T["ta3"] = md_table(pd.DataFrame(rows), widths=[1.6, 0.9, 0.7, 0.7, 0.7, 1.0, 1.0, 1.0, 1.0], align="llcccccc" + "c")
    # Приложение: словарь переменных (краткий)
    from data_dictionary import PANEL_VARS
    dd = pd.DataFrame([(a, b, c, d_, e) for a, b, c, d_, e in PANEL_VARS], columns=["Переменная", "Обозначение", "В ВКР", "Единицы", "Описание"])
    dd["Переменная"] = dd["Переменная"].map(lambda s: f"`{s}`")
    T["ta4"] = md_table(dd[["Переменная", "Обозначение", "В ВКР", "Описание"]], widths=[1.4, 1.0, 1.0, 4.0], align="llll")
    return T


# --------------------------------------------------------------- сборка
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-docx", action="store_true")
    ap.add_argument("--template", default=None, help="по умолчанию — склейка paper/parts/*.md")
    ap.add_argument("--out", default="article_ru")
    args = ap.parse_args()

    v = load_numbers()
    T = build_tables(v)
    env = jinja2.Environment(variable_start_string="<<", variable_end_string=">>", block_start_string="<%", block_end_string="%>",
                             comment_start_string="<#", comment_end_string="#>", undefined=jinja2.StrictUndefined,
                             trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    env.filters["rn"] = rn
    env.filters["pct"] = pct
    env.filters["drop"] = drop
    env.filters["p"] = pc
    env.filters["en"] = en
    if args.template:
        src = open(args.template, encoding="utf-8").read()
    else:
        parts = sorted((PAPER / "parts").glob("*.md"))
        src = "\n\n".join(open(f, encoding="utf-8").read().strip("\n") for f in parts)
    tpl = env.from_string(src)
    md = tpl.render(v=v, T=T, NBSP=NBSP, MINUS=MINUS)
    out_md = PAPER / f"{args.out}.md"
    out_md.write_text(md, encoding="utf-8")
    print("Markdown:", out_md, f"({len(md.split())} слов)")
    if args.no_docx:
        return
    subprocess.run([sys.executable, str(PAPER.parent / "code" / "09_make_docx.py"), "--md", str(out_md), "--out", str(PAPER / f"{args.out}.docx")],
                   check=True)


if __name__ == "__main__":
    main()
