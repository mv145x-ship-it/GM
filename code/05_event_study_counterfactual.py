"""Событийный анализ (event study) и контрфактуальные расчёты.

1) Event study: динамика парных эффектов «Россия × коалиция» и «Россия × Украина»
   относительно 2013 г. (PPML, ФЭ страна×год и пара); проверка пре-трендов.
2) Event study по подгруппам: подгруппы коалиции и подгруппы «прочих» партнёров
   (база сравнения — Аргентина, Бразилия, Индонезия, Мексика, ЮАР).
3) Контрфактуал без санкций: гравитационная модель (парные и годовые ФЭ, ВВП,
   институциональные индикаторы) оценивается на {все пары до года отсечения} ∪
   {пары без России после него}; затем прогнозируется торговля России. Для оценки
   точности контрфактуала проводится плацебо-прогноз на донаучной выборке (2009–2013).

Запуск:  python code/05_event_study_counterfactual.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

import ppml
from gravity_common import COALITION_2014, COALITION_2022, RESULTS, SUBGROUP_OF
from gravity_models import fit_model, load_panel

POLICY = ["eu_ij", "wto_ij", "eaeu_ij", "nato_ij", "brics_ij"]
EU8 = {"EST", "FIN", "FRA", "DEU", "ITA", "LVA", "LTU", "POL"}
ANGLO = {"USA", "CAN", "AUS", "GBR"}
JPNOR = {"JPN", "NOR"}


def event_dummies(d: pd.DataFrame, mask: pd.Series, label: str, years: list[int]) -> list[str]:
    cols = []
    for y in years:
        c = f"{label}_{y}"
        d[c] = (mask & (d["year"] == y)).astype(int)
        cols.append(c)
    return cols


def wald(r: ppml.FitResult, names: list[str]) -> tuple[float, int, float]:
    idx = [r.names.index(n) for n in names if n in r.names]
    b = r.params[idx]
    V = r.vcov[np.ix_(idx, idx)]
    w = float(b @ np.linalg.solve(V, b))
    return w, len(idx), float(1 - stats.chi2.cdf(w, len(idx)))


def tidy_event(r: ppml.FitResult, label: str, years: list[int], group: str, window: str) -> pd.DataFrame:
    t = r.table()
    rows = []
    for y in years:
        n = f"{label}_{y}"
        if n in t.index:
            c, s = t.loc[n, "coef"], t.loc[n, "se"]
            rows.append(dict(window=window, group=group, year=y, coef=c, se=s, lo95=c - 1.96 * s, hi95=c + 1.96 * s))
    return pd.DataFrame(rows)


def main() -> None:
    d = load_panel()
    rus = d["rus_pair"] == 1
    out_event, summary = [], {}

    # ------------------------------------------------------------------
    # 1) Основной event study
    # ------------------------------------------------------------------
    for window, y0 in [("2000-2024", 2000), ("2009-2024", 2009)]:
        dd = d[d.year >= y0].copy()
        years = [y for y in range(y0, 2025) if y != 2013]
        cols = event_dummies(dd, dd.core14_rus_ij.eq(1), "coal", years)
        cols += event_dummies(dd, dd.ukr_rus_ij.eq(1), "ukr", years)
        covs = cols + ["kor22_ij"] + POLICY
        r = fit_model(dd, covs, "cy+pair", dyadic=True)
        out_event += [tidy_event(r, "coal", years, "Россия × коалиция (14 стран ядра)", window),
                      tidy_event(r, "ukr", years, "Россия × Украина", window)]
        pre = [f"coal_{y}" for y in years if y <= 2012]
        w, k, p = wald(r, pre)
        summary[f"pretrend_wald_{window}"] = {"chi2": w, "df": k, "p": p}
        if window == "2009-2024":
            pre2 = [f"coal_{y}" for y in (2009, 2010, 2011, 2012)]
            w, k, p = wald(r, pre2)
            summary["pretrend_wald_2009_2012"] = {"chi2": w, "df": k, "p": p}
        print(f"[event study {window}] pre-trend Wald: chi2({k})={w:.2f}, p={p:.3f}")
    pd.concat(out_event, ignore_index=True).to_csv(RESULTS / "event_study_main.csv", index=False)

    # ------------------------------------------------------------------
    # 2) Event study по подгруппам (окно 2009–2024)
    # ------------------------------------------------------------------
    dd = d[d.year >= 2009].copy()
    years = [y for y in range(2009, 2025) if y != 2013]
    groups = {
        "ЕС-8": dd.partner.isin(EU8), "США, Канада, Австралия, Великобритания": dd.partner.isin(ANGLO),
        "Япония, Норвегия": dd.partner.isin(JPNOR),
        "Китай": dd.partner.eq("CHN"), "Индия": dd.partner.eq("IND"), "Турция": dd.partner.eq("TUR"),
        "ЕАЭС и Кавказ": dd.partner.map(SUBGROUP_OF).eq("EAEU_CIS"), "Залив и Египет": dd.partner.map(SUBGROUP_OF).eq("GULF_EGY"),
        "Украина": dd.partner.eq("UKR"),
    }
    cols, tags = [], {}
    for k, (name, m) in enumerate(groups.items()):
        tag = f"g{k}"
        tags[name] = tag
        cols += event_dummies(dd, rus.loc[dd.index] & m, tag, years)
    covs = cols + ["kor22_ij"] + POLICY
    r = fit_model(dd, covs, "cy+pair")
    ev2 = pd.concat([tidy_event(r, tag, years, name, "2009-2024, база: ARG, BRA, IDN, MEX, ZAF") for name, tag in tags.items()],
                    ignore_index=True)
    ev2.to_csv(RESULTS / "event_study_subgroups.csv", index=False)

    # ------------------------------------------------------------------
    # 3) Контрфактуал: прогноз торговли России без санкций
    # ------------------------------------------------------------------
    cf_rows, cf_partner = [], []
    specs = {
        # (первый год окна оценивания, последний «доисторический» год, регрессоры, метка)
        "основной: окно 2009–2013, ВВП": (2009, 2013, ["lgdp_sum"] + POLICY),
        "альт. 1: окно 2000–2013, ВВП": (2000, 2013, ["lgdp_sum"] + POLICY),
        "альт. 2: окно 2009–2013, без ВВП": (2009, 2013, POLICY),
        "плацебо A: окно 2000–2008": (2000, 2008, ["lgdp_sum"] + POLICY),
        "плацебо B: окно 2005–2010": (2005, 2010, ["lgdp_sum"] + POLICY),
    }
    grp_of = lambda p: "coalition" if p in COALITION_2022 else ("ukraine" if p == "UKR" else "other")
    for label, (y0, cutoff, covs) in specs.items():
        dd = d[d.year >= y0].reset_index(drop=True)
        est = ((dd.year <= cutoff) | (dd.rus_pair == 0)).values
        blocks = [(dd[covs].values.astype(float), covs), ppml.fe_year(dd.year), ppml.fe_pair(dd.pair_id)]
        r = ppml.fit(dd["trade"].values, blocks, cluster=dd["pair_id"].values, n_cov=len(covs), n_nested=0,
                     sample=est)
        dd["pred"] = r.mu_all
        rr = dd[(dd.rus_pair == 1) & (dd.year > cutoff)].copy()
        rr["group"] = rr.partner.map(grp_of)
        g = rr.groupby(["year", "group"])[["trade", "pred"]].sum().reset_index()
        g["ratio"] = g["trade"] / g["pred"]
        g["spec"], g["cutoff"] = label, cutoff
        cf_rows.append(g)
        pp = rr.groupby(["partner", "year"])[["trade", "pred"]].sum().reset_index()
        pp["ratio"] = pp["trade"] / pp["pred"]
        pp["spec"] = label
        cf_partner.append(pp)
        beta = r.table().loc["lgdp_sum", "coef"] if "lgdp_sum" in r.names else np.nan
        print(f"[{label}] β(ln GDP sum)={beta:.3f}")
    cf = pd.concat(cf_rows, ignore_index=True)
    cf.to_csv(RESULTS / "counterfactual_group_year.csv", index=False)
    pd.concat(cf_partner, ignore_index=True).to_csv(RESULTS / "counterfactual_partner_year.csv", index=False)

    # агрегаты: суммарные «недополученные» / «избыточные» потоки (млрд долл.)
    main = cf[cf.spec.str.startswith("основной")]
    agg = []
    for (g, lo, hi, lab) in [("coalition", 2014, 2021, "2014–2021"), ("coalition", 2022, 2024, "2022–2024"),
                              ("other", 2014, 2021, "2014–2021"), ("other", 2022, 2024, "2022–2024"),
                              ("ukraine", 2014, 2021, "2014–2021"), ("ukraine", 2022, 2024, "2022–2024")]:
        s = main[(main.group == g) & main.year.between(lo, hi)]
        agg.append(dict(group=g, period=lab, actual_bn=s.trade.sum() / 1e3, predicted_bn=s.pred.sum() / 1e3,
                        gap_bn=(s.trade.sum() - s.pred.sum()) / 1e3, ratio=s.trade.sum() / s.pred.sum()))
    agg = pd.DataFrame(agg)
    agg.to_csv(RESULTS / "counterfactual_aggregates.csv", index=False)
    print(agg.round(2).to_string(index=False))

    # точность плацебо-прогнозов
    pl = cf[cf.spec.str.startswith("плацебо")].copy()
    pl["abs_log_err"] = np.abs(np.log(pl.ratio))
    acc = pl.groupby(["spec", "group"]).agg(mean_ratio=("ratio", "mean"), mean_abs_log_err=("abs_log_err", "mean"),
                                            max_abs_log_err=("abs_log_err", "max")).reset_index()
    acc.to_csv(RESULTS / "counterfactual_placebo_accuracy.csv", index=False)
    print(acc.round(3).to_string(index=False))
    summary["counterfactual"] = agg.to_dict(orient="records")
    json.dump(summary, open(RESULTS / "summary_event_cf.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n== main spec group-year ratios\n", main.pivot(index="year", columns="group", values="ratio").round(2).to_string())


if __name__ == "__main__":
    main()
