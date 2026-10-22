"""Проверка собственного оценщика PPML (code/ppml.py) на эталонных реализациях.

1) Точечные оценки и кластерные ошибки сравниваются с pyfixest.fepois на
   «удвоенной» ориентированной выборке (каждая пара входит как i→j и j→i с
   одинаковой суммарной торговлей; экспортёр×год, импортёр×год и парные эффекты).
   Для симметричных данных оценки коэффициентов при парных/временных переменных
   совпадают с оценками модели с единым эффектом «страна×год».
2) МНК с фиксированными эффектами сравнивается с pyfixest.feols.

Запуск:  python code/03_validate_estimator.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ppml
from gravity_common import DATA, RESULTS

COVS = ["sanc14_ij", "sanc22_ij", "ukr14_ij", "ukr22_ij", "eu_ij", "wto_ij", "eaeu_ij"]


def main() -> None:
    import pyfixest as pf

    d = pd.read_csv(DATA / "gravity_panel.csv")
    y = d["trade"].values
    Z = d[COVS].values.astype(float)
    cy, cy_names = ppml.fe_country_year(d.iso3_i, d.iso3_j, d.year)
    pr, pr_names = ppml.fe_pair(d.pair_id)
    res = ppml.fit(y, [(Z, COVS), (cy, cy_names), (pr, pr_names)], cluster=d.pair_id.values, n_cov=len(COVS),
                   n_nested=pr.shape[1], dyadic=True, iso_i=d.iso3_i.values, iso_j=d.iso3_j.values)
    print(f"собственный PPML: сошёлся={res.converged}, итераций={res.iterations}, N={res.nobs}, "
          f"параметров={res.n_params_total}, кластеров={res.n_clusters}")
    tab = res.table()

    # удвоенная выборка для pyfixest
    a = d.copy()
    a["exp"], a["imp"] = a.iso3_i, a.iso3_j
    b = d.copy()
    b["exp"], b["imp"] = b.iso3_j, b.iso3_i
    dd = pd.concat([a, b], ignore_index=True)
    dd["exp_year"] = dd["exp"] + "_" + dd["year"].astype(str)
    dd["imp_year"] = dd["imp"] + "_" + dd["year"].astype(str)
    fml = "trade ~ " + " + ".join(COVS) + " | exp_year + imp_year + pair_id"
    m = pf.fepois(fml, data=dd, vcov={"CRV1": "pair_id"})
    pt = m.tidy()
    cmp = pd.DataFrame({"coef_own": tab["coef"], "coef_pyfixest": pt["Estimate"].reindex(tab.index),
                        "se_own": tab["se"], "se_pyfixest": pt["Std. Error"].reindex(tab.index)})
    cmp["abs_diff_coef"] = (cmp.coef_own - cmp.coef_pyfixest).abs()
    pd.set_option("display.width", 200)
    print(cmp.round(6).to_string())
    print("макс. |разница| коэффициентов:", cmp.abs_diff_coef.max())

    # МНК с ФЭ
    ly = np.log(y)
    r_ols = ppml.fit(ly, [(Z, COVS), (cy, cy_names), (pr, pr_names)], cluster=d.pair_id.values, n_cov=len(COVS),
                     n_nested=pr.shape[1], kind="ols")
    m2 = pf.feols("np.log(trade) ~ " + " + ".join(COVS) + " | exp_year + imp_year + pair_id", data=dd,
                  vcov={"CRV1": "pair_id"})
    pt2 = m2.tidy()
    cmp2 = pd.DataFrame({"coef_own": r_ols.table()["coef"], "coef_pyfixest": pt2["Estimate"].reindex(COVS)})
    cmp2["abs_diff"] = (cmp2.coef_own - cmp2.coef_pyfixest).abs()
    print(cmp2.round(6).to_string())
    RESULTS.mkdir(exist_ok=True)
    cmp.to_csv(RESULTS / "validation_ppml_vs_pyfixest.csv")
    assert cmp.abs_diff_coef.max() < 1e-5, "оценки PPML расходятся с pyfixest"


if __name__ == "__main__":
    main()
