"""Вспомогательные функции для спецификаций гравитационной модели.

Загрузка панели, построение производных переменных, универсальная обёртка
``fit_model`` над ppml.fit, форматирование таблиц результатов.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import ppml
from gravity_common import COALITION_2014, DATA, RESULTS

CORE = sorted(COALITION_2014)


def load_panel(path=None) -> pd.DataFrame:
    d = pd.read_csv(path or DATA / "gravity_panel.csv")
    d["ldist"] = np.log(d["dist"])
    d["ldist_cap"] = np.log(d["dist_cap"])
    d["lgdp_sum"] = np.log(d["gdp_i"]) + np.log(d["gdp_j"])
    d["lpop_sum"] = np.log(d["pop_i"]) + np.log(d["pop_j"])
    d["lpcdiff"] = (np.log(d["gdppc_i"]) - np.log(d["gdppc_j"])).abs()
    d["ltrade"] = np.log(d["trade"])
    rus = d["rus_pair"] == 1
    # центрированный логарифм расстояния (по российским партнёрам) для интерпретации сдвигов
    d["ld_c"] = d["ldist"] - d.loc[rus, "ldist"].mean()
    d["dist14"] = d["ld_c"] * rus * (d["year"] >= 2014)
    d["dist22"] = d["ld_c"] * rus * (d["year"] >= 2022)
    # периоды
    d["period"] = pd.cut(d["year"], [1999, 2013, 2021, 2100], labels=["p1", "p2", "p3"]).astype(str)
    for p in ("p1", "p2", "p3"):
        d[f"ldist_rus_{p}"] = d["ldist"] * rus * (d["period"] == p)
    d["post14"] = (d["year"] >= 2014).astype(int)
    d["post22"] = (d["year"] >= 2022).astype(int)
    return d


def build_blocks(d: pd.DataFrame, covs: list[str], fe: str):
    """Блоки регрессоров. fe ∈ {'none','year','country+year','cy','cy+pair','year+pair'}."""
    d = d.reset_index(drop=True)
    blocks = [(d[covs].values.astype(float), covs)]
    nested = 0
    if fe == "none":
        blocks.append((np.ones((len(d), 1)), ["const"]))
    elif fe == "year":
        blocks.append(ppml.fe_year(d["year"]))
    elif fe == "country+year":
        blocks.append(ppml.fe_country(d["iso3_i"], d["iso3_j"], drop_first=True))
        blocks.append(ppml.fe_year(d["year"]))
    elif fe == "cy":
        blocks.append(ppml.fe_country_year(d["iso3_i"], d["iso3_j"], d["year"], base_year=-1))
    elif fe == "cy+pair":
        blocks.append(ppml.fe_country_year(d["iso3_i"], d["iso3_j"], d["year"], base_year=int(d["year"].min())))
        pr = ppml.fe_pair(d["pair_id"])
        blocks.append(pr)
        nested = pr[0].shape[1]
    elif fe == "year+pair":
        blocks.append(ppml.fe_year(d["year"]))
        pr = ppml.fe_pair(d["pair_id"])
        blocks.append(pr)
        nested = pr[0].shape[1]
    else:
        raise ValueError(fe)
    return blocks, nested


def fit_model(d: pd.DataFrame, covs: list[str], fe: str, kind: str = "ppml", dyadic: bool = False) -> ppml.FitResult:
    """Оценить модель на подвыборке d. Постоянные (нулевая дисперсия) регрессоры отбрасываются."""
    d = d.reset_index(drop=True)
    covs = [c for c in covs if d[c].std() > 0]
    blocks, nested = build_blocks(d, covs, fe)
    y = d["trade"].values if kind == "ppml" else np.log(d["trade"].values)
    return ppml.fit(y, blocks, cluster=d["pair_id"].values, n_cov=len(covs), kind=kind,
                    n_nested=nested, dyadic=dyadic, iso_i=d["iso3_i"].values, iso_j=d["iso3_j"].values)


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt_cell(coef: float, se: float, p: float, digits: int = 3) -> str:
    return f"{coef:.{digits}f}{stars(p)} ({se:.{digits}f})"


def results_to_wide(results: dict[str, ppml.FitResult], labels: dict[str, str], order: list[str]) -> pd.DataFrame:
    """Сводная таблица: строки — переменные, столбцы — модели; ячейка «коэф.*** (ст.ош.)»."""
    rows = []
    for var in order:
        row = {"variable": labels.get(var, var)}
        for name, r in results.items():
            t = r.table()
            row[name] = fmt_cell(t.loc[var, "coef"], t.loc[var, "se"], t.loc[var, "p"]) if var in t.index else ""
        rows.append(row)
    foot = {"N": {}, "pseudo R2": {}, "clusters": {}}
    for name, r in results.items():
        foot["N"][name] = f"{r.nobs}"
        foot["pseudo R2"][name] = f"{r.pseudo_r2:.3f}"
        foot["clusters"][name] = f"{r.n_clusters}"
    for k, v in foot.items():
        rows.append({"variable": k, **v})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def save_table(df: pd.DataFrame, name: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f"{name}.csv", index=False)
    (RESULTS / f"{name}.md").write_text(to_markdown(df) + "\n", encoding="utf-8")
