"""Воспроизведение моделей 1–7 из ВКР (gretl, пулированный МНК) средствами Python.

Цель — убедиться, что (а) данные читаются корректно и (б) ниже обсуждаемые
методологические проблемы относятся именно к исходной спецификации, а не к
ошибкам переноса. Целевые значения взяты со скриншотов gretl в тексте ВКР.

Запуск:  python code/02_replicate_vkr.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from gravity_common import DATA, RESULTS

# Целевые значения из ВКР: {модель: (N, R2, DW, {переменная: (коэф., ст. ошибка)})}
TARGET = {
    1: (7440, 0.225220, 0.031011, {"const": (4.73930, 0.359124), "l_gdppc_i": (0.393403, 0.0208882),
                                   "l_gdppc_j": (0.839400, 0.0211747), "l_dist": (-0.288934, 0.0283729)}),
    2: (6944, 0.970984, 2.265239, {"const": (0.727073, 0.0731246), "l_gdppc_i": (-0.0124787, 0.00429076),
                                   "l_gdppc_j": (-0.00311931, 0.00471109), "l_dist": (-0.00995239, 0.00565546),
                                   "l_trade_lag1": (0.972416, 0.00228254)}),
    3: (12400, 0.241060, 0.042209, {"const": (2.12080, 0.306492), "l_gdppc_i": (0.375950, 0.0170098),
                                    "l_gdppc_j": (0.847894, 0.0172373), "l_dist": (0.0102965, 0.0235888),
                                    "post2014": (0.105999, 0.0423063), "contig": (2.33099, 0.0904001)}),
    4: (12400, 0.266895, 0.047681, {"const": (1.90768, 0.306333), "l_gdppc_i": (0.336334, 0.0175939),
                                    "l_gdppc_j": (0.818840, 0.0175683), "l_dist": (0.0963617, 0.0266920),
                                    "post2014": (0.107016, 0.0453004), "contig": (2.30016, 0.0898691),
                                    "eu_ij": (0.219489, 0.101989), "brics_ij": (2.95246, 0.173906),
                                    "nato_ij": (0.733007, 0.0744678), "post2022": (-0.111579, 0.0672757)}),
    5: (11904, 0.963869, 2.187397, {"const": (0.507139, 0.0699139), "l_gdppc_i": (-0.00617488, 0.00405856),
                                    "l_gdppc_j": (0.00351081, 0.00435248), "l_dist": (0.00226898, 0.00601669),
                                    "post2014": (-0.106651, 0.0100552), "contig": (0.0138917, 0.0207009),
                                    "eu_ij": (0.00131347, 0.0226820), "brics_ij": (0.0846042, 0.0387061),
                                    "nato_ij": (0.0372981, 0.0166870), "post2022": (0.000956323, 0.0147887),
                                    "l_trade_lag1": (0.972496, 0.00201332)}),
    6: (11408, 0.963900, 1.926261, {"const": (0.637826, 0.0666807), "l_gdppc_i": (-0.00958163, 0.00397499),
                                    "l_gdppc_j": (-0.00306748, 0.00433552), "post2014": (-0.141158, 0.00936854),
                                    "brics_ij": (0.0679019, 0.0382074), "l_dist": (-0.00230551, 0.00475076),
                                    "l_trade_lag1": (0.803298, 0.00947972), "l_trade_lag2": (0.172470, 0.00944067)}),
    7: (10912, 0.963460, 1.886141, {"const": (0.775006, 0.0698317), "l_gdppc_i": (-0.0203031, 0.00427200),
                                    "l_gdppc_j": (-0.0142897, 0.00457774), "post2014": (-0.175779, 0.0103864),
                                    "l_dist": (0.00631671, 0.00522872), "post2022": (0.0353810, 0.0145969),
                                    "nato_ij": (0.0471881, 0.0157489), "l_trade_lag1": (0.776163, 0.00977323),
                                    "l_trade_lag2": (0.0800375, 0.0127077), "l_trade_lag3": (0.121321, 0.00997095)}),
}
SAMPLE_END = {1: 2014, 2: 2014}


def durbin_watson_panel(resid: np.ndarray, pair: np.ndarray) -> float:
    """Статистика Дарбина–Уотсона по панели (как в gretl: разности внутри пар)."""
    num = den = 0.0
    prev_pair = None
    prev_e = 0.0
    for e, p in zip(resid, pair):
        den += e * e
        if p == prev_pair:
            num += (e - prev_e) ** 2
        prev_pair, prev_e = p, e
    return num / den


def main() -> None:
    df = pd.read_csv(DATA / "vkr_panel_renamed.csv").sort_values(["pair_id", "year"]).reset_index(drop=True)
    df["l_trade"] = np.log(df["trade"])
    df["l_gdppc_i"] = np.log(df["gdppc_i"])
    df["l_gdppc_j"] = np.log(df["gdppc_j"])
    df["l_dist"] = np.log(df["dist_vkr"])
    for k in (1, 2, 3):
        df[f"l_trade_lag{k}"] = df.groupby("pair_id")["l_trade"].shift(k)

    specs = {
        1: ["l_gdppc_i", "l_gdppc_j", "l_dist"],
        2: ["l_gdppc_i", "l_gdppc_j", "l_dist", "l_trade_lag1"],
        3: ["l_gdppc_i", "l_gdppc_j", "l_dist", "post2014", "contig"],
        4: ["l_gdppc_i", "l_gdppc_j", "l_dist", "post2014", "contig", "eu_ij", "brics_ij", "nato_ij", "post2022"],
        5: ["l_gdppc_i", "l_gdppc_j", "l_dist", "post2014", "contig", "eu_ij", "brics_ij", "nato_ij", "post2022",
            "l_trade_lag1"],
        6: ["l_gdppc_i", "l_gdppc_j", "post2014", "brics_ij", "l_dist", "l_trade_lag1", "l_trade_lag2"],
        7: ["l_gdppc_i", "l_gdppc_j", "post2014", "l_dist", "post2022", "nato_ij",
            "l_trade_lag1", "l_trade_lag2", "l_trade_lag3"],
    }
    rows = []
    for m, xs in specs.items():
        d = df[df.year <= SAMPLE_END.get(m, 2024)].dropna(subset=["l_trade"] + xs)
        X = sm.add_constant(d[xs])
        res = sm.OLS(d["l_trade"], X).fit()
        n_t, r2_t, dw_t, coef_t = TARGET[m]
        dw = durbin_watson_panel(res.resid.values, d["pair_id"].values)
        print(f"Модель {m}: N={int(res.nobs)} (ВКР {n_t}); R2={res.rsquared:.6f} (ВКР {r2_t}); DW={dw:.6f} (ВКР {dw_t})")
        for v, (c_t, se_t) in coef_t.items():
            c, se = res.params[v], res.bse[v]
            rows.append(dict(model=m, variable=v, coef=c, se=se, pval=res.pvalues[v], coef_gretl=c_t,
                             se_gretl=se_t, diff_coef=c - c_t, nobs=int(res.nobs), r2=res.rsquared, dw=dw))
    out = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "replication_vkr_ols.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(6).to_string(index=False))
    bad = out[out.diff_coef.abs() > 5e-5]
    print("\nмаксимальное |расхождение| коэффициентов:", out.diff_coef.abs().max())
    print("расхождения > 5e-5:", len(bad))


if __name__ == "__main__":
    main()
