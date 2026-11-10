"""Сборка рукописи статьи 1: шаблон (paper/parts_a1/*.md, Jinja2) + результаты → Markdown → DOCX.

Числа и таблицы подставляются из results/a1_*.csv|json, results/replication_vkr_ols.csv и
results/vkr_model8_dpanel.csv (см. code/02_replicate_vkr.py, gretl/03_vkr_dynamic_panel.inp,
code/11_article1_results.py).

Запуск:  python code/12_build_article1.py [--no-docx]
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys

import jinja2
import numpy as np
import pandas as pd

from gravity_common import PAPER, RESULTS

B = importlib.import_module("08_build_paper")     # общие функции форматирования
rn, pct, md_table, NBSP, MINUS = B.rn, B.pct, B.md_table, B.NBSP, B.MINUS


def load_numbers() -> dict:
    v: dict = {}
    rep = pd.read_csv(RESULTS / "replication_vkr_ols.csv")
    m8 = pd.read_csv(RESULTS / "vkr_model8_dpanel.csv").assign(model=8)
    allm = pd.concat([rep[["model", "variable", "coef", "se"]], m8[["model", "variable", "coef", "se"]]])
    v["m"] = {}
    for m, g in allm.groupby("model"):
        v["m"][int(m)] = {r.variable: dict(b=r.coef, s=rn(r.coef, 2), s3=rn(r.coef, 3), se=rn(r.se, 3), pct=pct(r.coef),
                                           x=rn(float(np.exp(r.coef)), 1), x2=rn(float(np.exp(r.coef)), 2)) for r in g.itertuples()}
    v["fit"] = {int(m): dict(n=f"{int(g.nobs.iloc[0]):,}".replace(",", NBSP), r2=rn(g.r2.iloc[0], 3), dw=rn(g.dw.iloc[0], 2),
                             r2p=rn(100 * g.r2.iloc[0], 0)) for m, g in rep.groupby("model")}
    # долгосрочные эффекты в модели 8: b / (1 − ρ)
    rho = v["m"][8]["l_trade_lag1"]["b"]
    v["lr"] = {k: rn(x["b"] / (1 - rho), 2) for k, x in v["m"][8].items() if k not in ("l_trade_lag1", "const")}
    v["lags7"] = rn(sum(v["m"][7][f"l_trade_lag{k}"]["b"] for k in (1, 2, 3)), 2)
    v["lags6"] = rn(sum(v["m"][6][f"l_trade_lag{k}"]["b"] for k in (1, 2)), 2)
    # описательные величины
    pr = pd.read_csv(RESULTS / "a1_partner_ranking.csv").set_index("partner")
    v["p"] = {k: dict(b13=rn(r.bn_2013, 1), b21=rn(r.bn_2021, 1), b24=rn(r.bn_2024, 1), s13=rn(r.share_2013, 1), s24=rn(r.share_2024, 1))
              for k, r in pr.iterrows()}
    py = pd.read_csv(RESULTS / "a1_russia_partner_year.csv").set_index("year")
    tot = py.sum(axis=1)
    v["tot"] = {int(y): rn(x, 0) for y, x in tot.items()}
    v["tot_peak_year"] = int(tot.loc[:2021].idxmax())
    v["top3_share24"] = rn(float(pr.share_2024.head(3).sum()), 0)
    fr = pd.read_csv(RESULTS / "a1_fts_regions.csv").set_index("group")
    fg = pd.read_csv(RESULTS / "a1_fts_goods.csv")
    v["fts"] = {k: dict(e24=rn(r.exp_2024, 1), i24=rn(r.imp_2024, 1), es=rn(r.exp_share_2024, 1), is_=rn(r.imp_share_2024, 1),
                        eg=rn(r.exp_growth, 1, sign=True), ig=rn(r.imp_growth, 1, sign=True)) for k, r in fr.iterrows()}

    def goods(key: str):
        r = fg[fg.group.str.contains(key)].iloc[0]
        return dict(es=rn(r.exp_share_2024, 1), is_=rn(r.imp_share_2024, 1))
    v["g"] = {"mineral": goods("Минеральные"), "metal": goods("Металлы"), "machines": goods("Машины"), "food": goods("Продовольственные"),
              "chem": goods("химической")}
    # прогнозы
    s = json.load(open(RESULTS / "a1_summary.json", encoding="utf-8"))
    f24 = pd.read_csv(RESULTS / "a1_forecast_2024.csv").set_index("partner")
    v["f24"] = dict(n=s["f24"]["n"], act=rn(s["f24"]["sum_actual_bn"], 0), pred=rn(s["f24"]["sum_pred_bn"], 0),
                    gap=rn(100 * (s["f24"]["sum_actual_bn"] / s["f24"]["sum_pred_bn"] - 1), 1), mae=rn(s["f24"]["mae_log"], 2),
                    n_above=s["f24"]["n_above"], n_below=s["f24"]["n"] - s["f24"]["n_above"], n_carried=s["f24"]["n_carried"],
                    p={k: dict(act=rn(r.actual_2024_mn / 1e3, 1), pred=rn(r.pred_2024_mn / 1e3, 1), diff=rn(abs(r.diff_mn) / 1e3, 1),
                               lr=rn(r.log_ratio, 2), pc=rn(100 * (np.exp(r.log_ratio) - 1), 0)) for k, r in f24.iterrows()})
    f25 = pd.read_csv(RESULTS / "a1_forecast_2025.csv").set_index("partner")
    v["f25"] = dict(n=s["f25"]["n"], act24=rn(s["f25"]["sum_actual24_bn"], 0), pred=rn(s["f25"]["sum_pred_bn"], 0),
                    chg=rn(100 * (s["f25"]["sum_pred_bn"] / s["f25"]["sum_actual24_bn"] - 1), 0), has_actual="mae_log" in s["f25"],
                    p={k: dict(pred=rn(r.pred_2025_mn / 1e3, 1), g=rn(r.pred_growth_pct, 0),
                               act=rn(r.actual_2025_mn / 1e3, 1) if "actual_2025_mn" in f25 and pd.notna(r.get("actual_2025_mn")) else "—")
                       for k, r in f25.iterrows()})
    if v["f25"]["has_actual"]:
        t = s["f25"]
        v["f25"].update(n_actual=t["n_actual"], act25=rn(t["sum_actual25_bn"], 0), pred_same=rn(t["sum_pred_same_bn"], 0),
                        act24_same=rn(t["sum_actual24_same_bn"], 0),
                        err_tot=rn(100 * (t["sum_actual25_bn"] / t["sum_pred_same_bn"] - 1), 0, sign=True),
                        chg_act=rn(100 * (t["sum_actual25_bn"] / t["sum_actual24_same_bn"] - 1), 0, sign=True),
                        mae=rn(t["mae_log"], 2), med=rn(t["median_abs_log"], 2), naive_mae=rn(t["naive_mae_log"], 2),
                        naive_med=rn(t["naive_median_abs_log"], 2), n_over=t["n_over"], n_under=t["n_actual"] - t["n_over"],
                        corr=rn(t["corr_growth"], 2))
    return v


def build_tables(v: dict) -> dict:
    T: dict = {}
    d = pd.read_csv(RESULTS / "a1_descriptive.csv")
    d["N"] = d["N"].map(lambda x: f"{int(x)}")
    for c in ["Среднее", "Ст. откл.", "Мин.", "Макс."]:
        d[c] = d[c].map(lambda x: rn(x, 2) if abs(x) < 100 else rn(x, 0))
    T["desc"] = md_table(d, widths=[3.0, 0.8, 1.0, 1.0, 0.8, 1.0])
    for k in ("static", "dynamic"):
        t = pd.read_csv(RESULTS / f"a1_table_{k}.csv", dtype=str).fillna("")
        T[k] = md_table(t, widths=[2.6] + [1.4] * (t.shape[1] - 1))
    pr = pd.read_csv(RESULTS / "a1_partner_ranking.csv").head(12)
    T["partners"] = md_table(pd.DataFrame({"Партнёр": pr.name, "2013": pr.bn_2013.map(lambda x: rn(x, 1)), "2021": pr.bn_2021.map(lambda x: rn(x, 1)),
                                           "2024": pr.bn_2024.map(lambda x: rn(x, 1)), "Доля, 2013, %": pr.share_2013.map(lambda x: rn(x, 1)),
                                           "Доля, 2024, %": pr.share_2024.map(lambda x: rn(x, 1))}), widths=[2.0, 1, 1, 1, 1.2, 1.2])
    fr = pd.read_csv(RESULTS / "a1_fts_regions.csv")
    fr["group"] = fr.group.str.capitalize()
    T["fts"] = md_table(pd.DataFrame({"Группа стран": fr.group, "Экспорт, млрд долл.": fr.exp_2024.map(lambda x: rn(x, 1)),
                                      "Доля, %": fr.exp_share_2024.map(lambda x: rn(x, 1)), "К 2023 г., %": fr.exp_growth.map(lambda x: rn(x, 1, sign=True)),
                                      "Импорт, млрд долл.": fr.imp_2024.map(lambda x: rn(x, 1)), "Доля, % ": fr.imp_share_2024.map(lambda x: rn(x, 1)),
                                      "К 2023 г., % ": fr.imp_growth.map(lambda x: rn(x, 1, sign=True))}), widths=[1.6, 1.2, 0.8, 1, 1.2, 0.8, 1])
    f = pd.read_csv(RESULTS / "a1_forecast_2025.csv").sort_values("actual_2024_mn", ascending=False)
    cols = {"Партнёр": f.name, "Факт 2024": (f.actual_2024_mn / 1e3).map(lambda x: rn(x, 1)), "Прогноз 2025": (f.pred_2025_mn / 1e3).map(lambda x: rn(x, 1)),
            "Прогноз к 2024, %": f.pred_growth_pct.map(lambda x: rn(x, 0, sign=True))}
    if "actual_2025_mn" in f:
        cols["Факт 2025"] = (f.actual_2025_mn / 1e3).map(lambda x: rn(x, 1) if pd.notna(x) else "—")
        cols["Факт к прогнозу, %"] = f.log_err.map(lambda x: rn(100 * (np.exp(x) - 1), 0, sign=True) if pd.notna(x) else "—")
    T["f25"] = md_table(pd.DataFrame(cols), widths=[2.0] + [1.1] * (len(cols) - 1))
    return T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-docx", action="store_true")
    args = ap.parse_args()
    v, T = load_numbers(), None
    T = build_tables(v)
    env = jinja2.Environment(variable_start_string="<<", variable_end_string=">>", block_start_string="<%", block_end_string="%>",
                             comment_start_string="<#", comment_end_string="#>", undefined=jinja2.StrictUndefined,
                             trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    env.filters["en"] = B.en
    env.filters["p"] = B.pc
    parts = sorted((PAPER / "parts_a1").glob("*.md"))
    src = "\n\n".join(open(f, encoding="utf-8").read().strip("\n") for f in parts)
    md = env.from_string(src).render(v=v, T=T, NBSP=NBSP, MINUS=MINUS)
    out_md = PAPER / "article1_ru.md"
    out_md.write_text(md, encoding="utf-8")
    print("Markdown:", out_md, f"({len(md.split())} слов)")
    if not args.no_docx:
        subprocess.run([sys.executable, str(PAPER.parent / "code" / "09_make_docx.py"), "--md", str(out_md),
                        "--out", str(PAPER / "article1_ru.docx")], check=True)


if __name__ == "__main__":
    main()
