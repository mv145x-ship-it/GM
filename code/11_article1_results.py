"""Расчёты, таблицы и рисунки для статьи 1 (эмпирическая гравитационная модель по материалам ВКР).

Воспроизводит на исходных данных ВКР (data/vkr_panel_renamed.csv):
  * описательную статистику и динамику оборота России с партнёрами выборки;
  * модели 1–7 (пулированный МНК; results/replication_vkr_ols.csv — скрипт 02) и модель 8
    (системный ОММ; results/vkr_model8_dpanel.csv — gretl/03_vkr_dynamic_panel.inp);
  * расчёт «ожидаемого» оборота 2024 г. по модели 2 (оценена на данных до 2014 г.);
  * прогноз оборота на 2025 г. по модели 8 (прогнозы ВВП на душу населения — МВФ, из файла ВКР)
    и, если доступна обновлённая панель (data/gravity_panel_updated.csv), сравнение с фактом 2025 г.

Выход: results/a1_*.csv, paper/figures/a1_*.png|pdf.

Запуск:  python code/02_replicate_vkr.py && python code/11_article1_results.py
"""
from __future__ import annotations

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gravity_common import COALITION_2022, COUNTRIES, DATA, NAME2ISO, RESULTS, ROOT, RU_NAME

F = importlib.import_module("07_figures")       # общий стиль рисунков

FORECAST_XLSX = ROOT / "Данные для прогноза 2014, 2025.xlsx"
FTS_REGIONS = ROOT / "Экспорт импорт по группам стран.xlsx"
FTS_GOODS = ROOT / "Экспорт импорт товаров.xlsx"
M8_NOBS, M8_INSTR = 11904, 305                     # из вывода gretl (dpanel)


def stars(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def cell(c: float, se: float) -> str:
    from scipy import stats
    p = 2 * stats.norm.sf(abs(c / se))
    return f"{c:.3f}{stars(p)} ({se:.3f})"


def load_vkr() -> pd.DataFrame:
    d = pd.read_csv(DATA / "vkr_panel_renamed.csv").sort_values(["pair_id", "year"]).reset_index(drop=True)
    d["rus"] = (d.iso3_i == "RUS") | (d.iso3_j == "RUS")
    d["partner"] = np.where(d.iso3_i == "RUS", d.iso3_j, np.where(d.iso3_j == "RUS", d.iso3_i, ""))
    return d


# ----------------------------------------------------------------------------
def descriptive(d: pd.DataFrame) -> None:
    v = pd.DataFrame({"Товарооборот, млн долл.": d.trade / 1e3, "ВВП на душу населения, страна i, долл.": d.gdppc_i,
                      "ВВП на душу населения, страна j, долл.": d.gdppc_j, "Расстояние, км": d.dist_vkr,
                      "Общая граница": d.contig, "ЕС (обе страны)": d.eu_ij, "БРИКС (обе страны)": d.brics_ij,
                      "НАТО (обе страны)": d.nato_ij, "Период с 2014 г.": d.post2014, "Период с 2022 г.": d.post2022})
    s = v.describe().T[["count", "mean", "std", "min", "max"]]
    s.columns = ["N", "Среднее", "Ст. откл.", "Мин.", "Макс."]
    s.index.name = "Переменная"
    s.reset_index().to_csv(RESULTS / "a1_descriptive.csv", index=False)

    # оборот России с партнёрами выборки по годам
    r = d[d.rus].copy()
    r["bn"] = r.trade / 1e6
    piv = r.pivot_table(index="year", columns="partner", values="bn", aggfunc="sum")
    piv.to_csv(RESULTS / "a1_russia_partner_year.csv")
    tot = piv.sum(axis=1)
    top24 = piv.loc[2024].sort_values(ascending=False)
    rank = pd.DataFrame({"partner": top24.index, "name": [RU_NAME[p] for p in top24.index], "bn_2013": piv.loc[2013, top24.index].values,
                         "bn_2021": piv.loc[2021, top24.index].values, "bn_2024": top24.values,
                         "share_2013": 100 * piv.loc[2013, top24.index].values / tot[2013],
                         "share_2024": 100 * top24.values / tot[2024]})
    rank.to_csv(RESULTS / "a1_partner_ranking.csv", index=False)

    # рисунок 1: суммарный оборот и крупнейшие партнёры
    fig, axes = plt.subplots(1, 2, figsize=(F.W, 2.7), gridspec_kw={"wspace": 0.28})
    ax = axes[0]
    ax.plot(tot.index, tot.values, color=F.INK2, lw=1.6)
    F.end_dot(ax, tot.index[-1], tot.values[-1], F.INK2)
    F.period_lines(ax)
    ax.set_title("а) Оборот с 31 партнёром, млрд долл.", loc="left")
    ax.set_ylim(0, None)
    F.style_axes(ax)
    ax = axes[1]
    show = ["CHN", "DEU", "TUR", "IND", "BLR", "KAZ"]
    cols = [F.C_OTHER, F.C_COAL, F.C_UKR, "#7a5cc9", F.C_GRAY, F.INK2]
    marks = ["o", "s", "^", "D", "v", "P"]
    for p, c, m in zip(show, cols, marks):
        ax.plot(piv.index, piv[p], color=c, lw=1.4, label=RU_NAME[p], marker=m, ms=3.0, markevery=4, mec=F.SURFACE, mew=0.6)
    F.period_lines(ax)
    ax.set_title("б) Крупнейшие партнёры, млрд долл.", loc="left")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, ncol=2, loc="upper left", handlelength=1.6, columnspacing=1.0)
    F.style_axes(ax)
    F.save(fig, "a1_fig1_dynamics", piv[show].assign(total=tot).reset_index())


def fts_tables() -> None:
    """Структура экспорта и импорта России по данным ФТС (файлы ВКР), 2023–2024 гг."""
    for path, name in ((FTS_REGIONS, "a1_fts_regions"), (FTS_GOODS, "a1_fts_goods")):
        raw = pd.read_excel(path, header=None)
        rows = []
        for _, r in raw.iterrows():
            lab = r[0]
            if not isinstance(lab, str) or not isinstance(r[1], (int, float)) or pd.isna(r[1]):
                continue
            rows.append({"group": lab.strip(), "exp_2023": r[1], "exp_2024": r[2], "imp_2023": r[4], "imp_2024": r[5]})
        t = pd.DataFrame(rows)
        tot = t.iloc[0]
        t["exp_share_2024"] = 100 * t.exp_2024 / tot.exp_2024
        t["imp_share_2024"] = 100 * t.imp_2024 / tot.imp_2024
        t["exp_growth"] = 100 * (t.exp_2024 / t.exp_2023 - 1)
        t["imp_growth"] = 100 * (t.imp_2024 / t.imp_2023 - 1)
        t.to_csv(RESULTS / f"{name}.csv", index=False)


# ----------------------------------------------------------------------------
def model_tables() -> None:
    rep = pd.read_csv(RESULTS / "replication_vkr_ols.csv")
    m8 = pd.read_csv(RESULTS / "vkr_model8_dpanel.csv")
    lab = {"const": "Константа", "l_gdppc_i": "ln ВВП на душу населения, страна i", "l_gdppc_j": "ln ВВП на душу населения, страна j",
           "l_dist": "ln расстояния", "contig": "Общая граница", "eu_ij": "ЕС (обе страны)", "brics_ij": "БРИКС (обе страны)",
           "nato_ij": "НАТО (обе страны)", "post2014": "Период с 2014 г. (D1)", "post2022": "Период с 2022 г. (D6)",
           "l_trade_lag1": "ln оборота (t−1)", "l_trade_lag2": "ln оборота (t−2)", "l_trade_lag3": "ln оборота (t−3)"}
    order = ["l_gdppc_i", "l_gdppc_j", "l_dist", "contig", "eu_ij", "brics_ij", "nato_ij", "post2014", "post2022",
             "l_trade_lag1", "l_trade_lag2", "l_trade_lag3", "const"]
    names = {1: "(1) МНК, 2000–2014", 2: "(2) МНК с лагом, 2000–2014", 3: "(3) МНК", 4: "(4) МНК", 5: "(5) МНК с лагом",
             6: "(6) МНК, два лага", 7: "(7) МНК, три лага", 8: "(8) Системный ОММ"}
    for tag, models in (("a1_table_static", [1, 3, 4]), ("a1_table_dynamic", [2, 5, 6, 7, 8])):
        rows = []
        for v in order:
            row = {"Переменная": lab[v]}
            used = False
            for m in models:
                if m == 8:
                    s = m8[m8.variable == v]
                else:
                    s = rep[(rep.model == m) & (rep.variable == v)]
                row[names[m]] = cell(float(s.coef.iloc[0]), float(s.se.iloc[0])) if len(s) else ""
                used |= bool(len(s))
            if used:
                rows.append(row)
        foot = {"Число наблюдений": {}, "R²": {}, "Статистика Дарбина — Уотсона": {}}
        for m in models:
            if m == 8:
                foot["Число наблюдений"][names[m]] = str(M8_NOBS)
                foot["R²"][names[m]] = "—"
                foot["Статистика Дарбина — Уотсона"][names[m]] = "—"
            else:
                s = rep[rep.model == m].iloc[0]
                foot["Число наблюдений"][names[m]] = str(int(s.nobs))
                foot["R²"][names[m]] = f"{s.r2:.3f}"
                foot["Статистика Дарбина — Уотсона"][names[m]] = f"{s.dw:.3f}"
        for k, vv in foot.items():
            rows.append({"Переменная": k, **vv})
        pd.DataFrame(rows).fillna("").to_csv(RESULTS / f"{tag}.csv", index=False)


# ----------------------------------------------------------------------------
def forecast_2024(d: pd.DataFrame) -> pd.DataFrame:
    """Модель 2 (МНК с лагом, оценена на 2000–2014 гг.): расчёт оборота 2024 г. по факту 2023 г."""
    rep = pd.read_csv(RESULTS / "replication_vkr_ols.csv")
    b = rep[rep.model == 2].set_index("variable").coef
    r = d[d.rus].copy()
    r["lag"] = r.groupby("pair_id").trade.shift(1)
    x = r[r.year == 2024].copy()
    x["pred"] = np.exp(b["const"] + b["l_gdppc_i"] * np.log(x.gdppc_i) + b["l_gdppc_j"] * np.log(x.gdppc_j)
                       + b["l_dist"] * np.log(x.dist_vkr) + b["l_trade_lag1"] * np.log(x["lag"]))
    out = pd.DataFrame({"partner": x.partner, "name": x.partner.map(RU_NAME), "actual_2023_mn": x["lag"] / 1e3,
                        "actual_2024_mn": x.trade / 1e3, "pred_2024_mn": x.pred / 1e3})
    out["diff_mn"] = out.actual_2024_mn - out.pred_2024_mn
    out["log_ratio"] = np.log(out.actual_2024_mn / out.pred_2024_mn)
    out["carried"] = (x.trade.values == x["lag"].values)        # значение 2024 г. совпадает с 2023 г.
    out = out.sort_values("log_ratio").reset_index(drop=True)
    # сверка с расчётом в файле ВКР
    chk = pd.read_excel(FORECAST_XLSX, sheet_name="итог расчёта")
    chk.columns = ["country", "actual", "pred", "diff"]
    chk["partner"] = chk.country.str.strip().map(NAME2ISO)
    mm = out.merge(chk, on="partner")
    print("прогноз 2024: макс. отн. расхождение с файлом ВКР:", float((mm.pred_2024_mn * 1e3 / mm.pred - 1).abs().max()))
    out.to_csv(RESULTS / "a1_forecast_2024.csv", index=False)
    return out


def forecast_2025(d: pd.DataFrame) -> pd.DataFrame:
    """Модель 8 (системный ОММ): прогноз оборота на 2025 г.; ВВП на душу населения — прогноз МВФ из файла ВКР."""
    b = pd.read_csv(RESULTS / "vkr_model8_dpanel.csv").set_index("variable").coef
    e = pd.read_excel(FORECAST_XLSX, sheet_name="для 8 модели")
    e = e.rename(columns={"Сk": "country", 2024: "y2024", "D1sanctions": "d1", "D4 BRICS": "brics", "прогноз 2025": "pred_vkr"})
    e["partner"] = e.country.str.strip().map(NAME2ISO)
    e["pred"] = np.exp(b["const"] + b["l_trade_lag1"] * np.log(e.y2024) + b["l_gdppc_i"] * np.log(e.vvpi)
                       + b["l_gdppc_j"] * np.log(e.vvpk) + b["l_dist"] * np.log(e.Rik) + b["post2014"] * e.d1 + b["brics_ij"] * e.brics)
    print("прогноз 2025: макс. отн. расхождение с файлом ВКР:", float((e.pred / e.pred_vkr - 1).abs().max()))
    out = pd.DataFrame({"partner": e.partner, "name": e.partner.map(RU_NAME), "actual_2024_mn": e.y2024 / 1e3,
                        "pred_2025_mn": e.pred / 1e3, "brics": e.brics})
    out["pred_growth_pct"] = 100 * (out.pred_2025_mn / out.actual_2024_mn - 1)
    upd = DATA / "gravity_panel_updated.csv"
    if upd.exists():
        u = pd.read_csv(upd)
        u = u[(u.rus_pair == 1) & (u.year == 2025)][["partner", "trade", "trade_src", "mirror_ratio"]]
        out = out.merge(u.rename(columns={"trade": "actual_2025_mn", "trade_src": "src_2025"}), on="partner", how="left")
        out["log_err"] = np.log(out.actual_2025_mn / out.pred_2025_mn)
        out["naive_log_err"] = np.log(out.actual_2025_mn / out.actual_2024_mn)     # «наивный» прогноз: оборот 2025 = оборот 2024
    out.to_csv(RESULTS / "a1_forecast_2025.csv", index=False)
    return out


def forecast_figures(f24: pd.DataFrame, f25: pd.DataFrame) -> None:
    def bars(ax, df, col, title):
        df = df.sort_values(col)
        colors = [F.C_COAL if p in COALITION_2022 else (F.C_UKR if p == "UKR" else F.C_OTHER) for p in df.partner]
        pos = np.arange(len(df))
        ax.barh(pos, df[col], color=colors, height=0.72, edgecolor=F.SURFACE, linewidth=0.6)
        ax.axvline(0, color=F.AXIS, lw=0.8)
        ax.set_xlabel(title)
        F.style_axes(ax, grid_axis="x")
        ax.set_yticks(pos)
        ax.set_yticklabels(df.name, fontsize=7, color=F.INK)
        ax.margins(y=0.01)

    from matplotlib.patches import Patch
    handles = [Patch(color=F.C_COAL, label="страны, вводившие ограничения"), Patch(color=F.C_OTHER, label="прочие партнёры"),
               Patch(color=F.C_UKR, label="Украина")]
    fig, ax = plt.subplots(figsize=(F.W * 0.72, 5.0))
    bars(ax, f24, "log_ratio", "ln(факт 2024 / расчёт по модели 2)")
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 0.94))
    F.save(fig, "a1_fig2_forecast_2024", f24)
    if "log_err" in f25:
        g = f25.dropna(subset=["log_err"])
        fig, ax = plt.subplots(figsize=(F.W * 0.72, 5.0))
        bars(ax, g, "log_err", "ln(факт 2025 / прогноз по модели 8)")
        fig.legend(handles=handles, frameon=False, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 0.94))
        F.save(fig, "a1_fig3_forecast_2025", g)


def forecast_summary(f24: pd.DataFrame, f25: pd.DataFrame) -> None:
    import json
    s: dict = {"f24": {"n": int(len(f24)), "sum_actual_bn": float(f24.actual_2024_mn.sum() / 1e3), "sum_pred_bn": float(f24.pred_2024_mn.sum() / 1e3),
                       "mae_log": float(f24.log_ratio.abs().mean()), "n_above": int((f24.log_ratio > 0).sum()),
                       "n_carried": int(f24.carried.sum())}}
    s["f25"] = {"n": int(len(f25)), "sum_actual24_bn": float(f25.actual_2024_mn.sum() / 1e3), "sum_pred_bn": float(f25.pred_2025_mn.sum() / 1e3)}
    if "log_err" in f25:
        g = f25.dropna(subset=["log_err"])
        s["f25"].update({"n_actual": int(len(g)), "sum_actual25_bn": float(g.actual_2025_mn.sum() / 1e3),
                         "sum_pred_same_bn": float(g.pred_2025_mn.sum() / 1e3), "sum_actual24_same_bn": float(g.actual_2024_mn.sum() / 1e3),
                         "mae_log": float(g.log_err.abs().mean()), "median_abs_log": float(g.log_err.abs().median()),
                         "naive_mae_log": float(g.naive_log_err.abs().mean()), "naive_median_abs_log": float(g.naive_log_err.abs().median()),
                         "n_over": int((g.log_err < 0).sum()),
                         "corr_growth": float(np.corrcoef(np.log(g.pred_2025_mn / g.actual_2024_mn), g.naive_log_err)[0, 1])})
    json.dump(s, open(RESULTS / "a1_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(s, ensure_ascii=False, indent=1))


def main() -> None:
    d = load_vkr()
    descriptive(d)
    fts_tables()
    model_tables()
    f24 = forecast_2024(d)
    f25 = forecast_2025(d)
    forecast_figures(f24, f25)
    forecast_summary(f24, f25)


if __name__ == "__main__":
    main()
