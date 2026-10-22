"""Рисунки для статьи (PNG 300 dpi + PDF; исходные данные каждого рисунка — CSV).

Оформление: светлая поверхность, тонкие линии (1,6 pt), волосяные сплошные сетки,
подписи — цветом текста (не цветом ряда), фиксированные цвета сущностей
(коалиция — синий, прочие партнёры — оранжевый, Украина — бирюзовый), маркеры
дублируют цвет (печать в градациях серого). Палитра проверена скриптом
validate_palette.js навыка dataviz (слоты 1–3: все проверки пройдены; для
бирюзового — предупреждение о контрасте: ряд подписан напрямую, значения даны в таблицах).

Запуск:  python code/07_figures.py
"""
from __future__ import annotations

import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from gravity_common import COALITION_2022, DATA, FIGS, RESULTS, RU_NAME
from gravity_models import load_panel

# ---------------------------------------------------------------- стиль
SURFACE = "#ffffff"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
C_COAL, C_OTHER, C_UKR = "#2a78d6", "#eb6834", "#1baf7a"
C_GRAY = "#9a9892"
mpl.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "axes.axisbelow": True, "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round", "axes.unicode_minus": True, "pdf.fonttype": 42,
})
W = 6.5  # ширина рисунка, дюймы
MINUS = lambda s: s.replace("-", "−")


def style_axes(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis)
    ax.tick_params(length=2.5, width=0.6, color=AXIS)


def save(fig, name: str, data: pd.DataFrame | None = None):
    FIGS.mkdir(parents=True, exist_ok=True)
    (FIGS / "data").mkdir(exist_ok=True)
    fig.savefig(FIGS / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    if data is not None:
        data.to_csv(FIGS / "data" / f"{name}.csv", index=False)
    plt.close(fig)
    print("сохранён", name)


def end_dot(ax, x, y, color, marker="o"):
    ax.plot([x], [y], marker=marker, ms=5.0, color=color, mec=SURFACE, mew=1.0, zorder=4, ls="none")


def period_lines(ax, labels=False, ytop=None):
    ax.axvline(2013.5, color=AXIS, lw=0.8)
    ax.axvline(2021.5, color=AXIS, lw=0.8)
    if labels:
        ax.text(2013.6, ytop, "2014", color=INK2, fontsize=7.8, va="bottom")
        ax.text(2021.6, ytop, "2022", color=INK2, fontsize=7.8, va="bottom")


# ================================================================ Рис. 1
def fig1_descriptive(d: pd.DataFrame):
    r = d[d.rus_pair == 1].copy()
    r["group"] = r.partner.map(lambda p: "coalition" if p in COALITION_2022 else ("ukraine" if p == "UKR" else "other"))
    g = r.groupby(["year", "group"]).trade.sum().unstack().fillna(0) / 1e3  # млрд долл.
    tot = g.sum(axis=1)
    share = 100 * g["coalition"] / tot
    wdist = r.groupby("year").apply(lambda x: np.average(x.dist, weights=x.trade), include_groups=False) / 1e3
    years = g.index.values

    fig = plt.figure(figsize=(W, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.45, wspace=0.30)
    ax = fig.add_subplot(gs[0, :])
    bottom = np.zeros(len(years))
    for key, col, lab in [("coalition", C_COAL, "Страны санкционной коалиции"), ("other", C_OTHER, "Прочие партнёры"),
                          ("ukraine", C_UKR, "Украина")]:
        ax.bar(years, g[key].values, bottom=bottom, width=0.72, color=col, label=lab, linewidth=0.8, edgecolor=SURFACE)
        bottom += g[key].values
    ax.set_ylabel("Товарооборот, млрд долл. США")
    ax.set_xlim(1999.3, 2024.7); ax.set_ylim(0, 640)
    style_axes(ax)
    period_lines(ax, labels=True, ytop=615)
    ax.legend(frameon=False, loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.14), handlelength=1.2)
    ax.set_title("а) Торговля России с 31 партнёром выборки по группам", loc="left", color=INK, pad=24)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(years, share, color=C_COAL, lw=1.7)
    end_dot(ax2, years[-1], share.iloc[-1], C_COAL)
    ax2.text(years[-1] + 0.7, share.iloc[-1], f"{share.iloc[-1]:.0f}%", va="center", fontsize=8, color=INK)
    ax2.text(years[0], share.iloc[0] + 3, f"{share.iloc[0]:.0f}%", va="bottom", fontsize=8, color=INK, ha="left")
    ax2.set_ylim(0, 80); ax2.set_xlim(1999.3, 2027.2)
    ax2.set_ylabel("Доля коалиции, %")
    period_lines(ax2); style_axes(ax2)
    ax2.set_title("б) Доля стран коалиции в торговле", loc="left", color=INK)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(years, wdist.values, color=C_OTHER, lw=1.7)
    end_dot(ax3, years[-1], wdist.iloc[-1], C_OTHER, "s")
    ax3.text(years[-1] + 0.7, wdist.iloc[-1], f"{wdist.iloc[-1]:.1f}", va="center", fontsize=8, color=INK)
    ax3.text(years[0], wdist.iloc[0] + 0.15, f"{wdist.iloc[0]:.1f}", va="bottom", fontsize=8, color=INK, ha="left")
    ax3.set_ylim(1, 6); ax3.set_xlim(1999.3, 2027.2)
    ax3.set_ylabel("Тыс. км")
    period_lines(ax3); style_axes(ax3)
    ax3.set_title("в) Средневзвешенное расстояние до партнёров", loc="left", color=INK)
    for a in (ax, ax2, ax3):
        a.set_xticks([2000, 2005, 2010, 2015, 2020, 2024])
    out = g.copy(); out["total"] = tot; out["share_coalition_pct"] = share; out["wavg_dist_thousand_km"] = wdist
    save(fig, "fig1_descriptive", out.reset_index())
    return out


# ================================================================ Рис. 2
def fig2_spec_path():
    long = pd.read_csv(RESULTS / "all_models_long.csv")
    rep = pd.read_csv(RESULTS / "replication_vkr_ols.csv")
    v4 = rep[rep.model == 4].set_index("variable")
    specs = [
        ("(1) МНК ВКР: временные дамми", None),
        ("(2) МНК ln T, год. ЭФ", "T1 (2) МНК, ln T, год. ЭФ"),
        ("(3) PPML, год. ЭФ", "T1 (3) PPML, год. ЭФ"),
        ("(4) PPML, страновые + год. ЭФ", "T1 (4) PPML, страновые + год. ЭФ"),
        ("(5) PPML, страна×год ЭФ", "T1 (5) PPML, страна×год ЭФ"),
        ("(6) PPML, страна×год + парные ЭФ", "T1 (6) PPML, страна×год + парные ЭФ"),
    ]
    rows = []
    for lab, key in specs:
        for var, vk in [("sanc14_ij", "post2014"), ("sanc22_ij", "post2022")]:
            if key is None:
                c, s = v4.loc[vk, "coef"], v4.loc[vk, "se"]
            else:
                t = long[(long.model == key) & (long.variable == var)].iloc[0]
                c, s = t.coef, t.se
            rows.append(dict(spec=lab, var=var, coef=c, se=s, lo=c - 1.96 * s, hi=c + 1.96 * s))
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), sharey=True, gridspec_kw={"wspace": 0.08})
    ys = np.arange(len(specs))[::-1]
    cfg = {"sanc14_ij": (-1.35, 0.95, 0.62, "а) Режим 2014 г. (t ≥ 2014)"),
           "sanc22_ij": (-2.3, 1.05, 0.62, "б) Режим 2022 г. (t ≥ 2022)")}
    for ax, var in zip(axes, ["sanc14_ij", "sanc22_ij"]):
        lo_x, hi_x, _, title = cfg[var]
        sub = df[df["var"] == var].reset_index(drop=True)
        xlab = hi_x - 0.05
        for k, row in sub.iterrows():
            col = C_GRAY if k == 0 else C_COAL
            ax.plot([row.lo, row.hi], [ys[k]] * 2, color=col, lw=1.6)
            ax.plot(row.coef, ys[k], marker="o" if k else "s", ms=5.5, color=col, mec=SURFACE, mew=1.0, zorder=3, ls="none")
            ax.text(xlab, ys[k], MINUS(f"{row.coef:+.2f}"), va="center", ha="right", fontsize=7.8, color=INK)
        ax.axvline(0, color=INK2, lw=0.8)
        style_axes(ax, "x")
        ax.set_xlim(lo_x, hi_x)
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Лог. пункты, 95% ДИ")
    axes[0].set_yticks(ys); axes[0].set_yticklabels([s[0] for s in specs], color=INK2)
    save(fig, "fig2_specification_path", df)


# ================================================================ Рис. 3
def fig3_event_study():
    ev = pd.read_csv(RESULTS / "event_study_main.csv")
    ev = ev[ev.window == "2000-2024"]
    coal = ev[ev.group.str.contains("коалиция")].copy()
    ukr = ev[ev.group.str.contains("Украина")].copy()

    def add_base(df):
        b = pd.DataFrame([dict(year=2013, coef=0.0, se=0.0, lo95=0.0, hi95=0.0)])
        return pd.concat([df[["year", "coef", "se", "lo95", "hi95"]], b]).sort_values("year")

    coal, ukr = add_base(coal), add_base(ukr)
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.30})
    for ax, df, col, title, ylim in [(axes[0], coal, C_COAL, "а) Россия × страны коалиции 2014 г. (14 пар)", (-3.4, 0.9)),
                                     (axes[1], ukr, C_UKR, "б) Россия × Украина", (-9.8, 1.3))]:
        if col == C_COAL:
            ax.axvspan(2008.5, 2013.5, color="#f0efec", lw=0)
        ax.fill_between(df.year, df.lo95, df.hi95, color=col, alpha=0.14, lw=0)
        ax.plot(df.year, df.coef, color=col, lw=1.6)
        ax.plot(df.year, df.coef, ls="none", marker="o", ms=3.0, color=col, mew=0)
        ax.axhline(0, color=INK2, lw=0.8)
        period_lines(ax)
        ax.set_ylim(*ylim); ax.set_xlim(1999.3, 2027.6)
        ax.set_xticks([2000, 2005, 2010, 2015, 2020, 2024]); style_axes(ax)
        ax.set_title(title, loc="left", color=INK, fontsize=8.5)
        ax.set_ylabel("Лог. пункты относительно 2013 г.")
        last = df.iloc[-1]
        end_dot(ax, 2024, last.coef, col)
        ax.text(2024.7, last.coef, MINUS(f"{100*(np.exp(last.coef)-1):.0f}%"), fontsize=8, color=INK, va="center")
    axes[0].text(2009.0, -3.3, "окно «параллельных\nтрендов»", fontsize=7.2, color=INK2, va="bottom")
    save(fig, "fig3_event_study", pd.concat([coal.assign(group="coalition"), ukr.assign(group="ukraine")]))


# ================================================================ Рис. 4
def fig4_subgroups():
    ev = pd.read_csv(RESULTS / "event_study_subgroups.csv")
    coal_groups = [("ЕС-8", "ЕС-8"), ("США, Канада, Австралия, Великобритания", "США, Канада, Австралия,\nВеликобритания"),
                   ("Япония, Норвегия", "Япония, Норвегия")]
    other_groups = ["Китай", "Индия", "Турция", "ЕАЭС и Кавказ", "Залив и Египет"]
    fig = plt.figure(figsize=(W, 4.0))
    gs = fig.add_gridspec(2, 5, hspace=0.62, wspace=0.34)
    row0 = [fig.add_subplot(gs[0, k]) for k in range(3)]
    row1 = [fig.add_subplot(gs[1, k]) for k in range(5)]
    note = fig.add_subplot(gs[0, 3:5]); note.axis("off")
    panels = [(ax, g, t, C_COAL) for ax, (g, t) in zip(row0, coal_groups)] + [(ax, g, g, C_OTHER) for ax, g in zip(row1, other_groups)]
    for ax, g, title, col in panels:
        df = ev[ev.group == g].copy()
        df = pd.concat([df[["year", "coef", "lo95", "hi95"]], pd.DataFrame([dict(year=2013, coef=0, lo95=0, hi95=0)])]).sort_values("year")
        ax.fill_between(df.year, df.lo95, df.hi95, color=col, alpha=0.14, lw=0)
        ax.plot(df.year, df.coef, color=col, lw=1.4)
        ax.plot(df.year, df.coef, ls="none", marker="o", ms=2.4, color=col, mew=0)
        ax.axhline(0, color=INK2, lw=0.7)
        period_lines(ax)
        ax.set_xlim(2008.3, 2024.7); ax.set_xticks([2010, 2016, 2022])
        ax.set_title(title, loc="left", fontsize=7.8, color=INK)
        style_axes(ax)
    for row in (row0, row1):
        lo = min(a.get_ylim()[0] for a in row); hi = max(a.get_ylim()[1] for a in row)
        for a in row:
            a.set_ylim(lo, hi)
    for a in row0[1:] + row1[1:]:
        a.tick_params(labelleft=True)
    row0[0].set_ylabel("Лог. пункты"); row1[0].set_ylabel("Лог. пункты")
    note.text(0.02, 0.95, "Синий — подгруппы коалиции\n(база: прочие российские партнёры).\n"
              "Оранжевый — подгруппы «прочих»\n(база: Аргентина, Бразилия,\nИндонезия, Мексика, ЮАР).\n"
              "Полоса — 95% ДИ; 2013 г. = 0.", fontsize=7.6, color=INK2, va="top", transform=note.transAxes)
    save(fig, "fig4_event_study_subgroups", ev)


# ================================================================ Рис. 5
def fig5_partner_shifts():
    ps = pd.read_csv(RESULTS / "partner_specific_shifts.csv")
    ps["grp"] = ps.partner.map(lambda p: "coalition" if p in COALITION_2022 else ("ukraine" if p == "UKR" else "other"))
    order = ps[ps.period == "2022–2024"].sort_values("coef").partner.tolist()
    ys = {p: k for k, p in enumerate(order)}
    colmap = {"coalition": C_COAL, "other": C_OTHER, "ukraine": C_UKR}
    fig, axes = plt.subplots(1, 2, figsize=(W, 6.0), sharey=True, gridspec_kw={"wspace": 0.06})
    for ax, per, xlim in zip(axes, ["2014–2021", "2022–2024"], [(-1.2, 1.1), (-4.6, 4.0)]):
        sub = ps[ps.period == per]
        for _, r in sub.iterrows():
            y = ys[r.partner]; col = colmap[r.grp]
            ax.plot([max(r.lo95, xlim[0]), min(r.hi95, xlim[1])], [y, y], color=col, lw=1.3)
            ax.plot(r.coef, y, "o", ms=4.6, color=col, mec=SURFACE, mew=0.8, zorder=3)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_xlim(*xlim); style_axes(ax, "x")
        ax.set_title(("а) " if per == "2014–2021" else "б) ") + per, loc="left", color=INK)
        ax.set_xlabel("Лог. пункты, 95% ДИ")
        ax.set_ylim(-0.8, len(order) - 0.2)
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([RU_NAME[p] for p in order], color=INK2, fontsize=7.6)
    handles = [Line2D([0], [0], marker="o", color=c, lw=1.3, ms=5, mec=SURFACE, label=l)
               for c, l in [(C_COAL, "Страны коалиции"), (C_OTHER, "Прочие партнёры"), (C_UKR, "Украина")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.025))
    save(fig, "fig5_partner_shifts", ps)


# ================================================================ Рис. 6
def fig6_counterfactual():
    cf = pd.read_csv(RESULTS / "counterfactual_group_year.csv")
    main = cf[cf.spec.str.startswith("основной")]
    alts = cf[cf.spec.str.startswith(("основной", "альт"))]
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.1), gridspec_kw={"wspace": 0.28})
    for ax, grp, col, title in [(axes[0], "coalition", C_COAL, "а) Страны коалиции"), (axes[1], "other", C_OTHER, "б) Прочие партнёры")]:
        m = main[main.group == grp].sort_values("year")
        a = alts[alts.group == grp].groupby("year").pred.agg(["min", "max"]) / 1e3
        ax.fill_between(a.index, a["min"], a["max"], color=C_GRAY, alpha=0.20, lw=0)
        ax.plot(m.year, m.pred / 1e3, color=C_GRAY, lw=1.6)
        ax.plot(m.year, m.trade / 1e3, color=col, lw=1.7)
        ax.plot(m.year, m.trade / 1e3, ls="none", marker="o", ms=3.0, color=col, mew=0)
        ax.set_xlim(2013.5, 2026.4); ax.set_xticks([2014, 2016, 2018, 2020, 2022, 2024])
        ax.axvline(2021.5, color=AXIS, lw=0.8)
        style_axes(ax); ax.set_ylim(bottom=0)
        ax.set_ylabel("Млрд долл. США"); ax.set_title(title, loc="left", color=INK)
        last = m.iloc[-1]
        end_dot(ax, 2024, last.trade / 1e3, col)
        ax.text(2024.5, last.trade / 1e3, f"{last.trade / last.pred:.2f}×".replace(".", ","), fontsize=8, color=INK, va="center")
    handles = [Line2D([0], [0], color=C_GRAY, lw=1.7, label="Гравитационный ориентир (основная спецификация)"),
               mpl.patches.Patch(facecolor=C_GRAY, alpha=0.25, label="Диапазон спецификаций ориентира"),
               Line2D([0], [0], color=INK2, lw=1.7, marker="o", ms=3.5, label="Фактический оборот (цвет группы)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07), fontsize=7.6)
    save(fig, "fig6_counterfactual", cf)


# ================================================================ Рис. 7
def fig7_distance():
    g = pd.read_csv(RESULTS / "table4_distance_gradient.csv")
    fig, ax = plt.subplots(figsize=(W * 0.82, 2.9))
    periods = ["2000–2013", "2014–2021", "2022–2024"]
    xs = np.arange(3)
    for spec, col, off, mk in [("без санкционных индикаторов", C_COAL, -0.11, "o"),
                               ("с санкционными индикаторами", C_OTHER, 0.11, "s")]:
        s = g[(g.spec == spec) & g.period.isin(periods)].set_index("period").loc[periods]
        for x, (_, row) in zip(xs + off, s.iterrows()):
            ax.plot([x, x], [row.lo95, row.hi95], color=col, lw=1.5)
            ax.plot([x], [row.elasticity_russia], marker=mk, ms=5.5, color=col, mec=SURFACE, mew=1.0, zorder=3)
            ax.text(x + (-0.04 if off < 0 else 0.04), row.hi95 + 0.03, MINUS(f"{row.elasticity_russia:.2f}"), fontsize=7.6,
                    color=INK, ha="right" if off < 0 else "left", va="bottom")
    base = g[(g.period == "прочие пары (база)")].elasticity_russia.mean()
    ax.axhline(base, color=INK2, lw=0.9)
    ax.text(-0.46, base + 0.04, MINUS(f"прочие пары: {base:.2f}"), fontsize=7.6, color=INK2, ha="left", va="bottom")
    ax.set_xticks(xs); ax.set_xticklabels(periods); ax.set_xlim(-0.5, 2.5)
    ax.set_ylabel("Эластичность торговли по расстоянию")
    style_axes(ax); ax.set_ylim(-2.9, 0.1)
    handles = [Line2D([0], [0], marker="o", color=C_COAL, lw=1.5, ms=5, mec=SURFACE, label="Без санкционных индикаторов"),
               Line2D([0], [0], marker="s", color=C_OTHER, lw=1.5, ms=5, mec=SURFACE, label="С санкционными индикаторами")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06), fontsize=7.8)
    save(fig, "fig7_distance_gradient", g)


# ================================================================ Рис. A1 (данные)
def figA1_data_quality(d: pd.DataFrame):
    v = pd.read_csv(DATA / "vkr_panel_renamed.csv")
    w = v.pivot(index=["iso3_i", "iso3_j"], columns="year", values="trade")
    dl = np.log(w).diff(axis=1)
    idx = dl.index.to_frame(index=False)
    nr = ~((idx.iso3_i == "RUS") | (idx.iso3_j == "RUS")).values
    share = ((dl[nr].abs() > 1.0).mean() * 100).loc[2001:]
    geo = pd.read_csv(DATA / "gravity_panel.csv")[["iso3_i", "iso3_j", "dist_cap"]].drop_duplicates()
    vp = v[["iso3_i", "iso3_j", "dist_vkr"]].drop_duplicates()
    a = vp.assign(k=vp.apply(lambda r: tuple(sorted([r.iso3_i, r.iso3_j])), axis=1))
    b = geo.assign(k=geo.apply(lambda r: tuple(sorted([r.iso3_i, r.iso3_j])), axis=1))[["k", "dist_cap"]]
    m = a.merge(b, on="k")
    m["ratio"] = m.dist_vkr / m.dist_cap
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"wspace": 0.32})
    ax = axes[0]
    cols = [C_COAL if y == 2022 else "#c3c2b7" for y in share.index]
    ax.bar(share.index, share.values, width=0.7, color=cols, linewidth=0)
    ax.set_ylabel("% пар"); ax.set_xticks([2002, 2008, 2014, 2020, 2024]); style_axes(ax)
    ax.set_title("а) Пары без России с |Δ ln T| > 1", loc="left", color=INK, fontsize=8.5)
    ax.text(2022, share.loc[2022] + 0.4, f"{share.loc[2022]:.1f}%".replace(".", ","), ha="center", fontsize=8, color=INK)
    ax.set_ylim(0, 17.5)
    ax = axes[1]
    ax.scatter(m.dist_cap, m.dist_vkr, s=9, color=C_COAL, alpha=0.7, linewidths=0)
    lim = [50, 22000]
    ax.plot(lim, lim, color=INK2, lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Расстояние между столицами, CEPII, км"); ax.set_ylabel("Расстояние в файле ВКР, км")
    style_axes(ax, "both")
    ax.set_title("б) Расстояния ВКР и CEPII", loc="left", color=INK, fontsize=8.5)
    for _, r in m[(m.ratio > 1.6) | (m.ratio < 0.74)].iterrows():
        ax.annotate(f"{r.k[0]}–{r.k[1]}", (r.dist_cap, r.dist_vkr), fontsize=6.8, color=INK2, xytext=(5, -2),
                    textcoords="offset points")
    save(fig, "figA1_data_quality", m)


# ================================================================ Рис. A2 (перестановки)
def figA2_permutation():
    f = RESULTS / "permutation_draws.csv"
    s = RESULTS / "summary_inference.json"
    if not f.exists() or not s.exists():
        return
    summ = json.load(open(s, encoding="utf-8")).get("permutation")
    if not summ or summ.get("n_perm", 0) < 100:
        return
    arr = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"wspace": 0.28})
    for ax, col, obs, title in [(axes[0], "sanc14", summ["obs_sanc14"], "а) SANC14"), (axes[1], "sanc22", summ["obs_sanc22"], "б) SANC22")]:
        ax.hist(arr[col], bins=28, color="#c3c2b7", linewidth=0.6, edgecolor=SURFACE)
        ax.axvline(obs, color=C_COAL, lw=2.0)
        ax.set_xlabel("Коэффициент при случайном назначении «коалиции»"); ax.set_ylabel("Число назначений")
        style_axes(ax); ax.set_title(title, loc="left", color=INK, fontsize=8.5)
        ax.text(obs + 0.02, ax.get_ylim()[1] * 0.92, MINUS(f"факт: {obs:.2f}"), color=INK, fontsize=8, ha="left", va="top")
    save(fig, "figA2_permutation", arr)


def main():
    d = load_panel()
    fig1_descriptive(d)
    fig2_spec_path()
    fig3_event_study()
    fig4_subgroups()
    fig5_partner_shifts()
    fig6_counterfactual()
    fig7_distance()
    figA1_data_quality(d)
    figA2_permutation()


if __name__ == "__main__":
    main()
