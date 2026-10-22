"""Объединение свежих выгрузок (data/raw_update/) с гармонизированной панелью.

Вход
----
  * ``data/gravity_panel.csv`` и ``data/macro_country_year.csv`` — текущие наборы;
  * ``data/raw_update/{wdi,weo,comtrade_directional}.csv`` — результат ``code/fetch_update.py``.

Выход
-----
  * ``data/gravity_panel_updated.csv`` — обновлённая панель (та же схема столбцов + ``mirror_ratio``);
  * ``data/gravity_panel_updated_gretl.csv`` — числовая версия для gretl;
  * ``data/update_report.csv`` — сводка: что добавлено, что заменено, где данных нет.

Правила
-------
1. Торговля пары i–j в году t = среднее двух зеркальных сообщений: «экспорт + импорт», сообщённые
   страной i о торговле с j, и «экспорт + импорт», сообщённые страной j о торговле с i
   (если сообщение одно — берётся оно). ``mirror_ratio`` — отношение большего сообщения к меньшему
   (значения > 3 указывают на ошибки отчётности и требуют проверки).
2. Политика замены ``--replace``:
     none     добавляются только годы после последнего года панели;
     flagged  + заменяются «протянутые» значения (flag_carry = 1), если есть данные Comtrade (по умолчанию);
     all      все значения заменяются данными Comtrade (единый источник; прежнее значение —
              запасной вариант, если новых данных нет).
3. Макроданные: значения за годы, уже имеющиеся в панели, сохраняются, кроме оценок
   «IMF-WEO x WDI-pop» (заменяются данными WDI, когда они появились); для новых лет — WDI,
   при отсутствии — WEO. ``--refresh-macro`` заменяет все годы последним выпуском WDI.
4. Индикаторы членства и санкций пересчитываются функцией ``add_membership_and_sanctions``
   из ``01_build_dataset.py`` (единые правила кодировки; хронология — ``gravity_common.py``).

После обновления модели и рукопись пересчитываются так:
    GM_PANEL=data/gravity_panel_updated.csv python code/04_main_models.py  (и далее 05–09)
Интерпретационные места рукописи (годы «2022–2024», сравнения с прежними данными) проверяются вручную.

Запуск:  python code/10_merge_update.py [--end-year 2025] [--replace flagged] [--refresh-macro]
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from gravity_common import COUNTRIES, DATA

b01 = importlib.import_module("01_build_dataset")   # модуль с числовым префиксом: обычным import не загрузить

STATIC_COLS = ["pair_id", "iso3_i", "iso3_j", "country_i", "country_j", "dist", "dist_cap", "contig", "comlang_off",
               "comlang_ethno", "colony", "comcol", "col45", "smctry"]
SRC_BOTH = "Comtrade (среднее двух зеркальных сообщений)"
SRC_I = "Comtrade (сообщение страны i)"
SRC_J = "Comtrade (сообщение страны j)"


def mirror_totals(cd: pd.DataFrame) -> pd.DataFrame:
    """Направленные потоки → суммарная торговля неупорядоченных пар (млн долл.).

    cd: reporter, partner, year, flow ('X'|'M'), value_usd.
    Возвращает: iso3_i < iso3_j, year, trade_new, n_reports, mirror_ratio, src_new.
    """
    w = cd.pivot_table(index=["reporter", "partner", "year"], columns="flow", values="value_usd",
                       aggfunc="sum").reset_index()
    for f in ("X", "M"):
        if f not in w:
            w[f] = np.nan
    w["rep_total"] = w[["X", "M"]].sum(axis=1, min_count=1) / 1e6
    swap = w.reporter > w.partner
    w["iso3_i"] = np.where(swap, w.partner, w.reporter)
    w["iso3_j"] = np.where(swap, w.reporter, w.partner)
    w["side"] = np.where(swap, "j", "i")
    p = w.pivot_table(index=["iso3_i", "iso3_j", "year"], columns="side", values="rep_total",
                      aggfunc="first").reset_index()
    for c in ("i", "j"):
        if c not in p:
            p[c] = np.nan
    both = p[["i", "j"]].notna().all(axis=1)
    p["n_reports"] = p[["i", "j"]].notna().sum(axis=1)
    p["trade_new"] = p[["i", "j"]].mean(axis=1)
    lo = p[["i", "j"]].min(axis=1).where(lambda s: s > 0)
    p["mirror_ratio"] = np.where(both, p[["i", "j"]].max(axis=1) / lo, np.nan)
    p["src_new"] = np.select([both, p.i.notna(), p.j.notna()], [SRC_BOTH, SRC_I, SRC_J], "")
    return p[["iso3_i", "iso3_j", "year", "trade_new", "n_reports", "mirror_ratio", "src_new"]]


def macro_updated(old: pd.DataFrame, wdi: pd.DataFrame | None, weo: pd.DataFrame | None, end: int,
                  refresh_all: bool = False) -> pd.DataFrame:
    """Страна × год: gdp (млрд долл.), pop_m (млн), gdppc (долл.), gdp_src."""
    idx = pd.MultiIndex.from_product([sorted(COUNTRIES), range(2000, end + 1)], names=["iso3", "year"])
    m = pd.DataFrame(index=idx).reset_index().merge(old, on=["iso3", "year"], how="left")
    m["gdp_src"] = m["gdp_src"].fillna("")
    old_gdp, old_pop, old_src = m["gdp"].copy(), m["pop_m"].copy(), m["gdp_src"].copy()
    if wdi is not None and len(wdi):
        w = wdi.rename(columns={"gdp_usd": "w_gdp", "pop": "w_pop"})[["iso3", "year", "w_gdp", "w_pop"]]
        m = m.merge(w, on=["iso3", "year"], how="left")
    else:
        m["w_gdp"] = np.nan
        m["w_pop"] = np.nan
    if weo is not None and len(weo):
        e = weo.rename(columns={"gdp_usd_bn": "e_gdp", "pop_m": "e_pop"})[["iso3", "year", "e_gdp", "e_pop"]]
        m = m.merge(e, on=["iso3", "year"], how="left")
    else:
        m["e_gdp"] = np.nan
        m["e_pop"] = np.nan
    w_gdp = m["w_gdp"] / 1e9
    keep_old = old_gdp.notna() & (old_src == "WDI") & (not refresh_all)
    use_wdi = ~keep_old & w_gdp.notna()
    keep_other = ~keep_old & ~use_wdi & old_gdp.notna()        # напр. «IMF-WEO x WDI-pop», если WDI ещё нет
    use_weo = ~keep_old & ~use_wdi & ~keep_other & m["e_gdp"].notna()
    m["gdp"] = np.select([keep_old, use_wdi, keep_other, use_weo], [old_gdp, w_gdp, old_gdp, m["e_gdp"]], np.nan)
    m["gdp_src"] = np.select([keep_old, use_wdi, keep_other, use_weo], [old_src, "WDI", old_src, "IMF-WEO"], "")
    pop_new = (m["w_pop"] / 1e6).where(m["w_pop"].notna(), m["e_pop"])
    m["pop_m"] = np.where(old_pop.notna() & (not refresh_all), old_pop, pop_new)
    m["gdppc"] = m["gdp"] * 1e9 / (m["pop_m"] * 1e6)
    return m[["iso3", "year", "gdp", "pop_m", "gdppc", "gdp_src"]]


def update_panel(base: pd.DataFrame, macro_old: pd.DataFrame, cd: pd.DataFrame, wdi: pd.DataFrame | None,
                 weo: pd.DataFrame | None, end: int, replace: str = "flagged",
                 refresh_macro: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Вернуть (обновлённая панель, сводка)."""
    assert replace in ("none", "flagged", "all")
    macro = macro_updated(macro_old, wdi, weo, end, refresh_macro)
    static = base.drop_duplicates("pair_id")[STATIC_COLS]
    grid = static.merge(pd.DataFrame({"year": list(range(int(base.year.min()), end + 1))}), how="cross")
    old = base[["pair_id", "year", "trade", "trade_src", "flag_carry"]].rename(
        columns={"trade": "trade_old", "trade_src": "src_old", "flag_carry": "carry_old"})
    grid = grid.merge(old, on=["pair_id", "year"], how="left").merge(mirror_totals(cd), on=["iso3_i", "iso3_j", "year"],
                                                                   how="left")
    has_old, has_new = grid.trade_old.notna(), grid.trade_new.notna()
    use_new = {"none": ~has_old & has_new,
               "flagged": (~has_old & has_new) | (has_old & has_new & (grid.carry_old == 1)),
               "all": has_new}[replace]
    grid["trade"] = np.where(use_new, grid.trade_new, grid.trade_old)
    grid["trade_src"] = np.where(use_new, grid.src_new, grid.src_old)
    n_missing = int(grid.trade.isna().sum())
    missing_years = sorted(grid.loc[grid.trade.isna(), "year"].unique().tolist())
    grid = grid[grid.trade.notna()].copy()

    for side in ("i", "j"):
        mm = macro.rename(columns={"iso3": f"iso3_{side}", "gdp": f"gdp_{side}", "pop_m": f"pop_{side}",
                                   "gdppc": f"gdppc_{side}", "gdp_src": f"gdp_src_{side}"})
        grid = grid.merge(mm, on=[f"iso3_{side}", "year"], how="left")
    grid["gdp_src"] = grid["gdp_src_i"]
    grid = b01.add_membership_and_sanctions(grid)
    grid = grid.sort_values(["pair_id", "year"]).reset_index(drop=True)
    prev = grid.groupby("pair_id")["trade"].shift(1)
    grid["flag_carry"] = (grid["trade"] == prev).astype(int)
    cols = list(base.columns) + ["mirror_ratio"]
    out = grid[cols]

    last_old = int(base.year.max())
    no_macro = int(out[["gdp_i", "gdp_j", "pop_i", "pop_j"]].isna().any(axis=1).sum())
    rep = [
        ("строк в обновлённой панели", len(out)),
        ("годы", f"{int(out.year.min())}–{int(out.year.max())}"),
        ("новых наблюдений (годы после последнего года прежней панели)", int((use_new & ~has_old).sum())),
        (f"заменено прежних значений (политика {replace})", int((use_new & has_old).sum())),
        ("пары-годы без торговли (исключены); годы", f"{n_missing}; {missing_years}"),
        ("строк без макроданных (ВВП/население)", no_macro),
        ("пар-годов с mirror_ratio > 3", int((out.mirror_ratio > 3).sum())),
        ("значений торговли, протянутых с прошлого года (flag_carry)", int(out.flag_carry.sum())),
        (f"из них в годах > {last_old}", int(out.loc[out.year > last_old, "flag_carry"].sum())),
    ]
    return out, pd.DataFrame(rep, columns=["показатель", "значение"])


def _read_optional(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", default=str(DATA / "raw_update"))
    ap.add_argument("--base", default=str(DATA / "gravity_panel.csv"))
    ap.add_argument("--macro-base", default=str(DATA / "macro_country_year.csv"))
    ap.add_argument("--end-year", type=int, default=None, help="по умолчанию — последний год в данных Comtrade")
    ap.add_argument("--replace", choices=["none", "flagged", "all"], default="flagged")
    ap.add_argument("--refresh-macro", action="store_true")
    ap.add_argument("--out", default=str(DATA / "gravity_panel_updated.csv"))
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    ct = raw / "comtrade_directional.csv"
    if not ct.exists():
        raise SystemExit(f"Нет {ct}. Сначала выполните: python code/fetch_update.py comtrade --start 2000 --end <год>")
    cd = pd.read_csv(ct)
    base = pd.read_csv(args.base)
    macro_old = pd.read_csv(args.macro_base)
    end = args.end_year or int(cd.year.max())
    out, rep = update_panel(base, macro_old, cd, _read_optional(raw / "wdi.csv"), _read_optional(raw / "weo.csv"), end,
                            args.replace, args.refresh_macro)
    out_path = Path(args.out)
    out.to_csv(out_path, index=False)
    b01.export_gretl(out.drop(columns=["mirror_ratio"]), suffix="_updated")
    rep.to_csv(out_path.with_name("update_report.csv"), index=False)
    print(rep.to_string(index=False))
    print("->", out_path)


if __name__ == "__main__":
    main()
