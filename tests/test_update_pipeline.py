"""Тесты конвейера обновления данных (code/fetch_update.py, code/10_merge_update.py).

Реальные запросы к API здесь НЕ выполняются: проверяются (а) разбор ответов на синтетических
примерах в документированном формате и (б) логика объединения на данных, построенных из самой
панели (зеркальные сообщения «режутся» на экспорт и импорт так, чтобы их сумма совпадала с
исходным значением). Синтетические значения существуют только внутри тестов и никуда не
сохраняются.

Запуск:  python tests/test_update_pipeline.py     (или pytest tests/)
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

fetch = importlib.import_module("fetch_update")
merge = importlib.import_module("10_merge_update")


# ----------------------------------------------------------------------------
# Разбор ответов API (формат — по документации)
# ----------------------------------------------------------------------------
def test_parse_wdi():
    payload = [{"page": 1, "pages": 1, "per_page": 20000, "total": 3},
               [{"indicator": {"id": "SP.POP.TOTL"}, "country": {"id": "RU"}, "countryiso3code": "RUS", "date": "2023", "value": 146e6},
                {"indicator": {"id": "SP.POP.TOTL"}, "country": {"id": "RU"}, "countryiso3code": "RUS", "date": "2024", "value": None},
                {"indicator": {"id": "SP.POP.TOTL"}, "country": {"id": "1W"}, "countryiso3code": "", "date": "2023", "value": 1.0}]]
    df = fetch.parse_wdi(payload, "pop")
    assert df.to_dict("records") == [{"iso3": "RUS", "year": 2023, "pop": 146e6}]
    try:
        fetch.parse_wdi([{"message": [{"id": "120", "key": "Invalid value"}]}], "pop")
    except RuntimeError:
        pass
    else:
        raise AssertionError("ожидалось исключение для ответа с ошибкой")


def test_parse_weo():
    payload = {"values": {"NGDPD": {"RUS": {"2024": 2161.0, "2025": 2076.5}, "CHN": {"2024": 18700.0, "2025": None}}}}
    df = fetch.parse_weo(payload, "NGDPD", "gdp_usd_bn").sort_values(["iso3", "year"])
    assert df[["iso3", "year"]].values.tolist() == [["CHN", 2024], ["RUS", 2024], ["RUS", 2025]]


def test_parse_comtrade():
    payload = {"count": 3, "error": "", "data": [
        {"reporterISO": "RUS", "partnerISO": "CHN", "period": "2023", "flowCode": "X", "primaryValue": 1.0e11},
        {"reporterISO": "RUS", "partnerISO": "CHN", "period": "2023", "flowCode": "M", "primaryValue": 0.9e11},
        {"reporterISO": "RUS", "partnerISO": "IND", "period": "2023", "flowCode": "X", "primaryValue": None}]}
    df = fetch.parse_comtrade(payload)
    assert len(df) == 2 and set(df.flow) == {"X", "M"}
    try:
        fetch.parse_comtrade({"data": [], "error": "Invalid key"})
    except RuntimeError:
        pass
    else:
        raise AssertionError("ожидалось исключение для ответа с ошибкой")


def test_parse_reporters():
    payload = {"results": [{"id": "643", "reporterCode": 643, "reporterCodeIsoAlpha3": "RUS"},
                           {"id": "all", "reporterCode": "all", "reporterCodeIsoAlpha3": None},
                           {"id": "699", "reporterCode": 699, "reporterCodeIsoAlpha3": "IND"}]}
    assert fetch.parse_comtrade_reporters(payload) == {"RUS": 643, "IND": 699}


# ----------------------------------------------------------------------------
# Зеркальная сверка
# ----------------------------------------------------------------------------
def test_mirror_totals():
    cd = pd.DataFrame([
        # пара CHN–RUS 2023: Китай сообщает 100 + 60, Россия — 90 + 50 (млн долл. -> в долларах)
        ("CHN", "RUS", 2023, "X", 100e6), ("CHN", "RUS", 2023, "M", 60e6),
        ("RUS", "CHN", 2023, "X", 90e6), ("RUS", "CHN", 2023, "M", 50e6),
        # пара IND–RUS 2023: только сообщение России
        ("RUS", "IND", 2023, "X", 10e6), ("RUS", "IND", 2023, "M", 30e6),
    ], columns=["reporter", "partner", "year", "flow", "value_usd"])
    t = merge.mirror_totals(cd).set_index(["iso3_i", "iso3_j", "year"])
    r = t.loc[("CHN", "RUS", 2023)]
    assert abs(r.trade_new - 150.0) < 1e-9 and r.n_reports == 2 and abs(r.mirror_ratio - 160 / 140) < 1e-9
    assert r.src_new == merge.SRC_BOTH
    r = t.loc[("IND", "RUS", 2023)]
    assert abs(r.trade_new - 40.0) < 1e-9 and r.n_reports == 1 and np.isnan(r.mirror_ratio) and r.src_new == merge.SRC_J


# ----------------------------------------------------------------------------
# Макроданные
# ----------------------------------------------------------------------------
def test_macro_updated():
    old = pd.DataFrame({"iso3": ["RUS", "RUS", "CHN"], "year": [2023, 2024, 2023], "gdp": [2020.0, 2161.0, 17800.0],
                        "pop_m": [146.0, 146.2, 1410.0], "gdppc": [13800.0, 14780.0, 12600.0],
                        "gdp_src": ["WDI", "IMF-WEO x WDI-pop", "WDI"]})
    wdi = pd.DataFrame({"iso3": ["RUS", "RUS"], "year": [2023, 2024], "gdp_usd": [9999e9, 2200e9], "pop": [1e6, 146.1e6]})
    weo = pd.DataFrame({"iso3": ["RUS", "CHN"], "year": [2025, 2025], "gdp_usd_bn": [2100.0, 19000.0], "pop_m": [146.0, 1409.0]})
    m = merge.macro_updated(old, wdi, weo, 2025).set_index(["iso3", "year"])
    assert m.loc[("RUS", 2023), "gdp"] == 2020.0 and m.loc[("RUS", 2023), "gdp_src"] == "WDI"          # прежнее WDI не трогаем
    assert m.loc[("RUS", 2024), "gdp"] == 2200.0 and m.loc[("RUS", 2024), "gdp_src"] == "WDI"          # оценка заменена фактом
    assert m.loc[("RUS", 2025), "gdp"] == 2100.0 and m.loc[("RUS", 2025), "gdp_src"] == "IMF-WEO"      # новый год — WEO
    assert abs(m.loc[("CHN", 2025), "gdppc"] - 19000e9 / 1409e6) < 1e-6
    m2 = merge.macro_updated(old, wdi, weo, 2025, refresh_all=True).set_index(["iso3", "year"])
    assert m2.loc[("RUS", 2023), "gdp"] == 9999.0                                                       # --refresh-macro


# ----------------------------------------------------------------------------
# Регрессионный тест объединения на данных панели
# ----------------------------------------------------------------------------
def _fixture_from_panel(base: pd.DataFrame, new_year: int | None = None, growth: float = 1.05) -> pd.DataFrame:
    """Направленные «сообщения Comtrade», сумма которых по каждой стране равна значению панели."""
    rows = []
    b = base[["iso3_i", "iso3_j", "year", "trade"]].copy()
    if new_year is not None:
        last = b[b.year == b.year.max()].copy()
        last["year"] = new_year
        last["trade"] = last["trade"] * growth
        b = pd.concat([b, last], ignore_index=True)
    for r in b.itertuples():
        t = r.trade * 1e6
        rows += [(r.iso3_i, r.iso3_j, r.year, "X", 0.6 * t), (r.iso3_i, r.iso3_j, r.year, "M", 0.4 * t),
                 (r.iso3_j, r.iso3_i, r.year, "X", 0.4 * t), (r.iso3_j, r.iso3_i, r.year, "M", 0.6 * t)]
    return pd.DataFrame(rows, columns=["reporter", "partner", "year", "flow", "value_usd"])


def test_update_panel_regression():
    base = pd.read_csv(ROOT / "data" / "gravity_panel.csv")
    macro_old = pd.read_csv(ROOT / "data" / "macro_country_year.csv")
    cd = _fixture_from_panel(base, new_year=2025)
    wdi = None
    weo = pd.DataFrame({"iso3": sorted(merge.COUNTRIES), "year": 2025, "gdp_usd_bn": 1000.0, "pop_m": 50.0})

    # 1) политика none: прежние годы не меняются, добавляется 2025
    out, rep = merge.update_panel(base, macro_old, cd, wdi, weo, 2025, replace="none")
    old_part = out[out.year <= 2024].reset_index(drop=True)
    cols = [c for c in base.columns if c != "trade_src"]
    # пустые строки в CSV читаются как NaN — приводим к одному виду
    pd.testing.assert_frame_equal(old_part[cols].fillna(""), base[cols].reset_index(drop=True).fillna(""),
                                  check_dtype=False, rtol=1e-9)
    assert (out.year == 2025).sum() == base.pair_id.nunique()
    assert np.allclose(out[out.year == 2025].sort_values("pair_id").trade.values,
                       base[base.year == 2024].sort_values("pair_id").trade.values * 1.05)
    assert out[out.year == 2025].gdp_src.eq("IMF-WEO").all()
    # индикатор БРИКС для пар с Индонезией включается в 2025 г.
    idn = out[(out.year == 2025) & ((out.iso3_i == "IDN") | (out.iso3_j == "IDN"))]
    assert idn.brics_ij.sum() == 7      # IDN–{BRA, CHN, IND, RUS, ZAF, EGY, ARE}: все члены БРИКС в 2025 г.

    # 2) политика all: значения совпадают с прежними (фикстура построена из них), источник — Comtrade
    out_all, _ = merge.update_panel(base, macro_old, cd, wdi, weo, 2025, replace="all")
    o = out_all[out_all.year <= 2024].sort_values(["pair_id", "year"]).reset_index(drop=True)
    assert np.allclose(o.trade.values, base.sort_values(["pair_id", "year"]).trade.values, rtol=1e-9)
    assert o.trade_src.eq(merge.SRC_BOTH).all()

    # 3) политика flagged: заменяются только протянутые значения; здесь они совпадают с прежними, поэтому
    #    число замен равно числу протянутых наблюдений
    out_fl, rep_fl = merge.update_panel(base, macro_old, cd, wdi, weo, 2025, replace="flagged")
    n_replaced = int(rep_fl.loc[rep_fl["показатель"].str.startswith("заменено"), "значение"].iloc[0])
    assert n_replaced == int(base.flag_carry.sum()), (n_replaced, int(base.flag_carry.sum()))
    # после замены на зеркальные данные протянутых значений не остаётся ни в 2024, ни ранее
    # (в фикстуре зеркальное значение равно прежнему, поэтому флаг сохраняется — проверяем лишь согласованность схемы)
    assert list(out_fl.columns) == list(base.columns) + ["mirror_ratio"]


def test_update_panel_missing_new_year_is_dropped():
    base = pd.read_csv(ROOT / "data" / "gravity_panel.csv")
    macro_old = pd.read_csv(ROOT / "data" / "macro_country_year.csv")
    cd = _fixture_from_panel(base, new_year=2025)
    # уберём все сообщения, где участвует Беларусь, за 2025 г.
    cd = cd[~((cd.year == 2025) & ((cd.reporter == "BLR") | (cd.partner == "BLR")))]
    weo = pd.DataFrame({"iso3": sorted(merge.COUNTRIES), "year": 2025, "gdp_usd_bn": 1000.0, "pop_m": 50.0})
    out, rep = merge.update_panel(base, macro_old, cd, None, weo, 2025, replace="none")
    assert (out.year == 2025).sum() == base.pair_id.nunique() - 31
    miss = rep.loc[rep["показатель"].str.startswith("пары-годы без торговли"), "значение"].iloc[0]
    assert miss.startswith("31;")


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name)
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print("FAIL", name, type(e).__name__, e)
            traceback.print_exc()
    sys.exit(1 if failed else 0)
