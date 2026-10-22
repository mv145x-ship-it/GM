"""Загрузка свежих данных из открытых API (требуется доступ в интернет).

Скрипт НЕ изменяет имеющиеся наборы данных: он сохраняет «сырые» выгрузки в
``data/raw_update/`` и файл ``manifest.json`` (источник, параметры, время запроса).
Объединение с панелью выполняет ``code/10_merge_update.py``.

Подкоманды
----------
  probe     проверить доступность хостов (диагностика сети и прокси);
  wdi       ВВП, численность населения, ВВП на душу — World Bank WDI (api.worldbank.org);
  weo       ВВП, население, ВВП на душу — МВФ, WEO DataMapper (оценки на последние годы);
  comtrade  суммарная торговля reporter→partner (экспорт X, импорт M), UN Comtrade API v1
            (нужен ключ: переменная окружения COMTRADE_API_KEY; бесплатный ключ выдаётся
            на comtradeplus.un.org);
  all       wdi + weo + comtrade.

Примеры
-------
  python code/fetch_update.py probe
  python code/fetch_update.py all --start 2000 --end 2025
  COMTRADE_API_KEY=... python code/fetch_update.py comtrade --start 2021 --end 2025

Статус: код написан по документации API и проверен только на тестовых ответах
(``tests/test_update_pipeline.py``); в среде разработки доступ к перечисленным хостам
был закрыт, поэтому реальные запросы не выполнялись. Если API изменит формат ответа,
правки потребуются в функциях ``parse_*``.

Формат выходных файлов (data/raw_update/)
-----------------------------------------
  wdi.csv                  iso3, year, gdp_usd, pop, gdppc_usd
  weo.csv                  iso3, year, gdp_usd_bn, pop_m, gdppc_usd
  comtrade_directional.csv reporter, partner, year, flow (X|M), value_usd
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gravity_common import COUNTRIES, DATA

RAW = DATA / "raw_update"
CACHE = RAW / "cache"
UA = "gm-gravity-pipeline/1.0 (+https://github.com/mv145x-ship-it/GM)"

WDI_INDICATORS = {"NY.GDP.MKTP.CD": "gdp_usd", "SP.POP.TOTL": "pop", "NY.GDP.PCAP.CD": "gdppc_usd"}
WEO_INDICATORS = {"NGDPD": "gdp_usd_bn", "LP": "pop_m", "NGDPDPC": "gdppc_usd"}
COMTRADE_URL = "https://comtradeapi.un.org/data/v1/get/C/A/HS"
COMTRADE_REPORTERS = "https://comtradeapi.un.org/files/v1/app/reference/Reporters.json"

PROBES = {
    "World Bank API": "https://api.worldbank.org/v2/country/RUS/indicator/SP.POP.TOTL?format=json&date=2020&per_page=1",
    "IMF DataMapper": "https://www.imf.org/external/datamapper/api/v1/NGDPD/RUS",
    "UN Comtrade (справочник)": COMTRADE_REPORTERS,
}


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------
def http_json(url: str, headers: dict | None = None, retries: int = 4, timeout: int = 90):
    """GET → JSON с повторами при временных ошибках (429/5xx, сетевые сбои)."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json", **(headers or {})})
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and k < retries - 1:
                time.sleep(2 ** (k + 2))
                continue
            raise
        except urllib.error.URLError:
            if k < retries - 1:
                time.sleep(2 ** (k + 1))
                continue
            raise


def write_manifest(source: str, params: dict, n_rows: int) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / "manifest.json"
    man = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    man[source] = {"fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "params": params,
                   "n_rows": int(n_rows)}
    path.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------
# Разбор ответов (отдельные функции — чтобы тестировать без сети)
# ----------------------------------------------------------------------------
def parse_wdi(payload: list, name: str) -> pd.DataFrame:
    """Ответ World Bank API: [meta, [записи]] → iso3, year, <name>."""
    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise RuntimeError(f"WDI: неожиданный ответ для {name}: {str(payload)[:300]}")
    rows = [(r["countryiso3code"], int(r["date"]), float(r["value"]))
            for r in payload[1] if r.get("value") is not None and r.get("countryiso3code")]
    return pd.DataFrame(rows, columns=["iso3", "year", name])


def parse_weo(payload: dict, code: str, name: str) -> pd.DataFrame:
    """Ответ IMF DataMapper: {"values": {code: {ISO3: {year: value}}}} → iso3, year, <name>."""
    try:
        vals = payload["values"][code]
    except (KeyError, TypeError) as e:
        raise RuntimeError(f"WEO: неожиданный ответ для {code}: {str(payload)[:300]}") from e
    rows = [(iso, int(y), float(v)) for iso, ser in vals.items() for y, v in ser.items() if v is not None]
    return pd.DataFrame(rows, columns=["iso3", "year", name])


def parse_comtrade_reporters(payload) -> dict[str, int]:
    """Справочник Reporters.json → {ISO3: reporterCode}. Формат: {"results": [{...}]} или список."""
    items = payload.get("results", payload) if isinstance(payload, dict) else payload
    out = {}
    for it in items:
        iso = it.get("reporterCodeIsoAlpha3") or it.get("ISO3")
        code = it.get("reporterCode", it.get("id"))
        if iso and code is not None and str(code).lstrip("-").isdigit():
            out[iso] = int(code)
    return out


def parse_comtrade(payload: dict) -> pd.DataFrame:
    """Ответ Comtrade v1: {"data": [{reporterISO, partnerISO, period, flowCode, primaryValue, ...}]}."""
    if not isinstance(payload, dict) or "data" not in payload:
        raise RuntimeError(f"Comtrade: неожиданный ответ: {str(payload)[:300]}")
    if payload.get("error"):
        raise RuntimeError(f"Comtrade: {payload['error']}")
    rows = []
    for r in payload["data"] or []:
        val = r.get("primaryValue")
        if val is None:
            continue
        rows.append((r.get("reporterISO"), r.get("partnerISO"), int(r.get("period") or r.get("refYear")),
                     r.get("flowCode"), float(val)))
    return pd.DataFrame(rows, columns=["reporter", "partner", "year", "flow", "value_usd"])


# ----------------------------------------------------------------------------
# Загрузчики
# ----------------------------------------------------------------------------
def fetch_wdi(start: int, end: int) -> pd.DataFrame:
    iso = ";".join(sorted(COUNTRIES))
    out = None
    for code, name in WDI_INDICATORS.items():
        url = (f"https://api.worldbank.org/v2/country/{iso}/indicator/{code}"
               f"?format=json&per_page=20000&date={start}:{end}")
        df = parse_wdi(http_json(url), name)
        out = df if out is None else out.merge(df, on=["iso3", "year"], how="outer")
    out = out.sort_values(["iso3", "year"])
    RAW.mkdir(parents=True, exist_ok=True)
    out.to_csv(RAW / "wdi.csv", index=False)
    write_manifest("wdi", {"start": start, "end": end, "indicators": WDI_INDICATORS}, len(out))
    return out


def fetch_weo(start: int, end: int) -> pd.DataFrame:
    iso = "/".join(sorted(COUNTRIES))
    out = None
    for code, name in WEO_INDICATORS.items():
        df = parse_weo(http_json(f"https://www.imf.org/external/datamapper/api/v1/{code}/{iso}"), code, name)
        out = df if out is None else out.merge(df, on=["iso3", "year"], how="outer")
    out = out[(out.year >= start) & (out.year <= end) & out.iso3.isin(COUNTRIES)].sort_values(["iso3", "year"])
    RAW.mkdir(parents=True, exist_ok=True)
    out.to_csv(RAW / "weo.csv", index=False)
    write_manifest("weo", {"start": start, "end": end, "indicators": WEO_INDICATORS}, len(out))
    return out


def fetch_comtrade(start: int, end: int, key: str, chunk: int = 5) -> pd.DataFrame:
    """Суммарная торговля (cmdCode=TOTAL) каждой страны выборки с остальными: потоки X и M.

    Запросы кэшируются в data/raw_update/cache/, поэтому прерванную загрузку можно продолжить.
    Число запросов: 32 страны × ceil(число лет / chunk); бесплатный тариф — 500 запросов в сутки.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    headers = {"Ocp-Apim-Subscription-Key": key}
    codes_all = parse_comtrade_reporters(http_json(COMTRADE_REPORTERS))
    missing = [c for c in COUNTRIES if c not in codes_all]
    if missing:
        raise RuntimeError(f"нет кодов Comtrade для: {missing}")
    codes = {c: codes_all[c] for c in COUNTRIES}
    frames = []
    years = list(range(start, end + 1))
    for rep in sorted(COUNTRIES):
        partners = ",".join(str(codes[p]) for p in sorted(COUNTRIES) if p != rep)
        for k in range(0, len(years), chunk):
            per = years[k:k + chunk]
            cache = CACHE / f"comtrade_{rep}_{per[0]}_{per[-1]}.json"
            if cache.exists():
                payload = json.loads(cache.read_text(encoding="utf-8"))
            else:
                q = dict(reporterCode=codes[rep], partnerCode=partners, period=",".join(map(str, per)), cmdCode="TOTAL",
                         flowCode="X,M", partner2Code=0, customsCode="C00", motCode=0, maxRecords=100000,
                         includeDesc="false")
                payload = http_json(COMTRADE_URL + "?" + urllib.parse.urlencode(q), headers=headers)
                cache.write_text(json.dumps(payload), encoding="utf-8")
                time.sleep(1.0)   # щадящий режим для бесплатного тарифа
            df = parse_comtrade(payload)
            print(f"  {rep} {per[0]}–{per[-1]}: {len(df)} записей", flush=True)
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out = out[out.reporter.isin(COUNTRIES) & out.partner.isin(COUNTRIES) & out.flow.isin(["X", "M"])]
    out = out.groupby(["reporter", "partner", "year", "flow"], as_index=False).value_usd.sum()
    out.to_csv(RAW / "comtrade_directional.csv", index=False)
    write_manifest("comtrade", {"start": start, "end": end, "cmdCode": "TOTAL", "flows": "X,M"}, len(out))
    return out


def probe() -> int:
    bad = 0
    for name, url in PROBES.items():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                print(f"OK    {name:28s} HTTP {r.status}")
        except Exception as e:  # noqa: BLE001 — диагностика: показываем любую причину
            bad += 1
            print(f"FAIL  {name:28s} {type(e).__name__}: {e}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["probe", "wdi", "weo", "comtrade", "all"])
    ap.add_argument("--start", type=int, default=2000)
    ap.add_argument("--end", type=int, default=datetime.now().year - 1,
                    help="последний год (по умолчанию — прошлый календарный год)")
    args = ap.parse_args()
    if args.cmd == "probe":
        sys.exit(1 if probe() else 0)
    if args.cmd in ("wdi", "all"):
        print("WDI …")
        print(len(fetch_wdi(args.start, args.end)), "строк")
    if args.cmd in ("weo", "all"):
        print("WEO …")
        print(len(fetch_weo(args.start, args.end)), "строк")
    if args.cmd in ("comtrade", "all"):
        key = os.environ.get("COMTRADE_API_KEY")
        if not key:
            sys.exit("Нужна переменная окружения COMTRADE_API_KEY (ключ UN Comtrade).")
        print("Comtrade …")
        print(len(fetch_comtrade(args.start, args.end, key)), "строк")


if __name__ == "__main__":
    main()
