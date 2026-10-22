"""Проверки статистического вывода для основной спецификации (PPML, ФЭ страна×год и пара, 2009–2024).

1) Перестановочный тест. «Обработанных» российских пар всего 14, поэтому
   кластерные ошибки могут быть ненадёжны. Метка «коалиция» случайно присваивается
   14 из 29 российских партнёров (кроме Украины и Респ. Кореи, имеющих собственные
   индикаторы); оценки по случайным назначениям формируют нулевое распределение.
2) Параметрический бутстрэп. Данные генерируются из подогнанной модели:
   y* = μ̂ · ε, ε — случайная выборка с возвращением из отношений y/μ̂ внутри
   20 страт по квантилям μ̂ (сохраняется зависимость дисперсии от размера потока).
   Сравниваются средние оценки с исходными (смещение от «несущественных
   параметров», Weidner & Zylkin, 2021), стандартные отклонения оценок с
   аналитическими ошибками и покрытие 95% доверительных интервалов.

Запуск:  python code/06_inference_checks.py [--perm 500] [--boot 200] [--seed 20260101]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

import ppml
from gravity_common import COALITION_2014, RESULTS
from gravity_models import build_blocks, fit_model, load_panel

POLICY = ["eu_ij", "wto_ij", "eaeu_ij", "nato_ij", "brics_ij"]
COVS = ["sanc14_ij", "sanc22_ij", "kor22_ij", "ukr14_ij", "ukr22_ij"] + POLICY


def permutation_test(d: pd.DataFrame, n_perm: int, rng: np.random.Generator) -> dict:
    d = d.reset_index(drop=True)
    base = fit_model(d, COVS, "cy+pair").table()
    obs14, obs22 = base.loc["sanc14_ij", "coef"], base.loc["sanc22_ij", "coef"]
    pool = sorted(set(d.loc[d.rus_pair == 1, "partner"]) - {"UKR", "KOR"})
    rus = d.rus_pair.values == 1
    draws = []
    t0 = time.time()
    for b in range(n_perm):
        pseudo = set(rng.choice(pool, size=len(COALITION_2014), replace=False))
        m = rus & d.partner.isin(pseudo).values
        dd = d.copy()
        dd["sanc14_ij"] = (m & (d.year.values >= 2014)).astype(int)
        dd["sanc22_ij"] = (m & (d.year.values >= 2022)).astype(int)
        t = fit_model(dd, COVS, "cy+pair").table()
        draws.append((t.loc["sanc14_ij", "coef"], t.loc["sanc22_ij", "coef"], len(pseudo & COALITION_2014)))
        if (b + 1) % 100 == 0:
            print(f"  permutation {b + 1}/{n_perm}  ({time.time() - t0:.0f} c)", flush=True)
    arr = pd.DataFrame(draws, columns=["sanc14", "sanc22", "overlap"])
    arr.to_csv(RESULTS / "permutation_draws.csv", index=False)
    p14 = (1 + (arr.sanc14.abs() >= abs(obs14)).sum()) / (1 + len(arr))
    p22 = (1 + (arr.sanc22.abs() >= abs(obs22)).sum()) / (1 + len(arr))
    # односторонние: доля случайных назначений с эффектом не слабее наблюдаемого (отрицательного)
    p14_1 = (1 + (arr.sanc14 <= obs14).sum()) / (1 + len(arr))
    p22_1 = (1 + (arr.sanc22 <= obs22).sum()) / (1 + len(arr))
    return {"n_perm": n_perm, "obs_sanc14": obs14, "obs_sanc22": obs22,
            "perm_sd_sanc14": float(arr.sanc14.std()), "perm_sd_sanc22": float(arr.sanc22.std()),
            "perm_p_two_sided_sanc14": float(p14), "perm_p_two_sided_sanc22": float(p22),
            "perm_p_one_sided_sanc14": float(p14_1), "perm_p_one_sided_sanc22": float(p22_1),
            "perm_q05_sanc14": float(arr.sanc14.quantile(0.05)), "perm_q95_sanc14": float(arr.sanc14.quantile(0.95)),
            "perm_q05_sanc22": float(arr.sanc22.quantile(0.05)), "perm_q95_sanc22": float(arr.sanc22.quantile(0.95))}


def parametric_bootstrap(d: pd.DataFrame, n_boot: int, rng: np.random.Generator) -> dict:
    d = d.reset_index(drop=True)
    covs = [c for c in COVS if d[c].std() > 0]
    blocks, nested = build_blocks(d, covs, "cy+pair")
    y = d.trade.values
    r0 = ppml.fit(y, blocks, cluster=d.pair_id.values, n_cov=len(covs), n_nested=nested)
    mu = r0.mu
    ratios = y / mu
    # страты по квантилям μ: дисперсия относительной ошибки зависит от размера потока
    bins = pd.qcut(pd.Series(mu), q=20, labels=False, duplicates="drop").values
    pools = {b: ratios[bins == b] for b in np.unique(bins)}
    t0 = r0.table()
    sims = []
    cover = {c: 0 for c in covs}
    for b in range(n_boot):
        eps = np.empty_like(ratios)
        for b_, pool in pools.items():
            idx = np.where(bins == b_)[0]
            eps[idx] = rng.choice(pool, size=len(idx), replace=True)
        ys = mu * eps
        r = ppml.fit(ys, blocks, cluster=d.pair_id.values, n_cov=len(covs), n_nested=nested)
        tt = r.table()
        sims.append(tt["coef"].values)
        for k, c in enumerate(covs):
            if abs(tt.loc[c, "coef"] - t0.loc[c, "coef"]) <= 1.96 * tt.loc[c, "se"]:
                cover[c] += 1
        if (b + 1) % 50 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}", flush=True)
    sims = np.array(sims)
    rows = []
    for k, c in enumerate(covs):
        rows.append(dict(variable=c, coef_hat=t0.loc[c, "coef"], se_analytic=t0.loc[c, "se"],
                         boot_mean=sims[:, k].mean(), bias=sims[:, k].mean() - t0.loc[c, "coef"],
                         bias_over_se=(sims[:, k].mean() - t0.loc[c, "coef"]) / t0.loc[c, "se"],
                         boot_sd=sims[:, k].std(ddof=1), coverage95=cover[c] / n_boot))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "bootstrap_bias_check.csv", index=False)
    return {"n_boot": n_boot, "resid_ratio_sd": float(np.std(ratios)),
            "max_abs_bias_over_se": float(out.bias_over_se.abs().max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=500)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260101)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    d = load_panel()
    d = d[d.year >= 2009]
    out = {}
    if args.perm > 0:
        print("перестановочный тест ...", flush=True)
        out["permutation"] = permutation_test(d, args.perm, rng)
        print(json.dumps(out["permutation"], indent=1, ensure_ascii=False))
    if args.boot > 0:
        print("параметрический бутстрэп ...", flush=True)
        out["bootstrap"] = parametric_bootstrap(d, args.boot, rng)
        print(pd.read_csv(RESULTS / "bootstrap_bias_check.csv").round(3).to_string(index=False))
    json.dump(out, open(RESULTS / "summary_inference.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
