"""Оценивание гравитационных моделей: PPML и МНК с высокоразмерными фиксированными эффектами.

Модуль реализует (без внешних зависимостей, кроме numpy/scipy/pandas):

  * PPML (Santos Silva & Tenreyro, 2006) методом Ньютона–Рафсона с поиском по
    лучу; фиксированные эффекты (ФЭ) задаются разреженными блоками индикаторов,
    что позволяет работать с «мультичленством» — единым эффектом «страна×год» для
    обеих стран неориентированной пары (суммарная торговля T_ij = X_ij + X_ji);
  * кластерно-робастные стандартные ошибки (кластер = пара стран) и
    «диадно-робастные» ошибки Aronow–Samii–Assenova (2015), учитывающие
    корреляцию ошибок между парами, имеющими общую страну;
  * МНК в логарифмах с теми же блоками ФЭ;
  * автоматическое исключение регрессоров, поглощённых ФЭ (коллинеарность);
  * оценивание на подвыборке (``sample``) с прогнозом на все наблюдения.

Проверка: 03_validate_estimator.py — совпадение с pyfixest.fepois/feols.

Обозначения: y — зависимая переменная (T_ijt), blocks = [(матрица, имена), ...];
первые n_cov столбцов первого блока — интересующие коэффициенты.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.linalg as sla
import scipy.sparse as sp


# ----------------------------------------------------------------------------
# Построители блоков фиксированных эффектов
# ----------------------------------------------------------------------------
def fe_pair(pair_id: Sequence) -> tuple[sp.csr_matrix, list[str]]:
    """Индикаторы пар (парные фиксированные эффекты η_ij)."""
    codes, uniq = pd.factorize(np.asarray(pair_id))
    n = len(codes)
    M = sp.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, len(uniq)))
    return M, [f"pair_{u}" for u in uniq]


def fe_year(year: Sequence, drop_first: bool = False) -> tuple[sp.csr_matrix, list[str]]:
    """Индикаторы лет; drop_first=True — если в модели есть константа/иные нормировки."""
    codes, uniq = pd.factorize(np.asarray(year), sort=True)
    n = len(codes)
    M = sp.csr_matrix((np.ones(n), (np.arange(n), codes)), shape=(n, len(uniq)))
    names = [f"year_{u}" for u in uniq]
    if drop_first:
        M, names = M[:, 1:], names[1:]
    return M, names


def fe_country(iso_i: Sequence, iso_j: Sequence, drop_first: bool = False) -> tuple[sp.csr_matrix, list[str]]:
    """Страновые эффекты с «мультичленством»: единица для каждой из двух стран пары."""
    iso_i, iso_j = np.asarray(iso_i), np.asarray(iso_j)
    uniq = sorted(set(iso_i) | set(iso_j))
    ix = {u: k for k, u in enumerate(uniq)}
    n = len(iso_i)
    rows = np.r_[np.arange(n), np.arange(n)]
    cols = np.r_[[ix[a] for a in iso_i], [ix[b] for b in iso_j]]
    M = sp.csr_matrix((np.ones(2 * n), (rows, cols)), shape=(n, len(uniq)))
    names = [f"c_{u}" for u in uniq]
    if drop_first:
        M, names = M[:, 1:], names[1:]
    return M, names


def fe_country_year(iso_i: Sequence, iso_j: Sequence, year: Sequence,
                    base_year: int | None = None) -> tuple[sp.csr_matrix, list[str]]:
    """Эффекты «страна×год» (контроль многосторонних сопротивлений) с мультичленством.

    Если в модели присутствуют парные эффекты, то страновые эффекты базового года
    нормируются на ноль (иначе они коллинеарны парным эффектам: a_ct = u_c,
    b_ij = −(u_i + u_j)). base_year=-1 — оставить все (без нормировки).
    """
    iso_i, iso_j, year = np.asarray(iso_i), np.asarray(iso_j), np.asarray(year)
    base = year.min() if base_year is None else base_year
    keys_i = pd.Series(iso_i) + "_" + pd.Series(year).astype(str)
    keys_j = pd.Series(iso_j) + "_" + pd.Series(year).astype(str)
    uniq = sorted(set(keys_i) | set(keys_j))
    uniq = [u for u in uniq if not u.endswith("_" + str(base))]
    ix = {u: k for k, u in enumerate(uniq)}
    n = len(iso_i)
    r, c = [], []
    for k, (a, b) in enumerate(zip(keys_i, keys_j)):
        if a in ix:
            r.append(k); c.append(ix[a])
        if b in ix:
            r.append(k); c.append(ix[b])
    M = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, len(uniq)))
    return M, [f"cy_{u}" for u in uniq]


# ----------------------------------------------------------------------------
# Результат
# ----------------------------------------------------------------------------
@dataclass
class FitResult:
    names: list[str]
    params: np.ndarray
    se: np.ndarray                 # кластерно-робастные (по парам)
    se_dyadic: np.ndarray | None   # диадно-робастные
    vcov: np.ndarray
    nobs: int
    n_params_total: int
    n_clusters: int
    loglik: float
    pseudo_r2: float               # квадрат корреляции y и \hat{y}
    iterations: int
    converged: bool
    mu: np.ndarray = field(repr=False, default=None)       # подогнанные значения на выборке оценивания
    mu_all: np.ndarray = field(repr=False, default=None)   # прогноз для всех наблюдений
    theta: np.ndarray = field(repr=False, default=None)    # полный вектор параметров (по исходным столбцам)
    kind: str = "ppml"
    dropped: list = field(default_factory=list)            # регрессоры, поглощённые ФЭ/коллинеарные

    def table(self) -> pd.DataFrame:
        from scipy import stats
        z = self.params / self.se
        df = pd.DataFrame({"coef": self.params, "se": self.se, "z": z,
                           "p": 2 * stats.norm.sf(np.abs(z))}, index=self.names)
        if self.se_dyadic is not None:
            zd = self.params / self.se_dyadic
            df["se_dyadic"] = self.se_dyadic
            df["p_dyadic"] = 2 * stats.norm.sf(np.abs(zd))
        df["pct_effect"] = 100 * (np.exp(df["coef"]) - 1)
        return df


def _dyadic_adjacency(iso_i: np.ndarray, iso_j: np.ndarray, pair_codes: np.ndarray) -> sp.csr_matrix:
    """Матрица смежности пар: 1, если пары имеют хотя бы одну общую страну (включая диагональ)."""
    n_pairs = pair_codes.max() + 1
    first = {}
    for a, b, p in zip(iso_i, iso_j, pair_codes):
        first[p] = (a, b)
    ps = np.arange(n_pairs)
    countries = sorted({c for ab in first.values() for c in ab})
    ix = {c: k for k, c in enumerate(countries)}
    rows = np.r_[ps, ps]
    cols = np.r_[[ix[first[p][0]] for p in ps], [ix[first[p][1]] for p in ps]]
    inc = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_pairs, len(countries)))
    W = (inc @ inc.T).tocsr()
    W.data[:] = 1.0
    return W


# ----------------------------------------------------------------------------
# Вспомогательные функции линейной алгебры
# ----------------------------------------------------------------------------
def _assemble(blocks):
    mats, names = [], []
    for m, nm in blocks:
        mats.append(sp.csr_matrix(m) if not sp.issparse(m) else m.tocsr())
        names += list(nm)
    return sp.hstack(mats).tocsr(), names


def _solve_psd(H: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Решить H x = rhs (H симметрична, неотрицательно определена)."""
    try:
        c = sla.cho_factor(H, lower=True, check_finite=False)
        return sla.cho_solve(c, rhs, check_finite=False)
    except (np.linalg.LinAlgError, sla.LinAlgError):
        # вырожденность: псевдообратная (параметры в ядре получают нулевые значения)
        return np.linalg.pinv(H, rcond=1e-12, hermitian=True) @ rhs


def _independent_covariates(X: sp.csr_matrix, n_cov: int, tol: float = 1e-8) -> np.ndarray:
    """Булев вектор: какие из первых n_cov столбцов линейно независимы от ФЭ и друг от друга.

    Используется дополнение Шура Sn = H_cc − H_cf H_ff^+ H_fc (после нормировки на диагональ):
    остаточная доля дисперсии регрессора, не объяснённая ФЭ. Поглощённые регрессоры
    (например, время-инвариантная переменная при парных ФЭ) отбрасываются.
    """
    H = (X.T @ X).toarray()
    Hcc, Hcf, Hff = H[:n_cov, :n_cov], H[:n_cov, n_cov:], H[n_cov:, n_cov:]
    S = Hcc - Hcf @ np.linalg.pinv(Hff, rcond=1e-10, hermitian=True) @ Hcf.T
    sc = np.sqrt(np.maximum(np.diag(Hcc), 1e-300))
    Sn = S / np.outer(sc, sc)
    Sn = 0.5 * (Sn + Sn.T)
    _, R, piv = sla.qr(Sn, pivoting=True)
    diag = np.abs(np.diag(R))
    keep = np.zeros(n_cov, dtype=bool)
    keep[piv[diag > tol]] = True
    return keep


# ----------------------------------------------------------------------------
# Основная процедура
# ----------------------------------------------------------------------------
def fit(y: np.ndarray, blocks: list, cluster: np.ndarray, n_cov: int, *,
        kind: str = "ppml", iso_i: np.ndarray | None = None, iso_j: np.ndarray | None = None,
        dyadic: bool = False, n_nested: int = 0, tol: float = 1e-10, maxit: int = 80,
        verbose: bool = False, drop_collinear: bool = True, sample: np.ndarray | None = None) -> FitResult:
    """Оценить PPML (kind='ppml') или МНК (kind='ols', y — в логарифмах).

    n_cov    — число интересующих коэффициентов (первые столбцы первого блока);
    n_nested — число параметров ФЭ, вложенных в кластеры (парные ФЭ при кластеризации
               по парам): не входят в поправку на степени свободы (как в reghdfe);
    sample   — булева маска строк для оценивания; прогноз ``mu_all`` строится для всех строк.
    """
    y_all = np.asarray(y, dtype=float)
    X_all, names_all = _assemble(blocks)
    n_all, p_all = X_all.shape
    mask = np.ones(n_all, dtype=bool) if sample is None else np.asarray(sample, dtype=bool)
    cl_all = np.asarray(cluster)
    iso_i_all = None if iso_i is None else np.asarray(iso_i)
    iso_j_all = None if iso_j is None else np.asarray(iso_j)

    X = X_all[mask].tocsr()
    y = y_all[mask]
    # столбцы без носителя в подвыборке исключаются (их параметры = 0)
    support = np.asarray(abs(X).sum(axis=0)).ravel() > 0
    col_idx = np.where(support)[0]
    X = X[:, col_idx].tocsr()
    n_cov_s = int(support[:n_cov].sum())
    dropped = [names_all[k] for k in range(n_cov) if not support[k]]
    # коллинеарные/поглощённые ФЭ регрессоры
    if drop_collinear and n_cov_s > 0:
        keep = _independent_covariates(X, n_cov_s)
        if not keep.all():
            dropped += [names_all[col_idx[k]] for k in range(n_cov_s) if not keep[k]]
            cols = [k for k in range(X.shape[1]) if k >= n_cov_s or keep[k]]
            X = X[:, cols].tocsr()
            col_idx = col_idx[cols]
            n_cov_s = int(keep.sum())
    names = [names_all[k] for k in col_idx]
    n, p = X.shape
    cl_codes, cl_uniq = pd.factorize(cl_all[mask])
    G = len(cl_uniq)
    Xt = X.T.tocsr()

    if kind == "ols":
        H = (Xt @ X).toarray()
        theta = _solve_psd(H, Xt @ y)
        mu = X @ theta
        resid = y - mu
        it, conv = 1, True
        ll = -0.5 * n * np.log(np.sum(resid ** 2) / n)
    else:
        if (y < 0).any():
            raise ValueError("PPML: отрицательные значения зависимой переменной")
        ly = np.log(np.maximum(y, 1e-12))
        H0 = (Xt @ X).toarray()
        theta = _solve_psd(H0, Xt @ ly)          # старт: МНК по log(y)
        eta = np.clip(X @ theta, -50, 50)
        mu = np.exp(eta)
        ll = np.sum(y * eta - mu)
        conv = False
        for it in range(1, maxit + 1):
            grad = Xt @ (y - mu)
            H = (Xt @ sp.diags(mu) @ X).toarray()
            step = _solve_psd(H, grad)
            t = 1.0
            while True:
                th_new = theta + t * step
                eta_new = np.clip(X @ th_new, -50, 50)
                mu_new = np.exp(eta_new)
                ll_new = np.sum(y * eta_new - mu_new)
                if ll_new >= ll - 1e-12 or t < 1e-6:
                    break
                t *= 0.5
            d_ll = ll_new - ll
            theta, eta, mu, ll = th_new, eta_new, mu_new, ll_new
            if verbose:
                print(f"  iter {it}: ll={ll:.6f} step={t:.3g} dll={d_ll:.3e}")
            if abs(d_ll) < tol * (1 + abs(ll)) and np.max(np.abs(t * step)) < 1e-7:
                conv = True
                break
        resid = y - mu
        H = (Xt @ sp.diags(mu) @ X).toarray()

    # --- ковариационная матрица (кластерная) для интересующих параметров ---
    k = n_cov_s
    Hinv_k = _solve_psd(H, np.eye(p)[:, :k])                       # p × k
    U = sp.diags(resid) @ X                                        # n × p  (скоры по наблюдениям)
    Ind = sp.csr_matrix((np.ones(n), (np.arange(n), cl_codes)), shape=(n, G))
    S = (U.T @ Ind).toarray()                                      # p × G
    C = Hinv_k.T @ S                                               # k × G
    dfc = (G / (G - 1.0)) * ((n - 1.0) / max(n - (p - n_nested), 1.0))
    V = dfc * (C @ C.T)
    se = np.sqrt(np.diag(V))

    se_dy = None
    if dyadic:
        if iso_i_all is None or iso_j_all is None:
            raise ValueError("для диадно-робастных ошибок нужны iso_i, iso_j")
        W = _dyadic_adjacency(iso_i_all[mask], iso_j_all[mask], cl_codes)
        M = C @ (W @ C.T)
        M = 0.5 * (M + M.T)
        ev, evec = np.linalg.eigh(M)
        M_psd = (evec * np.clip(ev, 0, None)) @ evec.T
        se_dy = np.sqrt(np.diag(dfc * M_psd))

    theta_full = np.zeros(p_all)
    theta_full[col_idx] = theta
    mu_all = np.exp(np.clip(X_all @ theta_full, -50, 50)) if kind == "ppml" else X_all @ theta_full
    yh = mu if kind == "ppml" else mu
    corr = np.corrcoef(y, yh)[0, 1] ** 2 if np.std(yh) > 0 else np.nan
    return FitResult(names=names[:k], params=theta[:k], se=se, se_dyadic=se_dy, vcov=V,
                     nobs=n, n_params_total=p, n_clusters=G, loglik=float(ll), pseudo_r2=float(corr),
                     iterations=it, converged=conv, mu=mu, mu_all=mu_all, theta=theta_full, kind=kind,
                     dropped=dropped)
