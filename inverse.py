"""
역설계: 목표 사양 → 공정 조건 추천 (Streamlit 의존성 없음, model.py 사용)

모델 계산이 빠르므로 후보 공정 조건을 격자로 '전부' 계산해서
제약(평균 두께 ± 허용 오차, 불균일도 1σ 상한, 최소 포화도)을 만족하는 조건 중
목표(불균일도 / 전구체 사용량 / 증착 온도 / 사이클 수 최소)가 가장 좋은 조건을 고른다.
→ 격자 간격 안에서는 최적이 보장된다 (베이지안 최적화 같은 근사 탐색이 필요 없음).

탐색 격자는 사이드바 조작부의 눈금과 같게 맞춘다 (추천값을 그대로 '적용'할 수 있도록).
- 웨이퍼 온도: 5 K 간격 · 전구체 온도: 1 °C 간격 · dose 시간: log₁₀ 0.01 간격 · 사이클 수: 10 간격
"""
import numpy as np
import pandas as pd

from model import PTS, WC, WR, coverage, p_vap, thickness

OBJECTIVES = {
    "nu": "불균일도 최소",
    "usage": "전구체 사용량 최소",
    "temp": "증착 온도 최소 (열 예산)",
    "cycles": "사이클 수 최소 (공정 시간)",
}
# 전구체 사용량 기준(= 1): 전구체 75 °C(P_ref) · dose 0.2 s · 100 cycle (논문 공정 조건)
REF_T_DOSE, REF_CYCLES = 0.2, 100
# 논문 실측 데이터 범위 (이 안이면 '측정 범위 안', 밖이면 외삽)
DATA_T_C = (100, 250)
DATA_TP_C = (55, 75)
DATA_T_DOSE = 0.2

T_K_STEP, TP_STEP, LOG_T_STEP, N_STEP = 5, 1, 0.01, 10
T_K_RANGE, TP_RANGE, N_RANGE = (300, 700), (40, 95), (10, 500)


def eval_points(region):
    """평가 지점: '9pt' = 측정 위치 9점, 'wafer' = 웨이퍼 전체 격자 (9점 바깥은 외삽)"""
    if region == "9pt":
        return np.array(PTS, dtype=float)
    g = np.linspace(-WR, WR, 15)
    X, Y = np.meshgrid(g + WC[0], g + WC[1])
    m = np.hypot(X - WC[0], Y - WC[1]) <= WR + 1e-9
    return np.stack([X[m], Y[m]], 1)


def candidate_grid(T_C_rng, Tp_rng, t_fixed, t_free, t_rng):
    TK = np.arange(T_K_RANGE[0], T_K_RANGE[1] + 1, T_K_STEP, dtype=float)
    TK = TK[(TK - 273.15 >= T_C_rng[0] - 1e-9) & (TK - 273.15 <= T_C_rng[1] + 1e-9)]
    Tp = np.arange(np.ceil(max(Tp_rng[0], TP_RANGE[0])), np.floor(min(Tp_rng[1], TP_RANGE[1])) + 1e-9, TP_STEP)
    if t_free:
        lt = np.unique(np.round(np.linspace(np.log10(t_rng[0]), np.log10(t_rng[1]), 21), 2))
    else:
        lt = np.array([round(float(np.log10(t_fixed)), 2)])
    TT, PP, LL = np.meshgrid(TK, Tp, lt, indexing="ij")
    return TT.ravel(), PP.ravel(), LL.ravel(), (TK, Tp, lt)


def evaluate(c, use_z, TK, Tp, log_t, pts, chunk=4000):
    """후보마다 사이클 1회 기준 평균 두께, 불균일도 1σ, 최소 포화도"""
    P0 = p_vap(Tp + 273.15, c)
    t = 10.0 ** log_t
    X, Y = pts[:, 0][None, :], pts[:, 1][None, :]
    D = np.hypot(X, Y)
    out = {k: np.empty(len(TK)) for k in ("mean1", "nu", "nur", "thmin")}
    for s in range(0, len(TK), chunk):
        sl = slice(s, s + chunk)
        T_, P_, t_ = TK[sl, None], P0[sl, None], t[sl, None]
        z = thickness(T_, P_, t_, 1, X, Y, c, use_zernike=use_z)
        th = coverage(T_, P_, t_, D, c)
        m = z.mean(1)
        out["mean1"][sl] = m
        out["nu"][sl] = z.std(1, ddof=1) / m * 100
        out["nur"][sl] = (z.max(1) - z.min(1)) / (2 * m) * 100
        out["thmin"][sl] = np.broadcast_to(th, z.shape).min(1)
    out["P0"] = P0
    return out


def pareto_mask(cx, cy):
    """두 값 모두 작을수록 좋은 Pareto 최적 집합 (1e-4 % 미만 차이는 계산 잡음으로 봄)"""
    keep = np.zeros(len(cx), bool)
    best = np.inf
    for i in np.lexsort((cy, cx)):
        if cy[i] < best - 1e-4:
            keep[i] = True
            best = cy[i]
    return keep


def search(c, use_z, target, tol, nu_max, theta_min, objective="nu", region="9pt",
           T_C_rng=(100, 250), Tp_rng=(55, 75), t_fixed=0.2, t_free=False, t_rng=(0.05, 1.0),
           n_fixed=100, n_free=True, margin=0.1, n_top=5):
    """margin: 공정 마진. 공급량(dose 시간)이 ±margin 변해도 조건을 만족해야 '만족'으로 본다 (0이면 검사 안 함)."""
    TK, Tp, LT, axes = candidate_grid(T_C_rng, Tp_rng, t_fixed, t_free, t_rng)
    pts = eval_points(region)
    ev = evaluate(c, use_z, TK, Tp, LT, pts)
    t = 10.0 ** LT
    # 공급량이 ±margin 변했을 때 (전구체 온도·dose 시간은 노출량 P0·t로만 작용하므로 t를 바꿔서 계산)
    var = [evaluate(c, use_z, TK, Tp, LT + np.log10(1 + s * margin), pts) for s in (-1, 1)] if margin > 0 else []

    # 사이클 수: 두께 ∝ 사이클 (모델 정의). 목표 두께에 맞는 사이클을 10 단위로 반올림
    if n_free:
        n = np.clip(np.round(target / np.maximum(ev["mean1"], 1e-12) / N_STEP) * N_STEP, *N_RANGE)
    else:
        n = np.full_like(TK, float(n_fixed))
    mean = ev["mean1"] * n
    usage = ev["P0"] / c["Pref"] * (t / REF_T_DOSE) * (n / REF_CYCLES)
    exposure = ev["P0"] * t                      # 모델에서 전구체 온도와 dose 시간은 이 곱으로만 작용

    # 최악값: 공칭 조건과 ±margin 조건 중 가장 나쁜 값 (사이클 수는 공칭 조건에서 정한 값 그대로)
    nu_w = np.max([ev["nu"]] + [v["nu"] for v in var], 0)
    th_w = np.min([ev["thmin"]] + [v["thmin"] for v in var], 0)
    dev_w = np.max([np.abs(mean - target)] + [np.abs(v["mean1"] * n - target) for v in var], 0)
    ok_thk = dev_w <= tol
    ok_nu = nu_w <= nu_max
    ok_th = th_w >= theta_min
    feas = ok_thk & ok_nu & ok_th
    T_C = TK - 273.15
    in_range = ((T_C >= DATA_T_C[0] - 2.5) & (T_C <= DATA_T_C[1] + 2.5) & (Tp >= DATA_TP_C[0]) & (Tp <= DATA_TP_C[1])
                & (np.abs(LT - np.log10(DATA_T_DOSE)) < 0.006))
    df = pd.DataFrame({"T_K": TK, "T_C": T_C, "Tp": Tp, "log_t": LT, "t": t, "n": n, "mean": mean,
                       "nu": ev["nu"], "nu_worst": nu_w, "nur": ev["nur"], "thmin": ev["thmin"], "thmin_worst": th_w, "dev_worst": dev_w, "P0": ev["P0"], "usage": usage,
                       "exposure": exposure, "ok_thk": ok_thk, "ok_nu": ok_nu, "ok_th": ok_th, "feasible": feas,
                       "in_range": in_range})

    # 순위: 모델 오차보다 훨씬 작은 차이는 같은 값으로 본다 (불균일도 0.1 %, 사용량 1 % 단위)
    nu_r = np.round(nu_w, 1)
    use_r = np.round(np.log(usage) / np.log(1.01))
    ext = (~in_range) * 1.0 + np.abs(LT - np.log10(DATA_T_DOSE)) + np.abs(np.log(n / REF_CYCLES)) * 0.1
    keys = {"nu": [nu_r, use_r, n, TK, ext], "usage": [use_r, nu_r, n, TK, ext],
            "temp": [TK, nu_r, use_r, n, ext], "cycles": [n, nu_r, use_r, TK, ext]}[objective]
    if feas.any():
        idx, status = np.flatnonzero(feas), "ok"
    else:   # 만족하는 조건이 없으면 제약 위반이 가장 작은 절충안
        viol = (np.maximum(0, dev_w - tol) / max(tol, 1e-9)
                + np.maximum(0, nu_w - nu_max) / max(nu_max, 1e-9)
                + np.maximum(0, theta_min - th_w) / max(theta_min, 1e-2))
        idx, status = np.arange(len(df)), "none"
        keys = [np.round(viol, 3)] + keys
    order = idx[np.lexsort([np.asarray(k)[idx] for k in reversed(keys)])]

    # 서로 다른 추천만: 온도 15 K 이상 또는 노출량(분압 × dose 시간) 15 % 이상 차이
    picks = []
    for i in order:
        r = df.iloc[i]
        if all(abs(r["T_K"] - q["T_K"]) >= 15 or abs(np.log(r["exposure"] / q["exposure"])) >= 0.15 for q in picks):
            picks.append(r)
        if len(picks) >= n_top:
            break
    top = pd.DataFrame(picks).reset_index(drop=True) if picks else df.iloc[:0]

    thk_ok = df[ok_thk & ok_th]
    pm = pareto_mask(thk_ok["usage"].to_numpy(), thk_ok["nu_worst"].to_numpy()) if len(thk_ok) else np.zeros(0, bool)
    return {"status": status, "top": top, "all": df, "pareto": thk_ok[pm].sort_values("usage"), "margin": margin, "axes": axes,
            "n_points": len(pts), "n_cand": len(df), "n_feas": int(feas.sum())}
