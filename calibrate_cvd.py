"""
실측 데이터로 CVD 모델 상수 보정 (cvd_model.py)  →  cvd_calibrated.json, cvd_calibration.png

사용법:
    python calibrate_cvd.py --data data/cvd_data.csv
    python calibrate_cvd.py --data data/cvd_data.csv --fit k0,Ea,D0,x0,H --cv
    python calibrate_cvd.py --demo            # 합성 데이터로 '알고 있는 정답'을 되찾는지 확인

데이터 CSV (한 행 = 측정점 하나)
  필수:  T_C (또는 T_K) · x_cm, y_cm (또는 Inlet 거리 d_cm) · thickness_nm (또는 rate_nm_min)
  선택:  P_Torr · p_prec_Torr · u_cm_s · time_min
         → 없는 열은 명령줄 값(--P, --p-prec, --u, --time)으로 채움. 행마다 다르면 열로 넣기
  예)
    T_C,x_cm,y_cm,thickness_nm
    500,0.0,10.5,48.2
    500,-1.5,9.0,50.1
    ...

피팅 파라미터 (--fit, 쉼표로 구분, 기본 k0,Ea,D0,x0)
    k0, Ea : 표면 반응 (내부적으로 k0 대신 '데이터 평균 온도에서의 k_s'로 바꿔 피팅 → k0·Eₐ 강한 상관 완화)
    D0     : 확산 계수 (1 atm, 300 K)
    x0     : 입구 → 웨이퍼 시작점 오프셋
    H      : 가스 통로 유효 높이 (전구체 고갈 정도를 정함)
    nu0, a_D, a_nu : 필요할 때만

식별 가능성 주의
- h_g = D_g / (5·√(ν(x+x₀)/u)) 이라 D₀와 ν₀는 D₀/√ν₀ 묶음으로만 들어감 → 기본은 ν₀ 고정, D₀만 피팅
- 반응 율속(저온) 데이터만 있으면 k_s만 정해지고 D₀·x₀는 정해지지 않음 (반대로 고온만 있으면 k₀·Eₐ가 불확실)
  → 온도를 여러 개, 가능하면 반응 율속 ↔ 물질 전달 율속 전환 구간을 걸치게
- 두께 스케일은 p_prec · M/ρ 와 k_eff의 곱 → p_prec가 부정확하면 k₀가 그만큼 보정해 줌(값 해석 주의)
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import cvd_model as M

LOG_P = {"k0", "D0", "x0", "H", "nu0"}  # log10 공간에서 피팅
BOUNDS = {
    "k0": (1e-3, 1e20), "Ea": (5.0, 400.0), "D0": (1e-3, 10.0), "x0": (1e-3, 50.0),
    "H": (0.05, 50.0), "nu0": (1e-3, 10.0), "a_D": (0.0, 2.5), "a_nu": (0.0, 2.5),
}
KS_BOUNDS = (1e-8, 1e6)   # 기준 온도에서의 k_s [cm/s]


# ---------------------------------------------------------------------------
# 데이터
# ---------------------------------------------------------------------------
def load(path, defaults):
    """CSV → 표준 열(T, P, p_prec, u, t_min, x, y, y_obs, obs_kind)"""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame(index=df.index)
    if "T_K" in df:
        out["T"] = df.T_K.astype(float)
    elif "T_C" in df:
        out["T"] = df.T_C.astype(float) + 273.15
    else:
        raise SystemExit("온도 열이 필요해요: T_C 또는 T_K")
    if {"x_cm", "y_cm"} <= set(df.columns):
        out["x"], out["y"] = df.x_cm.astype(float), df.y_cm.astype(float)
    elif "d_cm" in df:
        out["x"], out["y"] = 0.0, df.d_cm.astype(float)
    else:
        raise SystemExit("위치 열이 필요해요: x_cm,y_cm 또는 d_cm (Gas inlet에서의 거리)")
    for col, key in [("P_Torr", "P"), ("p_prec_Torr", "p_prec"), ("u_cm_s", "u"), ("time_min", "t_min")]:
        out[key] = df[col].astype(float) if col in df else float(defaults[key])
    if "thickness_nm" in df:
        out["y_obs"], kind = df.thickness_nm.astype(float), "thickness_nm"
    elif "rate_nm_min" in df:
        out["y_obs"], kind = df.rate_nm_min.astype(float), "rate_nm_min"
        out["t_min"] = 1.0   # 증착률을 맞출 때는 시간 = 1 min으로 두께 = 증착률
    else:
        raise SystemExit("측정값 열이 필요해요: thickness_nm 또는 rate_nm_min")
    out = out.dropna()
    return out, kind


def predict(c, d):
    return M.thickness(d["T"].values, d["P"].values, d["p_prec"].values, d["u"].values, d["t_min"].values,
                       d["x"].values, d["y"].values, c)


def condition_keys(d):
    return ["T", "P", "p_prec", "u", "t_min"]


# ---------------------------------------------------------------------------
# 피팅
# ---------------------------------------------------------------------------
class Packer:
    """상수 dict ↔ 피팅 벡터. k0는 k_s(T_ref)로 바꿔서 다룸"""

    def __init__(self, names, base, T_ref):
        self.names, self.base, self.T_ref = list(names), dict(base), float(T_ref)

    def _ks_ref(self, c):
        return c["k0"] * np.exp(-c["Ea"] * 1e3 / (M.R_GAS * self.T_ref))

    def vec(self, c):
        v = []
        for k in self.names:
            if k == "k0":
                v.append(np.log10(self._ks_ref(c)))
            elif k in LOG_P:
                v.append(np.log10(c[k]))
            else:
                v.append(c[k])
        return np.array(v, float)

    def dict(self, v):
        c = dict(self.base)
        for k, x in zip(self.names, v):
            if k != "k0":
                c[k] = 10 ** x if k in LOG_P else float(x)
        if "k0" in self.names:
            ks_ref = 10 ** v[self.names.index("k0")]
            c["k0"] = float(ks_ref * np.exp(c["Ea"] * 1e3 / (M.R_GAS * self.T_ref)))
        return c

    def bounds(self):
        lb, ub = [], []
        for k in self.names:
            lo, hi = KS_BOUNDS if k == "k0" else BOUNDS[k]
            if k == "k0" or k in LOG_P:
                lo, hi = np.log10(lo), np.log10(hi)
            lb.append(lo)
            ub.append(hi)
        return np.array(lb), np.array(ub)


def fit(d, names, base, x_start=None, n_start=40, seed=0, rel=False):
    """least_squares 다중 시작. rel=True면 상대 오차(두께 범위가 몇 배씩 차이 날 때)"""
    pk = Packer(names, base, T_ref=float(np.exp(np.mean(np.log(d["T"].values)))))
    lb, ub = pk.bounds()
    y = d["y_obs"].values
    w = 1 / np.maximum(np.abs(y), 1e-9) if rel else 1.0

    def res(v):
        with np.errstate(all="ignore"):
            r = (predict(pk.dict(v), d) - y) * w
        return np.where(np.isfinite(r), r, 1e6)

    starts = [np.clip(pk.vec(x_start or base), lb, ub)]
    rng = np.random.default_rng(seed)
    starts += [rng.uniform(lb, ub) for _ in range(n_start)]
    best = None
    for s in starts:
        try:
            r = least_squares(res, s, bounds=(lb, ub), x_scale="jac")
        except (ValueError, FloatingPointError):
            continue
        if best is None or r.cost < best.cost:
            best = r
    c = pk.dict(best.x)
    # 파라미터 불확도 (선형 근사, 피팅 공간 기준 1σ)
    sd = None
    try:
        J = best.jac
        dof = max(len(y) - len(names), 1)
        s2 = 2 * best.cost / dof
        cov = np.linalg.pinv(J.T @ J) * s2
        sd = dict(zip(names, np.sqrt(np.clip(np.diag(cov), 0, None)).tolist()))
    except (np.linalg.LinAlgError, ValueError):
        pass
    return c, best, sd, pk


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


# ---------------------------------------------------------------------------
# 합성 데이터 (--demo)
# ---------------------------------------------------------------------------
def demo_data(true_c, noise=0.02, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for T_C in [450, 500, 550, 600, 650, 700]:
        for (x, y) in M.PTS:
            rows.append(dict(T_C=T_C, x_cm=x, y_cm=y))
    df = pd.DataFrame(rows)
    p = M.DEFAULT_PROCESS
    th = M.thickness(df.T_C.values + 273.15, p["P"], p["p_prec"], p["u"], p["t_min"], df.x_cm.values,
                     df.y_cm.values, true_c)
    df["thickness_nm"] = th * (1 + noise * rng.standard_normal(len(df)))
    return df


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="CVD 모델 상수 보정")
    ap.add_argument("--data", default="data/cvd_data.csv")
    ap.add_argument("--fit", default="k0,Ea,D0,x0", help="피팅할 상수 (쉼표 구분)")
    ap.add_argument("--P", type=float, default=M.DEFAULT_PROCESS["P"], help="CSV에 P_Torr 열이 없을 때 [Torr]")
    ap.add_argument("--p-prec", type=float, default=M.DEFAULT_PROCESS["p_prec"], help="전구체 분압 [Torr]")
    ap.add_argument("--u", type=float, default=M.DEFAULT_PROCESS["u"], help="유속 [cm/s]")
    ap.add_argument("--time", type=float, default=M.DEFAULT_PROCESS["t_min"], help="증착 시간 [min]")
    ap.add_argument("--M", type=float, default=None, help="막 몰 질량 [g/mol] (기본 HfO₂)")
    ap.add_argument("--rho", type=float, default=None, help="막 밀도 [g/cm³] (기본 HfO₂)")
    ap.add_argument("--rel", action="store_true", help="상대 오차로 피팅 (온도별 두께가 몇 배씩 다를 때 권장)")
    ap.add_argument("--n-start", type=int, default=40, help="다중 시작 횟수")
    ap.add_argument("--cv", action="store_true", help="조건 하나씩 빼는 교차검증")
    ap.add_argument("--demo", action="store_true", help="합성 데이터로 파라미터 복원 테스트")
    ap.add_argument("--out", default="cvd_calibrated.json")
    ap.add_argument("--plot", default="cvd_calibration.png")
    a = ap.parse_args()

    base = dict(M.DEFAULT_CONSTS)
    if a.M:
        base["M"] = a.M
    if a.rho:
        base["rho"] = a.rho
    names = [s.strip() for s in a.fit.split(",") if s.strip()]
    bad = [n for n in names if n not in BOUNDS]
    if bad:
        raise SystemExit(f"피팅할 수 없는 상수: {bad}  (가능: {list(BOUNDS)})")
    defaults = {"P": a.P, "p_prec": a.p_prec, "u": a.u, "t_min": a.time}

    true_c = None
    if a.demo:
        true_c = {**base, "k0": 3.0e7, "Ea": 135.0, "D0": 0.07, "x0": 2.5}
        path = "/tmp/cvd_demo.csv"
        demo_data(true_c).to_csv(path, index=False)
        a.data, a.out, a.plot = path, "/tmp/cvd_demo_calibrated.json", "/tmp/cvd_demo_calibration.png"
        print(f"[demo] 합성 데이터(잡음 2 %) → {path}")

    d, kind = load(a.data, defaults)
    print(f"데이터 {len(d)}점 · 조건 {d.groupby(condition_keys(d)).ngroups}개 · 측정값 = {kind}")
    c, best, sd, pk = fit(d, names, base, n_start=a.n_start, rel=a.rel)
    p0 = predict(base, d)
    p1 = predict(c, d)
    y = d["y_obs"].values
    r2 = 1 - np.sum((p1 - y) ** 2) / np.sum((y - y.mean()) ** 2)

    print("== 보정된 상수 ==")
    for k in names:
        extra = f"   (정답 {true_c[k]:.4g})" if true_c else ""
        print(f"  {k:5s} = {c[k]:.4g}{extra}")
    if sd:
        print(f"  (1σ, 피팅 공간: k0→log10 k_s({pk.T_ref:.0f} K), log 파라미터는 log10) "
              + ", ".join(f"{k} ±{v:.3g}" for k, v in sd.items()))
    print("== 오차 ==")
    print(f"  보정 전 RMSE {rmse(p0 - y):.3f} · 보정 후 RMSE {rmse(p1 - y):.3f}  (R² {r2:.4f})")

    cv = None
    if a.cv:
        keys = condition_keys(d)
        groups = list(d.groupby(keys).groups.items())
        if len(groups) < 3:
            print("== 교차검증 건너뜀: 조건이 3개 미만 ==")
        else:
            e = []
            for _, idx in groups:
                tr, te = d.drop(idx), d.loc[idx]
                ci, *_ = fit(tr, names, base, x_start=c, n_start=0, rel=a.rel)
                e += list(predict(ci, te) - te["y_obs"].values)
            cv = {"loco_rmse": rmse(e)}
            print(f"== 교차검증 (조건 1개씩 제외, {len(groups)}-fold) ==  RMSE {cv['loco_rmse']:.3f}")

    out = {
        "source": a.data,
        "measured": kind,
        "fit": names,
        "assumed": {**defaults, "M": base["M"], "rho": base["rho"]},
        "consts": {k: float(v) for k, v in c.items()},
        "metrics": {"rmse_before": rmse(p0 - y), "rmse": rmse(p1 - y), "r2": float(r2), "n": int(len(d)),
                    **(cv or {})},
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"저장: {a.out}")
    if a.plot:
        make_plot(d, p1, c, a.plot, kind)
        print(f"저장: {a.plot}")


def make_plot(d, p, c, path, kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y = d["y_obs"].values
    unit = "nm" if kind == "thickness_nm" else "nm/min"
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    temps = np.sort(d["T"].unique())
    cmap = plt.cm.plasma
    for i, T in enumerate(temps):
        m = d["T"].values == T
        ax[0].scatter(y[m], p[m], s=14, color=cmap(i / max(len(temps) - 1, 1)), label=f"{T - 273.15:.0f} °C")
    hi = max(y.max(), p.max()) * 1.05
    ax[0].plot([0, hi], [0, hi], "k--", lw=1)
    ax[0].set(xlim=(0, hi), ylim=(0, hi), xlabel=f"Measured [{unit}]", ylabel=f"Predicted [{unit}]",
              title=f"Parity  RMSE {rmse(p - y):.3g} {unit}")
    ax[0].legend(fontsize=8, ncol=2)

    # 아레니우스: 웨이퍼 평균 (같은 P·p·u·t 조건끼리)
    g = d.assign(pred=p).groupby(condition_keys(d))[["y_obs", "pred"]].mean().reset_index()
    g = g.assign(rate_obs=g.y_obs / g.t_min, rate_pred=g.pred / g.t_min)
    ax[1].semilogy(1000 / g["T"], g.rate_obs, "o", color="#0d7f74", label="data (wafer mean)")
    TT = np.linspace(temps.min() - 50, temps.max() + 50, 200)
    r0 = g.iloc[0]
    d_mid = float(np.hypot(*M.WC))
    ax[1].semilogy(1000 / TT, M.rate(d_mid, TT, r0.P, r0.p_prec, r0.u, c), "-", color="#eb6834",
                   label=f"model @ wafer center ({d_mid:.1f} cm)")
    ax[1].semilogy(1000 / TT, M.c_g0(TT, r0.p_prec) * M.k_s(TT, c) * c["M"] / c["rho"] * 1e7 * 60, ":",
                   color="gray", label="reaction limit (k_s only)")
    ax[1].set_ylim(g.rate_obs.min() / 5, g.rate_obs.max() * 5)
    ax[1].set(xlabel="1000 / T [1/K]", ylabel="Growth rate [nm/min]", title="Arrhenius")
    ax[1].legend(fontsize=8)

    # 거리 프로파일 (온도별)
    dd = np.hypot(d["x"].values, d["y"].values)
    xs = np.linspace(max(dd.min() - 1, 0), dd.max() + 1, 100)
    for i, T in enumerate(temps):
        m = d["T"].values == T
        col = cmap(i / max(len(temps) - 1, 1))
        r = d[m].iloc[0]
        ax[2].plot(dd[m], y[m], "o", color=col, ms=4)
        ax[2].plot(xs, M.rate(xs, T, r.P, r.p_prec, r.u, c) * r.t_min, "-", color=col)
    ax[2].set(xlabel="Distance from gas inlet [cm]", ylabel=f"[{unit}]", title="Profile along flow (dots = data)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
