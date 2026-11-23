"""
PVD 모델 최적화 — 논문 Figure 2 조건 → pvd_calibrated.json, pvd_calibration.png

    python calibrate_pvd.py            # 최적화 + 그림
    python calibrate_pvd.py --no-plot

논문: Z. Cheng et al., Cryst. Growth Des. 24, 339–346 (2024), Figure 2 · Table 1 Group 1
데이터 (그림을 축 눈금 픽셀로 보정해서 디지타이즈)
  data/pvd_fig2cd_thickness.csv   Fig. 2(c, d) 실측 Mo 두께 18점 × 2장 (a = 25, 30 mm · d_ts = 9 cm · 평면 타깃)
  data/pvd_fig2a_theory.csv       Fig. 2(a) 논문 이론 곡선 (a = 3–7 cm · d_ts = 10 cm) — 비교용, 피팅에는 안 씀
  data/pvd_fig2b_eta_theory.csv   Fig. 2(b) 논문 이론 불균일도 — 비교용

무엇을 맞추나
  실측 두께 36점. 모델은 상대 분포만 계산하므로 웨이퍼마다 중심 두께 T_ref 를 최소제곱으로 맞추고(선형, 해석해)
  모양 오차만 본다:  잔차 = (T_ref · T_rel(x_i) − t_i) / t̄   (웨이퍼 평균 대비 %)

후보 모델 (kernel = 타깃 미소 면적의 기여)
  paper     H²/L⁴ · S                          λ = 1.6 cm, 피팅 변수 없음 (논문 식 그대로)
  lam       H²/L⁴ · S                          λ 피팅
  scatter   H²/L⁴ · S + γ · H/(2L³) · (1 − S)   λ = 1.6 cm 고정, γ 피팅
  both      위와 같음                           λ, γ 둘 다 피팅
  (S = exp(−L/λ).  산란 성분 H/(2πL³)은 '한 점에서 나와 확산하다 흡수 평면에 닿는 원자'의 도달 분포)

검증
  교차검증: 웨이퍼 한 장으로 맞추고 다른 한 장을 예측 (2-fold)
  잡음 바닥: 실측의 좌우 비대칭 성분 — 축대칭 모델은 원리상 이보다 잘 맞출 수 없다
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import pvd_model as P

HERE = Path(__file__).parent
H_EXP = 9.0                       # Table 1 Group 1: d_ts = 9 cm
H_THEORY = 10.0                   # Fig. 2(a, b) 이론 곡선: d_ts = 10 cm
A_EXP = {"t_a25mm_nm": 2.5, "t_a30mm_nm": 3.0}
ETA_PAPER = {2.5: 4.23, 3.0: 2.85}          # Fig. 2(c, d)에 적힌 값
GAMMA_MAX = 1.0                   # 산란된 원자가 방출된 것보다 많이 도달할 수는 없다
LAM_MAX = 1e3

MODELS = {
    # 이름: (피팅 변수, 시작값들)
    "paper": ((), [()]),
    "lam": (("lam",), [(1.6,), (5.0,), (30.0,)]),
    "scatter": (("gamma",), [(0.01,), (0.1,), (0.5,)]),
    "both": (("lam", "gamma"), [(1.6, 0.1), (5.0, 0.5), (0.8, 0.01), (20.0, 0.1)]),
}
BOUNDS = {"lam": (0.2, LAM_MAX), "gamma": (1e-5, GAMMA_MAX)}


def load():
    df = pd.read_csv(HERE / "data" / "pvd_fig2cd_thickness.csv")
    return {a: df[c].to_numpy(float) for c, a in A_EXP.items()}


def params(names, v):
    """log10 변수 → {'lam':…, 'gamma':…} (없는 건 논문 값: λ = 1.6 cm, γ = 0)"""
    p = {"lam": P.LAM_REF, "gamma": 0.0}
    p.update({n: 10.0 ** x for n, x in zip(names, v)})
    return p
    


def shape(a, p, H=H_EXP, r_max=None):
    """직경 18점의 상대 두께"""
    x = P.diameter_points(P.RW_2INCH, P.N_POINTS_PAPER, r_max)
    return P.t_relative(np.abs(x), a, H, p["lam"], gamma=p["gamma"])


def residual(v, names, data, r_max=None):
    p = params(names, v)
    out = []
    for a, t in data.items():
        m = shape(a, p, r_max=r_max)
        T_ref = (m @ t) / (m @ m)                 # 중심 두께: 선형 최소제곱
        out.append((T_ref * m - t) / t.mean())
    return np.concatenate(out)


def fit(model, data, r_max=None):
    names, starts = MODELS[model]
    if not names:
        return {}, residual((), names, data, r_max)
    lb = [np.log10(BOUNDS[n][0]) for n in names]
    ub = [np.log10(BOUNDS[n][1]) for n in names]
    best = None
    for s in starts:
        r = least_squares(residual, np.log10(s), bounds=(lb, ub), args=(names, data, r_max))
        if best is None or r.cost < best.cost:
            best = r
    return {n: float(10.0 ** x) for n, x in zip(names, best.x)}, best.fun


def metrics(res):
    return {"rmse_pct": float(np.sqrt(np.mean(res**2)) * 100), "mape_pct": float(np.mean(np.abs(res)) * 100),
            "max_abs_pct": float(np.max(np.abs(res)) * 100), "n": int(res.size)}


def eta_model(a, p, r_max=None):
    return P.nonuniformity(shape(a, p, r_max=r_max))[2]


def cross_validate(model, data):
    """한 장으로 맞추고 다른 장 예측 (T_ref 는 예측 대상 웨이퍼에서 다시 맞춤 — 모델이 절대 두께를 계산하지 않으므로)"""
    names = MODELS[model][0]
    out = {}
    for test in data:
        train = {a: t for a, t in data.items() if a != test}
        fitted, _ = fit(model, train)
        v = [np.log10(fitted[n]) for n in names]
        out[f"predict_a{test:g}cm"] = metrics(residual(v, names, {test: data[test]}))["rmse_pct"]
    return out


def noise_floor(data):
    """좌우 비대칭 성분 RMS [%] — 축대칭 모델이 맞출 수 없는 부분"""
    res = np.concatenate([(t - 0.5 * (t + t[::-1])) / t.mean() for t in data.values()])
    return float(np.sqrt(np.mean(res**2)) * 100)


def theory_compare(p):
    """논문 Fig. 2(a, b) 이론 곡선과의 차이 (d_ts = 10 cm). 피팅에는 쓰지 않음"""
    cur = pd.read_csv(HERE / "data" / "pvd_fig2a_theory.csv")
    eta_t = pd.read_csv(HERE / "data" / "pvd_fig2b_eta_theory.csv")
    x = cur["x_cm"].to_numpy()
    diff, eta = [], {}
    xs = np.linspace(-3.0, 3.0, 121)                       # 논문 (b)는 (a)의 x 범위(±3 cm)에서 계산한 값과 맞는다
    for a, e_paper in zip(eta_t["a_cm"], eta_t["eta_theory_pct"]):
        m = P.t_relative(x, float(a), H_THEORY, p["lam"], gamma=p["gamma"])
        diff.append(m - cur[f"a{a:g}cm"].to_numpy())
        e = P.nonuniformity(P.t_relative(np.abs(xs), float(a), H_THEORY, p["lam"], gamma=p["gamma"]))[2]
        eta[f"{a:g}"] = {"paper": float(e_paper), "model": float(e)}
    d = np.concatenate(diff)
    return {"curve_rmse": float(np.sqrt(np.mean(d**2))), "curve_max_abs": float(np.max(np.abs(d))), "eta_pm3cm": eta}


def scan(data):
    """목적함수 단면: 각 변수가 데이터로 얼마나 정해지는지"""
    g = np.logspace(-4, 0, 25)
    lam = np.logspace(np.log10(0.3), 2, 25)
    rg = [metrics(residual([np.log10(x)], ("gamma",), data))["rmse_pct"] for x in g]
    rl0 = [metrics(residual([np.log10(x)], ("lam",), data))["rmse_pct"] for x in lam]
    rl1 = [metrics(residual([np.log10(x), 0.0], ("lam", "gamma"), data))["rmse_pct"] for x in lam]
    return {"gamma": g, "rmse_vs_gamma": rg, "lam": lam, "rmse_vs_lam_gamma0": rl0, "rmse_vs_lam_gamma1": rl1}


def plot(data, results, chosen, sc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p0, p1 = results["paper"]["p"], results[chosen]["p"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.6))
    idx = np.arange(1, P.N_POINTS_PAPER + 1)
    for k, (a, t) in enumerate(data.items()):
        for p, lab, col, ls in ((p0, "paper model (λ=1.6, γ=0)", "#868d98", "--"), (p1, "optimized", "#0d7f74", "-")):
            m = shape(a, p)
            T = (m @ t) / (m @ m) * m
            ax[0, k].plot(idx, T, ls, color=col, lw=2, label=f"{lab}: η = {P.nonuniformity(T)[2]:.2f} %")
        ax[0, k].plot(idx, t, "o", color="#5b63b7", ms=7, label=f"measured: η = {P.nonuniformity(t)[2]:.2f} %")
        ax[0, k].set(title=f"Fig. 2({'cd'[k]})  a = {a * 10:g} mm, d_ts = 9 cm", xlabel="Position", ylabel="Thickness [nm]",
                     ylim=(0.8 * t.min(), 1.1 * t.max()))
        ax[0, k].legend(fontsize=8, loc="lower center")
        ax[0, k].grid(alpha=0.3)

    ax[0, 2].semilogx(sc["gamma"], sc["rmse_vs_gamma"], color="#0d7f74", lw=2)
    ax[0, 2].axhline(results["paper"]["fit"]["rmse_pct"], color="#868d98", ls="--", label="paper model")
    ax[0, 2].axhline(noise_floor(data), color="#d7191c", ls=":", label="noise floor (left–right asymmetry)")
    ax[0, 2].set(title="RMSE vs γ  (λ = 1.6 cm)", xlabel="γ (scattered-atom arrival)", ylabel="shape RMSE [%]", ylim=(0, None))
    ax[0, 2].legend(fontsize=8)
    ax[0, 2].grid(alpha=0.3)

    cur = pd.read_csv(HERE / "data" / "pvd_fig2a_theory.csv")
    x = cur["x_cm"].to_numpy()
    xx = np.linspace(0, 3, 61)
    cols = ["#333333", "#d7191c", "#1e40ff", "#22a04a", "#8a5cc2"]
    for k, (p, name) in enumerate(((p0, "paper model as coded"), (p1, "optimized"))):
        for a, c in zip(P.FIG2A["a_list"], cols):
            ax[1, k].plot(x, cur[f"a{a:g}cm"], ".", color=c, ms=4)
            ax[1, k].plot(xx, P.t_relative(xx, a, H_THEORY, p["lam"], gamma=p["gamma"]), color=c, lw=1.6, label=f"a = {a:g} cm")
        ax[1, k].set(title=f"Fig. 2(a) theory curves (dots) vs {name} (lines)", xlabel="x [cm]", ylabel="Relative thickness",
                     ylim=(0.35, 1.02))
        ax[1, k].legend(fontsize=8)
        ax[1, k].grid(alpha=0.3)

    ax[1, 2].semilogx(sc["lam"], sc["rmse_vs_lam_gamma0"], color="#868d98", lw=2, label="γ = 0 (paper kernel)")
    ax[1, 2].semilogx(sc["lam"], sc["rmse_vs_lam_gamma1"], color="#0d7f74", lw=2, label="γ = 1")
    ax[1, 2].axvline(P.LAM_REF, color="k", lw=0.8, ls=":")
    ax[1, 2].set(title="RMSE vs λ", xlabel="λ [cm]", ylabel="shape RMSE [%]", ylim=(0, None))
    ax[1, 2].legend(fontsize=8)
    ax[1, 2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "pvd_calibration.png", dpi=130)


def main(do_plot=True):
    data = load()
    floor = noise_floor(data)
    print(f"실측 36점 · 잡음 바닥(좌우 비대칭) {floor:.2f} %")
    for a, t in data.items():
        m, s, e = P.nonuniformity(t)
        print(f"  a = {a * 10:g} mm: 디지타이즈 {m:.2f} ± {s:.2f} nm, η = {e:.2f} %  (논문 표기 η = {ETA_PAPER[a]} %)")

    results = {}
    print(f"\n{'모델':8} {'λ [cm]':>9} {'γ':>8} {'RMSE %':>7} {'MAPE %':>7} {'max %':>6} {'η 25mm':>7} {'η 30mm':>7} {'CV RMSE %':>12}")
    for model in MODELS:
        fitted, res = fit(model, data)
        p = {"lam": P.LAM_REF, "gamma": 0.0, **fitted}
        cv = cross_validate(model, data)
        results[model] = {"p": p, "fitted": list(fitted), "fit": metrics(res), "cv_rmse_pct": cv,
                          "eta_model_pct": {f"{a:g}": eta_model(a, p) for a in data}}
        f = results[model]["fit"]
        print(f"{model:8} {p['lam']:9.3g} {p['gamma']:8.3g} {f['rmse_pct']:7.2f} {f['mape_pct']:7.2f} {f['max_abs_pct']:6.2f} "
              f"{eta_model(2.5, p):7.2f} {eta_model(3.0, p):7.2f}   {' / '.join(f'{v:.2f}' for v in cv.values())}")

    # 채택: 산란 성분 추가, λ 는 논문 값 유지 (아래 단면에서 보듯 λ 는 이 데이터로 정해지지 않음)
    chosen = "scatter"
    p = results[chosen]["p"]
    sc = scan(data)
    g_ok = sc["gamma"][np.array(sc["rmse_vs_gamma"]) <= results[chosen]["fit"]["rmse_pct"] + 0.05]
    print(f"\nγ 단면: RMSE 가 최적값 +0.05 %p 안에 드는 범위 γ ≥ {g_ok.min():.2g} (상한 {GAMMA_MAX:g}에 붙음)")

    for a in data:
        e_m = eta_model(a, p)
        results[chosen].setdefault("eta_error", {})[f"{a:g}"] = {
            "paper": ETA_PAPER[a], "model": e_m, "abs_err_pp": e_m - ETA_PAPER[a], "rel_err_pct": (e_m / ETA_PAPER[a] - 1) * 100}
    th0, th1 = theory_compare(results["paper"]["p"]), theory_compare(p)
    print(f"논문 Fig. 2(a) 이론 곡선과의 차이(상대 두께 RMSE): 논문 식 그대로 {th0['curve_rmse']:.3f} → 최적화 후 {th1['curve_rmse']:.3f}")

    # 측정 위치 가정(r_max)에 대한 민감도
    sens = {}
    for r_max in (2.54, 2.40, 2.25):
        fitted, res = fit(chosen, data, r_max=r_max)
        sens[f"{r_max:g}"] = {"gamma": fitted["gamma"], "rmse_pct": metrics(res)["rmse_pct"]}

    out = {
        "source": "Cheng et al., Cryst. Growth Des. 24, 339 (2024), Fig. 2(c, d) · Table 1 Group 1 (a = 25, 30 mm, d_ts = 9 cm, d = 0)",
        "data": "data/pvd_fig2cd_thickness.csv",
        "model": "kernel = H²/L⁴·S + γ·H/(2L³)·(1−S),  S = exp(−L/λ)",
        "params": {"lam": p["lam"], "gamma": p["gamma"]},
        "fitted": ["gamma"],
        "fixed": {"lam": "논문 값 1.6 cm (0.4 Pa)"},
        "noise_floor_pct": floor,
        "metrics": {"before": results["paper"]["fit"], "after": results[chosen]["fit"],
                    "cv_rmse_pct": results[chosen]["cv_rmse_pct"]},
        "eta_pct": {"before": results["paper"]["eta_model_pct"], "after": results[chosen]["eta_error"]},
        "candidates": {k: {"params": v["p"], "fit": v["fit"], "cv_rmse_pct": v["cv_rmse_pct"]} for k, v in results.items()},
        "theory_fig2ab": {"before": th0, "after": th1},
        "r_max_sensitivity": sens,
    }
    (HERE / "pvd_calibrated.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장: pvd_calibrated.json")
    if do_plot:
        plot(data, results, chosen, sc)
        print("저장: pvd_calibration.png")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true")
    main(do_plot=not ap.parse_args().no_plot)
