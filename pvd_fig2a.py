"""
PVD 모델 검산 + 논문 Figure 2(a) 조건 계산 → pvd_fig2a.png, pvd_fig2a.csv

    python pvd_fig2a.py            # 검산 후 그림·표 저장
    python pvd_fig2a.py --no-plot  # 검산과 표만

검산 항목 (scipy 적분과 독립 비교)
  1) λ → ∞, r = 0 해석해   F(0) = π·a²/(H² + a²)
  2) r = 0, 유한 λ          F(0) = 2π·H²·∫_H^√(H²+a²) e^(−L/λ)/L³ dL   (1차원 적분)
  3) r > 0                  scipy.integrate.dblquad 로 식 06을 직접 적분
  4) 맵: 중심 = T_ref, 축대칭, 웨이퍼 밖 NaN
  5) 산란 성분(γ > 0) 포함 식도 dblquad 와 비교

여기서 계산하는 표·그림은 논문 식 그대로(γ = 0)다. 실측으로 보정한 γ 는 calibrate_pvd.py 참고.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import integrate

import pvd_model as P

HERE = Path(__file__).parent


def check():
    worst = 0.0
    for a in (2.5, 3.0, 5.0, 7.0):
        for H in (5.0, 9.0, 10.0):
            # 1) 감쇠 없음, 중심
            e = abs(P.F(0.0, a, H, np.inf) / (np.pi * a * a / (H * H + a * a)) - 1)
            worst = max(worst, e)
            for lam in (0.8, 1.6, 5.0):
                # 2) 중심, 1차원 적분
                ref = 2 * np.pi * H * H * integrate.quad(lambda L: np.exp(-L / lam) / L**3, H, np.hypot(H, a),
                                                         epsabs=0, epsrel=1e-13)[0]
                worst = max(worst, abs(P.F(0.0, a, H, lam) / ref - 1))
                # 3) 중심 밖, 2차원 적분
                for r in (1.0, 2.54, 5.0):
                    ref = integrate.dblquad(
                        lambda phi, q: H * H * q / (H * H + r * r + q * q - 2 * r * q * np.cos(phi)) ** 2
                        * np.exp(-np.sqrt(H * H + r * r + q * q - 2 * r * q * np.cos(phi)) / lam),
                        0, a, 0, 2 * np.pi, epsabs=0, epsrel=1e-11)[0]
                    worst = max(worst, abs(P.F(r, a, H, lam) / ref - 1))
    # 5) 산란 성분 포함
    for a, H, lam, g, r in ((2.5, 9.0, 1.6, 1.0, 2.54), (5.0, 10.0, 1.6, 0.3, 1.0), (7.0, 5.0, 5.0, 1.0, 5.0)):
        def f(phi, q):
            L = np.sqrt(H * H + r * r + q * q - 2 * r * q * np.cos(phi))
            S = np.exp(-L / lam)
            return (H * H / L**4 * S + g * H / (2 * L**3) * (1 - S)) * q
        ref = integrate.dblquad(f, 0, a, 0, 2 * np.pi, epsabs=0, epsrel=1e-11)[0]
        worst = max(worst, abs(P.F(r, a, H, lam, gamma=g) / ref - 1))
    print(f"적분 검산: 최대 상대 오차 {worst:.2e}")
    assert worst < 1e-8, worst

    # 4) 맵
    X, Y = P.wafer_grid(n=161)
    m = P.thickness_map(X, Y, 200.0, 5.0, 10.0, 1.6)
    assert abs(m[80, 80] - 200.0) < 1e-9
    assert np.allclose(m, m.T, equal_nan=True) and np.allclose(m, m[::-1], equal_nan=True)
    assert np.isnan(m[0, 0]) and np.isfinite(m[80, 0])
    exact = P.thickness(np.hypot(X, Y)[np.isfinite(m)], 200.0, 5.0, 10.0, 1.6)
    err = np.max(np.abs(m[np.isfinite(m)] / exact - 1))
    print(f"맵 보간 검산: 최대 상대 오차 {err:.2e}")
    assert err < 1e-6
    assert abs(P.t_relative(0.0, 5.0, 10.0, 1.6) - 1) < 1e-14
    print("검산 통과")


def table():
    H, lam = P.FIG2A["H"], P.FIG2A["lam"]
    r = np.linspace(0, P.RW_2INCH, 11)
    rows = [("r_cm",) + tuple(f"a={a:g}cm" for a in P.FIG2A["a_list"])]
    cols = [P.t_relative(r, a, H, lam) for a in P.FIG2A["a_list"]]
    for i, ri in enumerate(r):
        rows.append((f"{ri:.3f}",) + tuple(f"{c[i]:.5f}" for c in cols))
    (HERE / "pvd_fig2a.csv").write_text("\n".join(",".join(x) for x in rows) + "\n", encoding="utf-8")
    print(f"\nFigure 2(a) 조건 · 논문 식(γ = 0): H = {H:g} cm, λ = {lam:g} cm, 2-inch 웨이퍼 (R_w = {P.RW_2INCH} cm)")
    print(f"{'a [cm]':>7} {'T_rel(R_w)':>11} {'η 직경18점 [%]':>15} {'η 면적가중 [%]':>15}")
    for a in P.FIG2A["a_list"]:
        s = P.summary(1.0, a, H, lam)
        print(f"{a:7.1f} {s['edge_ratio']:11.5f} {s['eta_18pt']:15.3f} {s['eta_area']:15.3f}")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, lam = P.FIG2A["H"], P.FIG2A["lam"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for a in P.FIG2A["a_list"]:
        r, t = P.radial_profile(a, H, lam)
        ax[0].plot(r, t, label=f"a = {a:g} cm")
    ax[0].set(xlabel="r [cm]", ylabel="T_relative = F(r)/F(0)",
              title=f"Flat target (d = 0), H = {H:g} cm, λ = {lam:g} cm")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    X, Y = P.wafer_grid(n=241)
    m = P.thickness_map(X, Y, 200.0, 5.0, H, lam)
    im = ax[1].pcolormesh(X, Y, m, shading="auto", cmap="viridis")
    ax[1].set_aspect("equal")
    ax[1].set(xlabel="x [cm]", ylabel="y [cm]", title="T(x, y) [nm], a = 5 cm, T_ref = 200 nm")
    fig.colorbar(im, ax=ax[1])
    fig.tight_layout()
    fig.savefig(HERE / "pvd_fig2a.png", dpi=130)
    print("\n저장: pvd_fig2a.png, pvd_fig2a.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    check()
    table()
    if not args.no_plot:
        plot()
