"""
논문 실측 데이터로 모델 상수 보정

데이터: Lee et al., "Machine learning-guided optimization of atomic layer deposition process",
        Supplementary Fig. 1(a) — HfOx 두께 9-point 맵 (증착온도 7 × 전구체온도 5 = 35 조건, 315점)

사용법:
    python calibrate.py                         # 기본: 100 cycle, dose 0.1 s 가정
    python calibrate.py --cycles 120 --dose 0.2 # 논문 본문의 실제 공정 조건을 알면 넣기
    python calibrate.py --cv                    # 조건 하나씩 빼는 교차검증(LOCO)까지

주의: s₀·P_ref·dose 는 곱으로만 결과에 들어가고, r은 사이클 수와 짝을 이뤄 두께 스케일을 정함.
      그래서 cycles/dose 가정을 바꾸면 s₀, P_ref, r 값은 달라지지만 예측 두께는 같다.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import model as M

P_NAMES = ["sref", "Ea", "dHvap", "Ldep", "Ton", "Tscale", "r"]
LOG_P = {"sref"}  # log10 공간에서 피팅
BOUNDS = {
    "sref": (1e-9, 1.0), "Ea": (0.0, 200.0), "dHvap": (40.0, 150.0),
    "Ldep": (0.01, 1000.0), "Ton": (300.0, 600.0), "Tscale": (10.0, 5000.0), "r": (0.2, 1.5),
}


def to_vec(c):
    return np.array([np.log10(c[k]) if k in LOG_P else c[k] for k in P_NAMES])


PREF = 0.1  # main()에서 --pref로 바꿈


def to_dict(v):
    d = {k: (10 ** x if k in LOG_P else float(x)) for k, x in zip(P_NAMES, v)}
    d["Pref"] = PREF
    return d


LB = to_vec({k: b[0] for k, b in BOUNDS.items()})
UB = to_vec({k: b[1] for k, b in BOUNDS.items()})



def predict(c, df, cycles, dose, use_zernike=False):
    Td = df.T_dep_C.values + 273.15
    P0 = M.p_vap(df.T_prec_C.values + 273.15, c)
    return M.thickness(Td, P0, dose, cycles, df.x_cm.values, df.y_cm.values, c, use_zernike)


def fit(df, cycles, dose, x0=None, n_start=60, seed=0):
    y = df.Tox_nm.values
    res_fn = lambda v: predict(to_dict(v), df, cycles, dose) - y
    starts = [x0] if x0 is not None else []
    rng = np.random.default_rng(seed)
    starts += [rng.uniform(LB, UB) for _ in range(n_start)]
    best = None
    for s in starts:
        try:
            r = least_squares(res_fn, s, bounds=(LB, UB), x_scale="jac")
        except (ValueError, FloatingPointError):
            continue
        if best is None or r.cost < best.cost:
            best = r
    return to_dict(best.x), best


def zfeat(df):
    X = df.x_cm.values / M.ZR
    Y = (df.y_cm.values - M.WC[1]) / M.ZR
    return np.stack([Y, X, X**2 + Y**2], 1)


def fit_zernike(c, df, cycles, dose, mask):
    """전구체가 충분한 조건(mask)의 상대 잔차로 저차 Zernike 계수 추정"""
    pr = predict(c, df, cycles, dose)
    A = zfeat(df)[mask] * pr[mask, None]
    b = (df.Tox_nm.values - pr)[mask]
    a = np.linalg.lstsq(A, b, rcond=None)[0]
    return {"z_tilt_y": float(a[0]), "z_tilt_x": float(a[1]), "z_defocus": float(a[2])}


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hfox_tox_suppfig1.csv")
    ap.add_argument("--cycles", type=int, default=100, help="사이클 수 (논문: 100)")
    ap.add_argument("--dose", type=float, default=0.2, help="TDMA-Hf pulse 시간 [s] (논문: 0.2 s)")
    ap.add_argument("--pref", type=float, default=0.1,
                    help="75 °C 전구체 분압 [Torr]. s₀와 곱으로만 들어가 데이터로 정할 수 없으므로 고정")
    ap.add_argument("--sat-prec", type=float, default=60, help="이 전구체 온도 이상만 Zernike 추정에 사용 [°C]")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--out", default="calibrated.json")
    ap.add_argument("--plot", default="calibration.png")
    a = ap.parse_args()
    global PREF
    PREF = a.pref

    df = pd.read_csv(a.data)
    y = df.Tox_nm.values
    c, r = fit(df, a.cycles, a.dose)
    sat = df.T_prec_C.values >= a.sat_prec
    z = fit_zernike(c, df, a.cycles, a.dose, sat)
    cz = {**c, **z}
    p0 = predict(c, df, a.cycles, a.dose)
    p1 = predict(cz, df, a.cycles, a.dose, use_zernike=True)

    print("== 보정된 상수 ==")
    for k in P_NAMES + ["Pref"]:
        print(f"  {k:7s} = {c[k]:.4g}")
    print(f"  Zernike: tilt_y={z['z_tilt_y']:+.4f}, tilt_x={z['z_tilt_x']:+.4f}, defocus={z['z_defocus']:+.4f}")
    print("== 오차 (nm) ==")
    r2 = 1 - np.sum((p0 - y) ** 2) / np.sum((y - y.mean()) ** 2)
    print(f"  물리 모델      RMSE {rmse(p0 - y):.3f}  (R² {r2:.3f})  | 전구체 충분 {rmse((p0 - y)[sat]):.3f}"
          f" · 부족(55°C) {rmse((p0 - y)[~sat]):.3f}")
    print(f"  + Zernike      RMSE {rmse(p1 - y):.3f}  | 전구체 충분 {rmse((p1 - y)[sat]):.3f}"
          f" · 부족(55°C) {rmse((p1 - y)[~sat]):.3f}")

    cv = None
    if a.cv:
        e0, e1 = [], []
        x0 = to_vec(c)
        for (td, tp), g in df.groupby(["T_dep_C", "T_prec_C"]):
            te = (df.T_dep_C == td) & (df.T_prec_C == tp)
            tr = df[~te]
            ci, _ = fit(tr, a.cycles, a.dose, x0=x0, n_start=0)
            zi = fit_zernike(ci, tr, a.cycles, a.dose, tr.T_prec_C.values >= a.sat_prec)
            gt = df[te]
            e0 += list(predict(ci, gt, a.cycles, a.dose) - gt.Tox_nm.values)
            e1 += list(predict({**ci, **zi}, gt, a.cycles, a.dose, True) - gt.Tox_nm.values)
        cv = {"loco_rmse_phys": rmse(e0), "loco_rmse_zernike": rmse(e1)}
        print(f"== 교차검증 (조건 1개씩 제외, 35-fold) ==\n  물리 {cv['loco_rmse_phys']:.3f} nm · "
              f"+Zernike {cv['loco_rmse_zernike']:.3f} nm")

    out = {
        "source": "Lee et al., Supplementary Fig. 1(a), HfOx thickness 9-point maps",
        "assumed": {"cycles": a.cycles, "dose_s": a.dose, "Pref_torr_at_75C": a.pref},
        "consts": cz,
        "metrics": {"rmse_phys": rmse(p0 - y), "rmse_zernike": rmse(p1 - y), "r2_phys": float(r2), **(cv or {})},
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"저장: {a.out}")

    if a.plot:
        make_plot(df, p0, p1, a.plot)
        print(f"저장: {a.plot}")


def make_plot(df, p0, p1, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    y = df.Tox_nm.values
    sat = df.T_prec_C.values >= 60
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    for a_, p, t in [(ax[0], p0, "Physics model"), (ax[1], p1, "Physics + low-order Zernike")]:
        a_.scatter(y[sat], p[sat], s=12, c="#1f77b4", label="T_prec ≥ 60 °C")
        a_.scatter(y[~sat], p[~sat], s=16, c="#d62728", label="T_prec = 55 °C (starved)")
        a_.plot([0, 18], [0, 18], "k--", lw=1)
        a_.set(xlim=(0, 18), ylim=(0, 18), xlabel="Measured Tox [nm]", ylabel="Predicted Tox [nm]",
               title=f"{t}  RMSE {rmse(p - y):.2f} nm")
        a_.legend(loc="upper left", fontsize=9)
    # 웨이퍼 평균 두께: 증착온도별
    g = df.assign(pred=p0).groupby(["T_dep_C", "T_prec_C"])[["Tox_nm", "pred"]].mean().reset_index()
    for i, tp in enumerate(sorted(g.T_prec_C.unique())):
        s = g[g.T_prec_C == tp]
        col = plt.cm.viridis(i / 4)
        ax[2].plot(s.T_dep_C, s.Tox_nm, "o", color=col, label=f"{tp} °C meas.")
        ax[2].plot(s.T_dep_C, s.pred, "-", color=col)
    ax[2].set(xlabel="Deposition temperature [°C]", ylabel="Wafer-mean Tox [nm]",
              title="Mean thickness (dots = data, lines = model)")
    ax[2].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
