"""
논문 Fig. 2 (정규화 증착률, 사이드 인젝터 없음)로 반경 프로파일 모델(cvd_radial.py) 최적화
→ cvd_radial.json, cvd_radial_fit.png

사용법:
    python fit_cvd_radial.py                 # 기본: Zernike 9항 (교차검증으로 고른 값)
    python fit_cvd_radial.py --select        # 3~14항을 전부 피팅해서 교차검증 최소인 항 수를 고름 (몇 분 걸림)
    python fit_cvd_radial.py --n-terms 6

데이터: data/cvd_fig2_norm_dr.csv  (Fig. 2를 축 눈금 픽셀로 보정해서 디지타이즈)
    r_mm               반경 [mm]  (0–140 mm는 1 mm, 140–149 mm는 0.25 mm 간격)
    norm_dr_exp        실험 (검은 파선)
    norm_dr_sim_paper  논문 COMSOL 결과 (빨간 점선, 비교용 · 점선 사이 빈 곳은 비움)

최적화
  변수: ΔT(r)의 Zernike 계수 a₁…a_N, 가장자리 TCS 저하 A, 감쇠 길이 λ (0.1–5 mm)
  목적: Σ (N_model(r) − N_exp(r))²   (scipy least_squares, Trust Region Reflective, 다중 시작)
  항 수 선택: 5 mm 구간을 하나씩 통째로 빼고 맞춘 뒤 빈 구간을 예측하는 블록 교차검증
             (촘촘한 점을 하나씩 빼면 옆 점이 거의 같은 값이라 과적합을 못 잡음)
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import cvd_radial as C

LAM_MAX = 5.0   # 가장자리 효과 감쇠 길이 상한 [mm] — 논문 Fig. 2에서 급락 구간이 ~3 mm


class Fitter:
    def __init__(self, r, y, n_max=16):
        self.r, self.y = np.asarray(r, float), np.asarray(y, float)
        self.B = C.basis(self.r, n_max)          # 기저는 한 번만 계산
        self.Tc, self.w = C.PAPER_COND["T_c"], C.PAPER_COND["w_TCS"]
        self.D0 = C.dr_kinetic(self.Tc, 1.0, self.w)

    def model(self, v, N, idx):
        a, A, lam = v[:N], v[N], 10 ** v[N + 1]
        f = lambda x: 1 - A * np.exp(-(C.R_WAFER_MM - x) / lam)
        phi = f(self.r[idx]) / f(0.0)
        return C.dr_kinetic(self.Tc + self.B[idx, :N] @ a, phi, self.w) / self.D0

    def fit(self, N, idx=None, starts=None):
        idx = np.arange(len(self.r)) if idx is None else idx
        starts = starts or [(A, l) for A in (0.05, 0.2, 0.5) for l in (0.3, 0.8, 2.0, 4.0)]
        lb = np.r_[-20 * np.ones(N), 0.0, -1.0]
        ub = np.r_[20 * np.ones(N), 0.95, np.log10(LAM_MAX)]
        best = None
        for A0, l0 in starts:
            x0 = np.r_[np.zeros(N), A0, np.log10(l0)]
            s = least_squares(lambda v: self.model(v, N, idx) - self.y[idx], x0, bounds=(lb, ub), x_scale="jac")
            if best is None or s.cost < best.cost:
                best = s
        return best

    def block_cv(self, N, v_full, width=5.0):
        blocks = np.floor(self.r / width).astype(int)
        err = np.empty_like(self.y)
        for b in np.unique(blocks):
            m = blocks == b
            s = self.fit(N, np.where(~m)[0], starts=[(v_full[N], 10 ** v_full[N + 1])])
            err[m] = self.model(s.x, N, np.where(m)[0]) - self.y[m]
        return err

    def params(self, v, N):
        return {"a": [float(x) for x in v[:N]], "A_edge": float(v[N]), "lam_edge": float(10 ** v[N + 1]),
                "T_c": self.Tc, "w_TCS": self.w, "R": C.R_WAFER_MM}


def metrics(pred, obs):
    e = np.asarray(pred) - np.asarray(obs)
    rel = e / np.asarray(obs) * 100
    return {"rmse_pct": float(np.sqrt(np.mean(rel ** 2))), "mape_pct": float(np.mean(np.abs(rel))),
            "max_abs_pct": float(np.max(np.abs(rel))), "n": int(len(e))}


def main():
    ap = argparse.ArgumentParser(description="CVD 반경 프로파일 최적화 (논문 Fig. 2)")
    ap.add_argument("--data", default="data/cvd_fig2_norm_dr.csv")
    ap.add_argument("--n-terms", type=int, default=9, help="ΔT(r) Zernike 항 수 (R₂ … R₂ₙ)")
    ap.add_argument("--select", action="store_true", help="3~14항 교차검증으로 항 수 선택")
    ap.add_argument("--out", default="cvd_radial.json")
    ap.add_argument("--plot", default="cvd_radial_fit.png")
    a = ap.parse_args()

    d = pd.read_csv(a.data)
    r, y = d.r_mm.values, d.norm_dr_exp.values
    F = Fitter(r, y)

    scan = []
    if a.select:
        print(" N   RMSE[%]   최대[%]   블록CV RMSE[%]   A_edge   λ[mm]   ΔT 범위[K]")
        for N in range(3, 15):
            s = F.fit(N)
            m = metrics(F.model(s.x, N, np.arange(len(r))), y)
            cv = metrics(F.block_cv(N, s.x) + y, y)
            dT = F.B[:, :N] @ s.x[:N]
            scan.append({"N": N, **{k: m[k] for k in ("rmse_pct", "max_abs_pct")}, "cv_rmse_pct": cv["rmse_pct"]})
            print(f"{N:2d}   {m['rmse_pct']:.4f}    {m['max_abs_pct']:.3f}     {cv['rmse_pct']:.4f}"
                  f"          {s.x[N]:.3f}    {10 ** s.x[N + 1]:.2f}   {dT.min():+.2f} ~ {dT.max():+.2f}", flush=True)
        a.n_terms = min(scan, key=lambda z: z["cv_rmse_pct"])["N"]
        print(f"→ 교차검증 최소: N = {a.n_terms}")

    N = a.n_terms
    s = F.fit(N)
    p = F.params(s.x, N)
    pred = C.norm_dr(r, p)
    cv_pred = F.block_cv(N, s.x) + y
    fit_m, cv_m = metrics(pred, y), metrics(cv_pred, y)

    # 비교 기준: 논문 COMSOL 결과 vs 실험 (같은 반경에서, 빨간 점선이 있는 곳만)
    ms = d.norm_dr_sim_paper.notna().values
    paper_m = metrics(d.norm_dr_sim_paper.values[ms], y[ms])
    ours_same = metrics(pred[ms], y[ms])

    dT_pt = C.invert_temperature(y)              # 점별 역산 (가장자리 농도 효과까지 온도로 본 값, 참고)
    dT_fit = C.delta_T(r, p)
    u_exp, u_fit = C.uniformity(r, y), C.uniformity(r, pred)

    print(f"== 최적화 결과 (Zernike {N}항 + 가장자리 TCS 항) ==")
    print(f"  a = {np.round(p['a'], 4).tolist()}  [K]")
    print(f"  가장자리: A = {p['A_edge']:.4f}, λ = {p['lam_edge']:.3f} mm  "
          f"→ 웨이퍼 끝(149 mm) TCS 농도 {C.phi_edge(149.0, p) * 100:.1f} % (중심 대비)")
    print(f"  추정 온도 편차 ΔT(r): {dT_fit.min():+.2f} ~ {dT_fit.max():+.2f} K "
          f"(논문 웨이퍼 온도 1093–1103 °C 범위 안)")
    print(f"  증착률 온도 민감도 {C.arrhenius_sens() * 100:.2f} %/K @ {C.PAPER_COND['T_c'] - 273.15:.0f} °C")
    print("== 오차율 (상대 오차 = (모델 − 실험)/실험) ==")
    print(f"  피팅      RMSE {fit_m['rmse_pct']:.4f} % · 평균 {fit_m['mape_pct']:.4f} % · 최대 {fit_m['max_abs_pct']:.3f} %")
    print(f"  블록 CV   RMSE {cv_m['rmse_pct']:.4f} % · 평균 {cv_m['mape_pct']:.4f} % · 최대 {cv_m['max_abs_pct']:.3f} %")
    print(f"  논문 COMSOL (같은 {paper_m['n']}점) RMSE {paper_m['rmse_pct']:.4f} % · 최대 {paper_m['max_abs_pct']:.3f} %"
          f"  vs 이 모델 RMSE {ours_same['rmse_pct']:.4f} % · 최대 {ours_same['max_abs_pct']:.3f} %")
    print(f"  균일도 (면적 가중 1σ)  실험 {u_exp[1]:.3f} % · 모델 {u_fit[1]:.3f} %")

    out = {
        "source": "Le et al., Mater. Sci. Semicond. Process. 172, 108085 (2024), Fig. 2 (no side injector)",
        "data": a.data,
        "model": "N(r) = DR(T_c+ΔT(r), φ(r)) / DR(T_c, 1),  Habuka kinetics (paper eq. 22-26)",
        "params": p,
        "n_terms": N,
        "metrics": {"fit": fit_m, "block_cv": cv_m, "paper_comsol": paper_m, "ours_on_paper_points": ours_same,
                    "uniformity_1sigma_pct": {"exp": u_exp[1], "model": u_fit[1]}},
        "scan": scan,
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"저장: {a.out}")
    if a.plot:
        make_plot(d, r, y, pred, cv_pred, p, dT_pt, a.plot, fit_m, cv_m, paper_m, N)
        print(f"저장: {a.plot}")


def make_plot(d, r, y, pred, cv_pred, p, dT_pt, path, fit_m, cv_m, paper_m, N):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, MUTED, A1, A2 = "#1f2328", "#868d98", "#0d7f74", "#eb6834"
    rr = np.linspace(0, 149.5, 1200)
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), gridspec_kw=dict(height_ratios=[1.35, 1]))

    a0 = ax[0, 0]
    a0.plot(r, y, "o", ms=3, color=INK, label="Experiment (Fig. 2, digitized)")
    ms = d.norm_dr_sim_paper.notna()
    a0.plot(r[ms], d.norm_dr_sim_paper[ms], ".", ms=3, color=A2, alpha=0.7, label="Paper COMSOL")
    a0.plot(rr, C.norm_dr(rr, p), "-", lw=2, color=A1, label=f"This model (Zernike {N} + edge TCS)")
    a0.set(xlabel="Wafer radius [mm]", ylabel="Normalized DR", xlim=(0, 150),
           title=f"Fit  RMSE {fit_m['rmse_pct']:.3f} %  ·  block-CV {cv_m['rmse_pct']:.3f} %  ·  "
                 f"paper COMSOL {paper_m['rmse_pct']:.3f} %")
    a0.legend(fontsize=9, frameon=False)
    a0.grid(alpha=0.25)

    a1 = ax[0, 1]
    a1.plot(r, y, "o", ms=4, color=INK)
    a1.plot(r[ms], d.norm_dr_sim_paper[ms], ".", ms=4, color=A2, alpha=0.7)
    a1.plot(rr, C.norm_dr(rr, p), "-", lw=2, color=A1)
    a1.set(xlim=(125, 150), xlabel="Wafer radius [mm]", title="Edge region")
    a1.grid(alpha=0.25)

    a2 = ax[1, 0]
    a2.axhline(0, color=MUTED, lw=1)
    a2.plot(r, (pred - y) / y * 100, "-", color=A1, lw=1.6, label="This model (fit)")
    a2.plot(r, (cv_pred - y) / y * 100, ":", color=A1, lw=1.4, label="This model (block CV)")
    a2.plot(r[ms], (d.norm_dr_sim_paper[ms] - y[ms]) / y[ms] * 100, ".", color=A2, ms=3, label="Paper COMSOL")
    a2.set(xlabel="Wafer radius [mm]", ylabel="Relative error [%]", xlim=(0, 150), title="Error vs experiment")
    a2.legend(fontsize=9, frameon=False)
    a2.grid(alpha=0.25)

    a3 = ax[1, 1]
    a3.plot(r, dT_pt, ".", ms=3, color=MUTED, label="Point inversion (all → T)")
    a3.plot(rr, C.delta_T(rr, p), "-", color=A2, lw=2, label="Fitted ΔT(r)")
    a3.set(xlabel="Wafer radius [mm]", ylabel="ΔT from center [K]", xlim=(0, 150),
           title="Inferred wafer temperature  /  edge TCS ratio φ(r)")
    a3b = a3.twinx()
    a3b.plot(rr, C.phi_edge(rr, p), "-", color=A1, lw=1.6, label="φ(r) = [TCS]/[TCS]_c")
    a3b.set_ylabel("φ(r)")
    h1, l1 = a3.get_legend_handles_labels()
    h2, l2 = a3b.get_legend_handles_labels()
    a3.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False, loc="lower left")
    a3.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
