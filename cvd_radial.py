"""
CVD 반경 방향 두께 프로파일 모델 (Streamlit 의존성 없음 → cvd_sim.py와 fit_cvd_radial.py가 함께 사용)

참고 논문
  B.-P. Le, W.-J. Lin, J.-C. Chen et al., "Numerical and experimental investigation of the effect of side
  injectors on the deposition rate near the wafer edge during atmospheric pressure chemical vapor
  deposition", Mater. Sci. Semicond. Process. 172, 108085 (2024)
  - 반응기: Epi 300 Centura (APCVD, 1 atm), 300 mm Si 웨이퍼, 서셉터 32 rpm 회전, SiHCl3(TCS)/H2
  - Fig. 2: 사이드 인젝터 없이 반경별 '정규화 증착률' (중심 = 1) — 실험 vs COMSOL

사용 방식
  웨이퍼 중심 두께  t_c        = cvd_model 물리 모델 (웨이퍼 중심 좌표에서 계산)
  반경 r 에서 두께   t(r)       = t_c · N(r)
  정규화 증착률      N(r)       = DR(r) / DR(0)      ← 이 파일

N(r) 모델 — 논문 식 (22)–(26) (Habuka 표면 반응) 을 그대로 씀
  k_ad = 2.72e6 · exp(−1.72e5 / RT),   k_r = 5.63e3 · exp(−1.8e5 / RT)
  k_overall = k_r·k_ad / (k_ad·[TCS] + k_r·[H2])
  DR ∝ k_overall · [TCS] · [H2]

  웨이퍼 위 비균일의 원인 두 가지를 분리해서 넣는다
  1) 램프 가열 불균일 → 웨이퍼 온도 분포 T(r) = T_c + ΔT(r)
     논문: "중심 온도는 실측, 전체 온도 분포는 증착률 실측에서 역산"(Jeon et al. 방식),
           "중심부 편차는 주로 웨이퍼 온도 분포 불균일 때문"
     ΔT(r) = Σ a_n · [R_2n⁰(ρ) − R_2n⁰(0)],  ρ = r / R_w   (짝수 Zernike 반경 다항식)
       · 회전 평균이라 축대칭, ρ = 0에서 dT/dr = 0 (대칭 조건)이 자동으로 만족됨
       · ΔT(0) = 0 → 중심 온도 = T_c (실측 기준)
  2) 웨이퍼 가장자리 TCS 농도 저하 (pre-heater/서셉터 쪽으로 TCS가 먼저 소모·희석)
     [TCS](r) = [TCS]_c · φ(r),  φ(r) = 1 − A·exp(−(R_w − r)/λ)   (중심값으로 정규화)
     논문: 사이드 인젝터 없으면 가장자리에서 DR이 급락, TCS를 가장자리로 보내면 회복됨
  · 기체 농도는 이상기체 C = x·p/(R·T) → 온도가 높은 곳은 농도가 약간 낮음 (자동 반영)

단위: r [mm], T [K], ΔT [K]
"""
import math

import numpy as np

R_GAS = 8.314            # J/mol/K
P_ATM = 101325.0         # Pa
M_TCS, M_H2 = 135.45, 2.016   # g/mol
R_WAFER_MM = 150.0       # 논문 웨이퍼 반경 (300 mm)

# 논문 Table 1 · 2 (메인 Inlet, 사이드 인젝터 없음)
PAPER_COND = {
    "T_c": 1100.0 + 273.15,   # 웨이퍼 중심 온도 [K] (웨이퍼 1093–1103 °C, 서셉터 1100 °C)
    "w_TCS": 0.55,            # TCS 질량 분율
    "p": P_ATM,               # 1 atm
}


# ---------------------------------------------------------------------------
# Habuka 표면 반응 (논문 식 22–26)
# ---------------------------------------------------------------------------
def k_ad(T):
    return 2.72e6 * np.exp(-1.72e5 / (R_GAS * np.asarray(T, float)))


def k_r(T):
    return 5.63e3 * np.exp(-1.8e5 / (R_GAS * np.asarray(T, float)))


def mole_fracs(w_TCS):
    """질량 분율 → 몰 분율 (TCS, H2)"""
    n_t, n_h = w_TCS / M_TCS, (1 - w_TCS) / M_H2
    return n_t / (n_t + n_h), n_h / (n_t + n_h)


def dr_kinetic(T, phi=1.0, w_TCS=PAPER_COND["w_TCS"], p=PAPER_COND["p"]):
    """논문 식 (26)의 DR ∝ k_overall·[TCS]·[H2]  (상대값이라 M_Si/ρ_Si는 뺌)"""
    T = np.asarray(T, float)
    x_t, x_h = mole_fracs(w_TCS)
    c_tot = p / (R_GAS * T)                       # mol/m³
    c_t, c_h = x_t * c_tot * phi, x_h * c_tot
    kad, kr = k_ad(T), k_r(T)
    k_ov = kr * kad / (kad * c_t + kr * c_h)
    return k_ov * c_t * c_h


def arrhenius_sens(T_c=PAPER_COND["T_c"], w_TCS=PAPER_COND["w_TCS"]):
    """d ln DR / dT [1/K] at T_c  (온도 1 K당 증착률 변화율)"""
    h = 0.01
    return float((np.log(dr_kinetic(T_c + h, w_TCS=w_TCS)) - np.log(dr_kinetic(T_c - h, w_TCS=w_TCS))) / (2 * h))


# ---------------------------------------------------------------------------
# 반경 기저: 짝수 Zernike 반경 다항식 R_2n⁰(ρ)
# ---------------------------------------------------------------------------
def zernike_radial(n2, rho):
    """R_{n2}^0(ρ), n2 짝수"""
    rho = np.asarray(rho, float)
    n = n2 // 2
    out = np.zeros_like(rho)
    for k in range(n + 1):
        coef = (-1) ** k * math.factorial(n2 - k) / (math.factorial(k) * math.factorial(n - k) ** 2)
        out = out + coef * rho ** (n2 - 2 * k)
    return out


def basis(r_mm, n_terms, R=R_WAFER_MM):
    """[R_2⁰−R_2⁰(0), R_4⁰−R_4⁰(0), …]  shape (len(r), n_terms)"""
    rho = np.asarray(r_mm, float) / R
    cols = [zernike_radial(2 * (i + 1), rho) - zernike_radial(2 * (i + 1), np.array(0.0)) for i in range(n_terms)]
    return np.stack(cols, -1) if cols else np.zeros((len(rho), 0))


# ---------------------------------------------------------------------------
# 모델
# ---------------------------------------------------------------------------
def delta_T(r_mm, p):
    """웨이퍼 온도 편차 ΔT(r) [K] (중심 = 0)"""
    a = np.asarray(p["a"], float)
    return basis(r_mm, len(a), p.get("R", R_WAFER_MM)) @ a


def phi_edge(r_mm, p):
    """가장자리 TCS 농도 비율 φ(r) (중심 = 1)"""
    R = p.get("R", R_WAFER_MM)
    A, lam = p["A_edge"], p["lam_edge"]
    f = lambda r: 1.0 - A * np.exp(-(R - np.asarray(r, float)) / lam)
    return f(r_mm) / f(0.0)


def norm_dr(r_mm, p, T_c=None, w_TCS=None):
    """정규화 증착률 N(r) = DR(r)/DR(0)

    T_c, w_TCS를 바꾸면 같은 ΔT(r)·φ(r)에서 다른 공정 조건의 프로파일을 예측
    (기본: 피팅에 쓴 논문 조건)
    """
    T_c = p.get("T_c", PAPER_COND["T_c"]) if T_c is None else T_c
    w = p.get("w_TCS", PAPER_COND["w_TCS"]) if w_TCS is None else w_TCS
    T = T_c + delta_T(r_mm, p)
    return dr_kinetic(T, phi_edge(r_mm, p), w) / dr_kinetic(T_c, 1.0, w)


def invert_temperature(norm_dr_obs, T_c=PAPER_COND["T_c"], w_TCS=PAPER_COND["w_TCS"], phi=1.0):
    """측정 N(r) → 국소 온도 편차 ΔT [K] (Jeon et al. 방식: 점별 역산, 뉴턴법)"""
    target = np.asarray(norm_dr_obs, float) * dr_kinetic(T_c, 1.0, w_TCS)
    dT = np.zeros_like(target)
    for _ in range(30):
        f = dr_kinetic(T_c + dT, phi, w_TCS)
        g = (dr_kinetic(T_c + dT + 0.01, phi, w_TCS) - dr_kinetic(T_c + dT - 0.01, phi, w_TCS)) / 0.02
        step = (f - target) / g
        dT = dT - step
        if np.max(np.abs(step)) < 1e-9:
            break
    return dT


# ---------------------------------------------------------------------------
# 앱용: 4-inch 웨이퍼(cvd_model 기하)에 반경 정규화로 매핑
# ---------------------------------------------------------------------------
R_DATA_MAX_MM = 149.0   # 실측(Fig. 2) 마지막 점. 그 바깥(엣지 익스클루전)은 외삽이라 이 값으로 자름



def r_paper(x, y, wc, wr_cm, p):
    """앱 웨이퍼 좌표 (x, y) [cm] → 논문 웨이퍼 반경 [mm] (반경을 웨이퍼 반경으로 정규화해서 옮김)"""
    rho = np.hypot(np.asarray(x, float) - wc[0], np.asarray(y, float) - wc[1]) / wr_cm
    return np.clip(rho * p.get("R", R_WAFER_MM), 0, R_DATA_MAX_MM)


def thickness_map(t_center, x, y, wc, wr_cm, p):
    """t(x, y) = t_c · N(r),  r = 정규화 반경 × 150 mm"""
    return t_center * norm_dr(r_paper(x, y, wc, wr_cm, p), p)


def load_params(path):
    """fit_cvd_radial.py 결과(json) → 파라미터 dict (없으면 None)"""
    import json
    from pathlib import Path
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def uniformity(r_mm, n):
    """면적 가중(∝ r) 평균·1σ 불균일도[%]·(max−min)/(2·mean)[%]"""
    r_mm, n = np.asarray(r_mm, float), np.asarray(n, float)
    w = np.gradient(r_mm) * np.maximum(r_mm, 1e-9)
    mean = float(np.sum(w * n) / np.sum(w))
    sd = float(np.sqrt(np.sum(w * (n - mean) ** 2) / np.sum(w)))
    return mean, sd / mean * 100, (n.max() - n.min()) / (2 * mean) * 100
