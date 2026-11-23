"""
CVD 성장 모델 (Streamlit 의존성 없음 → cvd_sim.py와 calibrate_cvd.py가 함께 사용)

Cross-flow 반응기 · 평판 경계층 + 직렬 저항(물질 전달 ↔ 표면 반응) + 흐름 방향 전구체 고갈

    1) 경계층 두께     δ(x)   = 5.0 · √( ν·(x + x₀) / u )
    2) 물질 전달 계수   h_g(x) = D_g / δ(x)
    3) 표면 반응 속도   k_s(T) = k₀ · exp(−Eₐ / (R·T))
       유효 속도 상수   k_eff  = [ 1/h_g + 1/k_s ]⁻¹          (직렬 저항)
    4) 전구체 고갈      C_g(x) = C_g0 · exp( −∫₀ˣ k_eff(x') dx' / (u·H) )
    5) 증착률          R(x,T) = C_g(x) · k_eff(x) · M / ρ_film

메모
- 4)에서 k_eff는 x에 따라 변하므로(h_g ∝ 1/√x) 지수 안을 k_eff·x 대신 적분으로 둔다.
  k_eff가 상수이면 원래 식 C_g0·exp(−k_eff·x/(u·H))와 같다.
  1/k_eff = b·s + c  (s = √(x+x₀), b = 5√(ν/u)/D_g, c = 1/k_s) 라서 적분은 해석해가 있다:
      ∫₀ˣ k_eff dx' = (2/b)·[ (s − s₀) − (c/b)·ln((b·s + c)/(b·s₀ + c)) ]
  반응 율속(c ≫ b·s)에서 두 항이 상쇄돼 수치 오차가 커지므로 상쇄 없는 형태로 계산한다 (_int_keff).
- 기체 물성은 1 atm · 300 K 기준값을 넣고 온도·압력으로 환산한다 (끄려면 지수 = 0).
      ν(T,P)   = ν₀ · (T/300)^α_ν · (760/P)
      D_g(T,P) = D₀ · (T/300)^α_D · (760/P)
- Hot-wall 가정: 기체 온도 = 웨이퍼 온도. Inlet 농도 C_g0 = p_prec / (R·T)  (이상기체)
- 평판 경계층 식은 δ < H 일 때 성립. δ가 H를 넘으면(저압·저유속) 통로가 완전 발달 흐름이라
  h_g를 과소평가하므로 화면에서 경고를 띄운다.
- 거리 x = Gas inlet(원점)에서의 거리. ALD 모델(model.py)과 같은 기하·9-point 좌표를 쓴다.

단위: cm, s, mol, K, Torr  →  두께 nm, 증착률 nm/min
"""
import numpy as np

from model import PTS, WC, WR  # 같은 반응기 기하 · 9-point 좌표  # noqa: F401

R_GAS = 8.314          # J/mol/K
R_CM3_TORR = 62363.6   # cm³·Torr/(mol·K)  → C [mol/cm³] = p[Torr] / (R·T)
T0 = 300.0             # 기체 물성 기준 온도 [K]
P_ATM_TORR = 760.0


# 기본 상수 (보정 전 초깃값 — 실측 데이터로 calibrate_cvd.py에서 피팅)
DEFAULT_CONSTS = {
    # 표면 반응
    "k0": 8.0e6,        # 빈도 인자 [cm/s]
    "Ea": 120.0,        # 활성화 에너지 [kJ/mol]
    # 기체 물성 (1 atm, 300 K 기준)
    "nu0": 0.15,        # 동점성 계수 [cm²/s]  (N₂ ≈ 0.157)
    "D0": 0.10,         # 확산 계수 [cm²/s]
    "a_nu": 1.7,        # ν 온도 지수
    "a_D": 1.75,        # D 온도 지수
    # 기하
    "x0": 1.0,          # 입구 → 웨이퍼 시작점 오프셋 [cm]
    "H": 4.0,           # 가스 통로 유효 높이 [cm]
    # 막
    "M": 210.49,        # 몰 질량 [g/mol]  (HfO₂)
    "rho": 9.68,        # 밀도 [g/cm³]     (HfO₂)
}


# 공정 조건 기본값
DEFAULT_PROCESS = {
    "T": 773.15,        # 웨이퍼(=기체) 온도 [K]
    "P": 760.0,         # 챔버 전체 압력 [Torr] (상압 CVD)
    "p_prec": 1.0,      # 전구체 분압 [Torr]
    "u": 20.0,          # 주 흐름 유속 [cm/s]
    "t_min": 5.0,       # 증착 시간 [min]
}


# ---------------------------------------------------------------------------
# 1) ~ 3) 국소 계수
# ---------------------------------------------------------------------------
def gas_props(T, P, c):
    """(ν, D_g) [cm²/s] at 온도 T [K], 압력 P [Torr]"""
    T = np.asarray(T, float)
    scale_p = P_ATM_TORR / np.asarray(P, float)
    nu = c["nu0"] * (T / T0) ** c["a_nu"] * scale_p
    D = c["D0"] * (T / T0) ** c["a_D"] * scale_p
    return nu, D


def delta(x, T, P, u, c):
    """경계층 두께 δ(x) [cm]"""
    nu, _ = gas_props(T, P, c)
    return 5.0 * np.sqrt(nu * (np.asarray(x, float) + c["x0"]) / u)


def h_g(x, T, P, u, c):
    """물질 전달 계수 h_g(x) [cm/s]"""
    _, D = gas_props(T, P, c)
    return D / delta(x, T, P, u, c)


def k_s(T, c):
    """표면 반응 속도 상수 k_s(T) [cm/s] (아레니우스)"""
    return c["k0"] * np.exp(-c["Ea"] * 1e3 / (R_GAS * np.asarray(T, float)))


def k_eff(x, T, P, u, c):
    """유효 반응 속도 상수 [cm/s] = (1/h_g + 1/k_s)⁻¹"""
    return 1.0 / (1.0 / h_g(x, T, P, u, c) + 1.0 / k_s(T, c))


# ---------------------------------------------------------------------------
# 4) 전구체 고갈
# ---------------------------------------------------------------------------
def _g(z):
    """g(z) = (z − ln(1+z)) / z²  (z ≥ 0), z → 0에서 1/2. 작은 z는 급수로 계산"""
    z = np.asarray(z, float)
    small = z < 1e-4
    zs = np.where(small, 1.0, z)
    direct = (zs - np.log1p(zs)) / zs**2
    series = 0.5 - z / 3 + z**2 / 4
    return np.where(small, series, direct)


def _int_keff(x, T, P, u, c):
    """∫₀ˣ k_eff dx' [cm²/s] — 해석해, 상쇄 없는 형태

    (s−s₀) − (c/b)·ln(1+z),  z = b(s−s₀)/(b·s₀+c)
      = (s−s₀)·b·s₀/(b·s₀+c) + (c/b)·z²·g(z)
    """
    nu, D = gas_props(T, P, c)
    b = 5.0 * np.sqrt(nu / u) / D                 # 1/h_g = b·√(x+x₀)
    cc = 1.0 / k_s(T, c)                          # 1/k_s
    s0 = np.sqrt(c["x0"])
    s = np.sqrt(np.asarray(x, float) + c["x0"])
    ds = s - s0
    den = b * s0 + cc
    z = b * ds / den
    inner = ds * b * s0 / den + (cc / b) * z**2 * _g(z)
    return (2.0 / b) * inner


def c_ratio(x, T, P, u, c):
    """C_g(x) / C_g0 (전구체 잔류 비율)"""
    return np.exp(-_int_keff(x, T, P, u, c) / (u * c["H"]))


def c_g0(T, p_prec):
    """Inlet 전구체 농도 [mol/cm³] (이상기체, hot-wall)"""
    return np.asarray(p_prec, float) / (R_CM3_TORR * np.asarray(T, float))


# ---------------------------------------------------------------------------
# 5) 증착률 · 두께
# ---------------------------------------------------------------------------
def rate(x, T, P, p_prec, u, c):
    """국소 증착률 [nm/min]"""
    flux = c_g0(T, p_prec) * c_ratio(x, T, P, u, c) * k_eff(x, T, P, u, c)   # mol/cm²/s
    return flux * c["M"] / c["rho"] * 1e7 * 60.0                               # cm/s → nm/min


def rate_no_depletion(x, T, P, p_prec, u, c):
    """고갈을 무시한 증착률 [nm/min] (C_g = C_g0) — 비교용"""
    flux = c_g0(T, p_prec) * k_eff(x, T, P, u, c)
    return flux * c["M"] / c["rho"] * 1e7 * 60.0


def regime(x, T, P, u, c):
    """k_s / h_g  (≪ 1: 표면 반응 율속, ≫ 1: 물질 전달 율속)"""
    return k_s(T, c) / h_g(x, T, P, u, c)


def thickness(T, P, p_prec, u, t_min, x, y, c):
    """두께 [nm] at 웨이퍼 좌표 (x, y) [cm] — Inlet(원점)에서의 거리로 계산"""
    d = np.hypot(np.asarray(x, float), np.asarray(y, float))
    return rate(d, T, P, p_prec, u, c) * t_min
