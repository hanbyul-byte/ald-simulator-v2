"""
ALD 성장 모델 (Streamlit 의존성 없음 → app.py와 calibrate.py가 함께 사용)

TDMA-Hf · Cross-flow 반응기 · Langmuir 포화 + 흐름 방향 전구체 소모(Bohart–Adams)

변경 요약 (v2)
- 전구체 고갈: 기존 P(d) = P0·exp(-d/L_dep) (표면 상태와 무관한 고정 감쇠)
  → Bohart–Adams 해: 상류 표면이 포화되면 전구체가 하류로 더 멀리 감.
    · τ = s(T)·Φ(P0,T)·t / (N_max·f(T))     (Inlet에서의 Langmuir 노출량, 기존과 동일)
    · ξ = d / L_dep · s(T)/s₀ · √(300/T)     (빈 표면이 전구체를 소모하는 정도)
    · θ = (e^τ − 1) / (e^τ + e^ξ − 1)
    d → 0 이면 θ = 1 − e^(−τ) 로 기존 Langmuir 식과 같아지고,
    공급이 부족하면 '포화 전선'이 웨이퍼 중간에 멈춰 Inlet 쪽만 두꺼워짐.
- 전구체 온도 → 분압: Clausius–Clapeyron  P0 = P_ref·exp(−ΔH/R·(1/T_p − 1/T_ref))
- (선택) 장비 고유 패턴: 저차 Zernike 곱셈 보정  t·(1 + a·Y + b·X + c·(X²+Y²))
"""
import math

import numpy as np

NA = 6.02214076e23
kB = 1.380649e-23
R = 8.314
TORR = 133.322
M_FILM = 210.49   # HfO2 g/mol
RHO = 9.68        # g/cm3
M_PREC = 354.79   # TDMA-Hf g/mol
C_PACK = 0.906
T_PREC_REF = 348.15  # 75 °C, 전구체 분압 기준 온도

# 기하 (cm, Gas inlet = 원점)
PTS = [(0, 10.5), (-1.5, 9), (1.5, 9), (-3, 7.5), (0, 7.5), (3, 7.5), (-1.5, 6), (1.5, 6), (0, 4.5)]
WC = (0.0, 7.5)
WR = 4.5
ZR = 3.0  # Zernike 정규화 반경 (9-point가 차지하는 반경)


def gpc_max0(r):
    """반경 r[nm] → (GPC_max,0 [Å/cycle], N_max [sites/nm²])"""
    n = C_PACK / (math.pi * r * r)
    gpc_cm = n * 1e14 * M_FILM / (RHO * NA)
    return gpc_cm * 1e8, n


def flux(p_pa, T):
    """충돌 플럭스 [분자/nm²/s]"""
    return p_pa / np.sqrt(2 * math.pi * (M_PREC * 1e-3 / NA) * kB * T) * 1e-18


def stick(T, c):
    return np.clip(c["sref"] * np.exp(-c["Ea"] * 1e3 / R * (1 / T - 1 / 300)), 0, 1)


def f_T(T, c):
    T = np.asarray(T, dtype=float)
    return np.where(T <= c["Ton"], 1.0, np.exp(-(T - c["Ton"]) / c["Tscale"]))


def p_vap(Tp_K, c):
    """전구체(캐니스터) 온도 [K] → Inlet 분압 [Torr]"""
    return c["Pref"] * np.exp(-c["dHvap"] * 1e3 / R * (1 / np.asarray(Tp_K, float) - 1 / T_PREC_REF))


def tau_xi(T, P0_torr, t, d, c):
    g_gpc, g_n = gpc_max0(c["r"])
    f = f_T(T, c)
    s = stick(T, c)
    tau = s * flux(P0_torr * TORR, T) * t / (g_n * f)
    xi = np.asarray(d, float) / c["Ldep"] * (s / c["sref"]) * np.sqrt(300 / np.asarray(T, float))
    return tau, xi


def _ba(tau, xi):
    """Bohart–Adams: (θ, 전구체 잔류 비율 C/C0). 오버플로 방지 형태."""
    tau, xi = np.broadcast_arrays(np.asarray(tau, float), np.asarray(xi, float))
    m = np.maximum(tau, xi)
    den = np.exp(tau - m) + np.exp(xi - m) - np.exp(-m)
    theta = (np.exp(tau - m) - np.exp(-m)) / den
    frac = np.exp(tau - m) / den
    return theta, frac


def coverage(T, P0_torr, t, d, c):
    tau, xi = tau_xi(T, P0_torr, t, d, c)
    return _ba(tau, xi)[0]


def p_local(T, P0_torr, t, d, c):
    """dose 종료 시점의 국소 전구체 분압 [Torr]"""
    tau, xi = tau_xi(T, P0_torr, t, d, c)
    return P0_torr * _ba(tau, xi)[1]


def gpc(T, P0_torr, t, d, c):
    """국소 GPC [Å/cycle]"""
    g_gpc, _ = gpc_max0(c["r"])
    return g_gpc * f_T(T, c) * coverage(T, P0_torr, t, d, c)


def zernike_factor(x, y, c):
    X = np.asarray(x, float) / ZR
    Y = (np.asarray(y, float) - WC[1]) / ZR
    return 1 + c.get("z_tilt_y", 0.0) * Y + c.get("z_tilt_x", 0.0) * X + c.get("z_defocus", 0.0) * (X**2 + Y**2)


def thickness(T, P0_torr, t, n, x, y, c, use_zernike=False):
    """두께 [nm] at (x, y) [cm]"""
    d = np.hypot(x, y)
    th = gpc(T, P0_torr, t, d, c) * n / 10
    if use_zernike:
        th = th * zernike_factor(x, y, c)
    return th
