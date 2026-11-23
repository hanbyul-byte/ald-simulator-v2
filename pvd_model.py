"""
PVD 두께 분포 모델 (Streamlit 의존성 없음 → pvd_sim.py와 스크립트가 함께 사용)

평면 원형 타깃 마그네트론 스퍼터링 · 타깃 ∥ 웨이퍼 (동축 평행 원판) · 중심 기준 상대 두께

참고 논문
  Z. Cheng et al., "Realizing Ultrauniform Films at Wafer Scale through the Magnetron Sputtering Method",
  Cryst. Growth Des. 24, 339–346 (2024), DOI 10.1021/acs.cgd.3c01048
  - 논문의 곡면 타깃 일반식 Eq. (2)–(4), (7)–(9)에 평면 타깃 조건 d = 0 을 넣은 형태
    (인쇄된 Eq. (10)은 Eq. (4), (8), (9)를 직접 대입한 결과와 달라서 쓰지 않음)

모델 식 (번호 = "PVD 평면 타깃 물리식 정리" 문서의 절 번호)
  02) 이동 거리      L(r, q, φ) = √(H² + r² + q² − 2·r·q·cos φ)
  03) 방출각·입사각   cos α = cos β = H / L            (코사인 방출 × 입사각 투영)
  04) Ar 충돌 감쇠    S(L) = exp(−L / λ)               (충돌 없이 도달하는 비율)
  05) 국소 기여       t_PQ = m / (π·ρ₀) · [ H²/L⁴ · S(L) + γ · H/(2L³) · (1 − S(L)) ]
        직진 성분 H²/L⁴ · S          논문 Eq. (3) 그대로
        산란 성분 γ · H/(2L³) · (1−S)  05-b) 이 프로젝트에서 넣은 확장 (γ = 0 이면 논문 식)
  06) 타깃 전체 적분   F(r) = ∫₀ᵃ ∫₀²ᵖⁱ [ H²/L⁴ · S + γ · H/(2L³) · (1 − S) ] · q dφ dq
  07) 상대 두께       T_rel(r) = F(r) / F(0)
  08) 반경별 두께     T(r) = T_ref · T_rel(r)          (T_ref = 사용자가 넣는 중심 두께 [nm])
  09) 2D 웨이퍼맵     T(x, y) = T(√(x² + y²)),  x² + y² ≤ R_w²   (축대칭)
  10) 불균일도        η[%] = σ_T / T̄ × 100           (논문 Eq. (1), 모집단 표준편차, 직경 방향 N = 18점)

기호
  H 타깃–웨이퍼 거리, r 웨이퍼 반경 위치, q 타깃 위 방출 지점 반경, φ 두 위치 사이 방위각,
  a 타깃 반경, λ 스퍼터 원자의 유효 평균자유행로, R_w 웨이퍼 반경

메모
- 절대 증착량은 계산하지 않는다. m/(π·ρ₀)는 위치와 무관해서 F(r)/F(0)에서 상쇄된다.
  T(r)은 "중심이 T_ref일 때의 환산 두께"이지 실측 절대 두께 예측이 아니다.
- 05-b 산란 성분: Ar과 충돌한 원자(비율 1 − S)가 확산해서 웨이퍼에 닿는다고 보고, 한 점에서 나와 확산하다
  흡수 평면에 닿는 분포 H/(2πL³)를 쓴다 (직진 코사인 성분 H²/(πL⁴)와 같은 정규화: 무한 평면 적분 = 1).
  γ = 산란된 원자가 웨이퍼 쪽으로 도달하는 비율(0–1). 타깃·챔버 벽으로 되돌아가는 원자, 충돌 위치의 분포는
  따로 계산하지 않는 단순화다 (기체 수송 시뮬레이션이 아님).
  넣은 이유: 논문 조건(0.4 Pa, λ = 1.6 cm, H = 9 cm)에서는 S ≈ exp(−5.6) ≈ 0.4 %라 직진 성분만으로는
  실측(Fig. 2(c, d))보다 분포가 훨씬 가파르다. calibrate_pvd.py 참고.
- 보정값: γ = 1 (calibrate_pvd.py → pvd_calibrated.json, 논문 Fig. 2(c, d) 실측 36점). λ는 논문 값 유지.
- 타깃 침식은 면 전체에서 균일하다고 본다 (마그네트론 레이스트랙 미반영).
- 수치 적분: φ는 주기 함수라 등간격 합(주기 사다리꼴, 지수 수렴), q는 Gauss–Legendre.
  기본 분할(48 × 96)에서 H ≥ 2 cm이면 상대 오차 < 1e-10. H가 a·r에 비해 아주 작으면 n_q, n_phi를 늘린다.
- 검산용 해석해: λ → ∞ 이고 r = 0 이면 F(0) = π·a² / (H² + a²)

단위: cm (H, r, q, a, λ, R_w, x, y), 두께 nm, 압력 Pa
"""
import numpy as np

RW_2INCH = 2.54        # 2-inch 웨이퍼 반경 [cm]
RHO_MO = 10.2          # Mo 타깃 밀도 [g/cm³] (논문 표기) — 상대 두께에서는 상쇄돼 쓰이지 않음

# 평균자유행로 기준점 (논문: 0.4 Pa에서 1.6 cm)
LAM_REF = 1.6          # [cm]
P_REF = 0.4            # [Pa]

# 기본 조건 = 논문 Figure 2(a) 이론 곡선 조건
DEFAULT_PARAMS = {
    "a": 5.0,          # 타깃 반경 [cm]
    "H": 10.0,         # 타깃–웨이퍼 거리 [cm]
    "lam": LAM_REF,    # 평균자유행로 [cm]
    "gamma": 1.0,      # 산란 원자 도달 비율 (calibrate_pvd.py 보정값, 논문 식만 쓰려면 0)
    "Rw": RW_2INCH,    # 웨이퍼 반경 [cm]
    "T_ref": 200.0,    # 웨이퍼 중심 기준 두께 [nm]
}

# 논문 Figure 2(a) 재현용 입력 (이론 시뮬레이션 조건)
FIG2A = {"a_list": (3.0, 4.0, 5.0, 6.0, 7.0), "H": 10.0, "lam": LAM_REF}
# 논문 Figure 2(c, d) 실제 증착 조건 — 이론 곡선(FIG2A)과 섞어 쓰지 않는다
# T_mean, eta: 그림에 적힌 실측 평균 두께 [nm], 불균일도 [%]
FIG2CD_EXPERIMENT = {"a_list": (2.5, 3.0), "H": 9.0, "T_mean": (135.95, 169.11), "eta": (4.23, 2.85)}

N_Q, N_PHI = 48, 96    # 적분 분할 기본값 (q: Gauss–Legendre 노드 수, φ: 등간격 점 수)
N_POINTS_PAPER = 18    # 논문 불균일도 측정 점 수 (직경 방향)


# ---------------------------------------------------------------------------
# 02) ~ 05) 국소 항
# ---------------------------------------------------------------------------
def path_length(r, q, phi, H):
    """02) 방출 지점 Q(q, φ) → 평가 지점 P(r) 직선 거리 L [cm]"""
    r, q, phi = np.asarray(r, float), np.asarray(q, float), np.asarray(phi, float)
    return np.sqrt(H * H + r * r + q * q - 2.0 * r * q * np.cos(phi))


def cos_angle(L, H):
    """03) cos α = cos β = H / L  (α: 웨이퍼 법선 기준 입사각, β: 타깃 법선 기준 방출각)"""
    return H / np.asarray(L, float)


def survival(L, lam):
    """04) Ar 충돌 감쇠 S(L) = exp(−L/λ).  lam = inf 이면 감쇠 없음(1)"""
    return np.exp(-np.asarray(L, float) / lam)


def kernel(r, q, phi, H, lam, gamma=0.0):
    """05) 타깃 미소 면적의 국소 기여 (공통 계수 m/(π·ρ₀) 제외)

    직진 성분  H²/L⁴ · S(L)                  (논문 Eq. (3), S = exp(−L/λ))
    산란 성분  γ · H/(2·L³) · (1 − S(L))       (05-b, 이 프로젝트에서 넣은 확장)
    gamma = 0 이면 논문 식 그대로다.
    """
    L = path_length(r, q, phi, H)
    S = survival(L, lam)
    k = H * H / L**4 * S
    if gamma:
        k = k + gamma * H / (2.0 * L**3) * (1.0 - S)
    return k


def mean_free_path(P_pa, lam_ref=LAM_REF, P_ref=P_REF):
    """압력 → 평균자유행로 [cm].  λ = λ_ref · P_ref / P  (기체 운동론 λ ∝ 1/P, 온도 일정 가정)

    문서·논문에는 0.4 Pa → 1.6 cm 한 점만 있다. 다른 압력으로의 환산은 이 프로젝트에서 넣은 확장이다.
    """
    return lam_ref * P_ref / np.asarray(P_pa, float)


# ---------------------------------------------------------------------------
# 06) 타깃 전체 적분
# ---------------------------------------------------------------------------
def _quad_nodes(a, n_q, n_phi):
    """(q 노드, q 가중치, φ 노드, φ 가중치)"""
    x, w = np.polynomial.legendre.leggauss(n_q)      # [-1, 1]
    q = 0.5 * a * (x + 1.0)
    wq = 0.5 * a * w
    phi = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    return q, wq, phi, 2.0 * np.pi / n_phi


def F(r, a, H, lam, gamma=0.0, n_q=N_Q, n_phi=N_PHI):
    """06) 분포 모양 함수 F(r; a, H, λ) = ∫₀ᵃ∫₀²ᵖⁱ H²/L⁴ · exp(−L/λ) · q dφ dq

    r: 스칼라 또는 배열 [cm] → 같은 모양으로 반환 (무차원)
    """
    if a <= 0 or H <= 0 or lam <= 0 or gamma < 0:
        raise ValueError("a, H, lam은 양수, gamma는 0 이상이어야 합니다.")
    r_arr = np.asarray(r, float)
    flat = np.abs(r_arr).ravel()
    q, wq, phi, wphi = _quad_nodes(a, n_q, n_phi)
    out = np.empty(flat.shape)
    step = max(1, 2_000_000 // (n_q * n_phi))        # 메모리 상한용 청크
    for i in range(0, flat.size, step):
        rr = flat[i:i + step, None, None]
        k = kernel(rr, q[None, :, None], phi[None, None, :], H, lam, gamma)   # (r, q, φ)
        out[i:i + step] = (k.sum(axis=2) * wphi) @ (wq * q)
    return out.reshape(r_arr.shape) if r_arr.ndim else float(out[0])


# ---------------------------------------------------------------------------
# 07) ~ 09) 상대 두께 · 두께 · 웨이퍼맵
# ---------------------------------------------------------------------------
def t_relative(r, a, H, lam, **kw):
    """07) 중심 정규화 상대 두께 T_rel(r) = F(r)/F(0)  (T_rel(0) = 1)"""
    return F(r, a, H, lam, **kw) / F(0.0, a, H, lam, **kw)


def thickness(r, T_ref, a, H, lam, **kw):
    """08) 반경별 두께 [nm] = T_ref · T_rel(r)  (T_ref: 웨이퍼 중심 기준 두께 [nm])"""
    return T_ref * t_relative(r, a, H, lam, **kw)


def radial_profile(a, H, lam, Rw=RW_2INCH, n=201, T_ref=1.0, **kw):
    """(r [cm], T(r)) — 중심 0 ~ 가장자리 R_w 를 n점으로 (그래프·보간용)"""
    r = np.linspace(0.0, Rw, n)
    return r, thickness(r, T_ref, a, H, lam, **kw)


def thickness_map(x, y, T_ref, a, H, lam, Rw=RW_2INCH, wc=(0.0, 0.0), n_table=257, **kw):
    """09) 2D 웨이퍼맵 T(x, y) [nm] — 축대칭. 웨이퍼 밖(반경 > R_w)은 NaN

    x, y: 좌표 배열 [cm] (np.meshgrid 결과 등), wc: 웨이퍼 중심 좌표
    반경 표(n_table점)를 한 번 계산해 선형 보간한다 (보간 오차 ≲ 1e-6, 격자 점 수와 무관한 속도).
    """
    rho = np.hypot(np.asarray(x, float) - wc[0], np.asarray(y, float) - wc[1])
    r_tab, t_tab = radial_profile(a, H, lam, Rw=Rw, n=n_table, T_ref=T_ref, **kw)
    out = np.interp(rho, r_tab, t_tab)
    return np.where(rho <= Rw * (1 + 1e-12), out, np.nan)


def wafer_grid(Rw=RW_2INCH, n=161):
    """웨이퍼를 덮는 정사각 격자 (X, Y) [cm], 웨이퍼 중심 = 원점"""
    g = np.linspace(-Rw, Rw, n)
    return np.meshgrid(g, g)


# ---------------------------------------------------------------------------
# 10) 두께 불균일도
# ---------------------------------------------------------------------------
def diameter_points(Rw=RW_2INCH, n=N_POINTS_PAPER, r_max=None):
    """직경 방향 등간격 n점의 위치 [cm] (−r_max … +r_max, 기본 r_max = R_w)

    논문은 직경 방향 18점을 쓴다. 문서에 점의 실제 좌표는 없어서 등간격으로 둔다
    (논문의 측정 위치·가장자리 제외 폭을 알면 r_max나 직접 만든 좌표를 넣는다).
    """
    r_max = Rw if r_max is None else r_max
    return np.linspace(-r_max, r_max, n)


def nonuniformity(T):
    """10) 논문 Eq. (1): (평균 T̄, 표준편차 σ_T, η[%] = σ_T/T̄ × 100)  — σ는 1/N 모집단 표준편차"""
    T = np.asarray(T, float)
    T = T[np.isfinite(T)]
    mean = float(T.mean())
    sd = float(np.sqrt(np.mean((T - mean) ** 2)))
    return mean, sd, sd / mean * 100.0


def nonuniformity_diameter(a, H, lam, Rw=RW_2INCH, n=N_POINTS_PAPER, r_max=None, T_ref=1.0, **kw):
    """직경 방향 n점(기본 18점)에서의 (위치 [cm], 두께, T̄, σ_T, η[%]) — 논문과 같은 방식"""
    pos = diameter_points(Rw, n, r_max)
    T = thickness(np.abs(pos), T_ref, a, H, lam, **kw)
    return (pos, T) + nonuniformity(T)


def nonuniformity_area(a, H, lam, Rw=RW_2INCH, n=400, **kw):
    """웨이퍼 전체 면적 가중 (T̄_rel, σ, η[%]) — 직경 18점 값과 다르게 나오므로 논문 값과 직접 비교하지 않는다"""
    edges = np.linspace(0.0, Rw, n + 1)
    r = 0.5 * (edges[1:] + edges[:-1])
    w = r / r.sum()                                   # 고리 면적 ∝ r
    T = t_relative(r, a, H, lam, **kw)
    mean = float(np.sum(w * T))
    sd = float(np.sqrt(np.sum(w * (T - mean) ** 2)))
    return mean, sd, sd / mean * 100.0


def summary(T_ref, a, H, lam, Rw=RW_2INCH, **kw):
    """화면 표시용 요약 dict"""
    edge = float(t_relative(Rw, a, H, lam, **kw))
    _, _, mean, sd, eta = nonuniformity_diameter(a, H, lam, Rw=Rw, T_ref=T_ref, **kw)
    return {
        "T_center": float(T_ref),
        "T_edge": float(T_ref) * edge,
        "edge_ratio": edge,                            # T(R_w)/T(0)
        "mean_18pt": mean,
        "sigma_18pt": sd,
        "eta_18pt": eta,                               # 직경 18점 불균일도 [%]
        "eta_area": nonuniformity_area(a, H, lam, Rw=Rw, **kw)[2],
    }
