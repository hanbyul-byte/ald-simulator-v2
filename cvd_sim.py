"""
CVD 시뮬레이터 페이지 (app.py에서 실행됨)
회전 서셉터 · 두께 t(r) = t_중심 · N(r)
  - t_중심: cvd_model.py (평판 경계층 + 직렬 저항(물질 전달 ↔ 표면 반응) + 전구체 고갈)
  - N(r): cvd_radial.py (논문 Fig. 2로 최적화한 반경 프로파일)

- 모델 식은 cvd_model.py (calibrate_cvd.py와 같이 씀)
- st.set_page_config는 app.py에서 이미 호출하므로 여기서 다시 부르지 않는다.
- ALD 페이지와 session_state 키가 겹치지 않도록 CVD 쪽 키는 "cvd_" 접두어를 쓴다.
"""
import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import cvd_model as M
import cvd_radial as RAD
from nav import back_button

HERE = Path(__file__).parent
RADIAL = RAD.load_params(HERE / "cvd_radial.json")   # fit_cvd_radial.py 결과 (논문 Fig. 2 반경 프로파일)
RADIAL_DATA_PATH = HERE / "data" / "cvd_fig2_norm_dr.csv"
CALIB_PATH = HERE / "cvd_calibrated.json"
DATA_PATH = HERE / "data" / "cvd_data.csv"
CALIB = json.loads(CALIB_PATH.read_text(encoding="utf-8")) if CALIB_PATH.exists() else None

ACCENT = "#0d7f74"
S2 = "#eb6834"
REF = "#868d98"
THK_RAMP = ["#d7191c", "#ff8c00", "#ffff00", "#22c55e", "#00e5ff", "#1e40ff", "#0000c8"]
COLORSCALE = [[i / (len(THK_RAMP) - 1), c] for i, c in enumerate(THK_RAMP)]
N_DIAM = 18   # 두께 평가 점 수: 웨이퍼 직경 방향 −R_w ~ +R_w 등간격 (PVD 페이지와 같은 방식)

# ---------------------------------------------------------------------------
# 상수 정의
# ---------------------------------------------------------------------------
CONSTS = [
    dict(g="표면 반응", id="k0", label="빈도 인자 k₀", unit="cm/s", min=1e-3, max=1e20, fmt="{:.3e}",
         hint="k_s(T) = k₀·exp(−Eₐ/RT). 1e7처럼 입력 가능"),
    dict(g="표면 반응", id="Ea", label="활성화 에너지 Eₐ", unit="kJ/mol", min=1.0, max=400.0, fmt="{:.1f}",
         hint="반응 율속 구간(저온)에서 아레니우스 기울기"),
    dict(g="기체 물성", id="nu0", label="동점성 계수 ν₀ (1 atm, 300 K)", unit="cm²/s", min=1e-3, max=10.0,
         fmt="{:.3g}", hint="N₂ ≈ 0.157. 압력에 반비례, 온도^α_ν 로 환산"),
    dict(g="기체 물성", id="D0", label="확산 계수 D₀ (1 atm, 300 K)", unit="cm²/s", min=1e-3, max=10.0,
         fmt="{:.3g}", hint="반응 기체의 확산 계수. 압력에 반비례, 온도^α_D 로 환산"),
    dict(g="기체 물성", id="a_nu", label="ν 온도 지수 α_ν", unit="", min=0.0, max=2.5, fmt="{:.2f}",
         hint="ν ∝ T^α_ν. 0이면 온도 보정 없음"),
    dict(g="기체 물성", id="a_D", label="D 온도 지수 α_D", unit="", min=0.0, max=2.5, fmt="{:.2f}",
         hint="D ∝ T^α_D. 0이면 온도 보정 없음"),
    dict(g="기하 · 막", id="x0", label="입구 오프셋 x₀", unit="cm", min=1e-3, max=50.0, fmt="{:.3g}",
         hint="입구에서 경계층이 시작되는 점까지의 거리 (x = 0에서 δ = 0 방지)"),
    dict(g="기하 · 막", id="H", label="가스 통로 유효 높이 H", unit="cm", min=0.05, max=50.0, fmt="{:.3g}",
         hint="작을수록 같은 소모량에서 전구체 농도가 빨리 떨어짐"),
    dict(g="기하 · 막", id="M", label="막 몰 질량 M", unit="g/mol", min=1.0, max=1000.0, fmt="{:.2f}",
         hint="기본 HfO₂ 210.49"),
    dict(g="기하 · 막", id="rho", label="막 밀도 ρ_film", unit="g/cm³", min=0.1, max=30.0, fmt="{:.3g}",
         hint="기본 HfO₂ 9.68"),
]

PROC0 = M.DEFAULT_PROCESS
CTRL_DEFAULTS = {
    "cvd_T_C": int(round(PROC0["T"] - 273.15)),
    "cvd_logP": round(math.log10(PROC0["P"]), 2),
    "cvd_logp": round(math.log10(PROC0["p_prec"]), 2),
    "cvd_u": float(PROC0["u"]),
    "cvd_t": float(PROC0["t_min"]),
}
PRESETS = [
    ("기본 (상압)", {}),
    ("반응 율속 · 저온", {"cvd_T_C": 450}),
    ("물질 전달 율속 · 고온", {"cvd_T_C": 750, "cvd_u": 40.0}),
]


def sig(v):
    if v >= 100:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.2f}"
    if v >= 0.1:
        return f"{v:.3f}"
    return f"{v:.2g}"


st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', ui-monospace, Menlo, monospace; font-size: 1.45rem; }
    .model-box { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.8rem; line-height: 1.8; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 기본 상수 = 보정값(cvd_calibrated.json) 우선, 없으면 cvd_model.DEFAULT_CONSTS
DEFAULTS = dict(M.DEFAULT_CONSTS)
if CALIB:
    DEFAULTS.update({k: float(f"{v:.4g}") for k, v in CALIB["consts"].items() if k in DEFAULTS})

ss = st.session_state
ss.setdefault("cvd_stage", "setup")
ss.setdefault("cvd_consts", dict(DEFAULTS))
for k, v in CTRL_DEFAULTS.items():
    ss.setdefault(k, v)
ss.setdefault("cvd_preset", 0)


def apply_preset(i):
    for k, v in CTRL_DEFAULTS.items():
        ss[k] = v
    for k, v in PRESETS[i][1].items():
        ss[k] = v
    ss.cvd_preset = i


def clear_preset():
    ss.cvd_preset = None


if ss.cvd_stage == "setup":
    back_button()
st.title("CVD 웨이퍼 시뮬레이터" if ss.cvd_stage == "sim" else "CVD 시뮬레이터")
st.caption("회전 서셉터 · 중심 두께 = 경계층 물질 전달 + 아레니우스 표면 반응 (직렬 저항) · "
           "반경 분포 = 논문 보정 프로파일 N(r)")
if not RADIAL:
    st.error("cvd_radial.json이 없어요. `python fit_cvd_radial.py`를 먼저 실행하세요.")
    st.stop()

# ---------------------------------------------------------------------------
# 1단계: 상수 입력
# ---------------------------------------------------------------------------
if ss.cvd_stage == "setup":
    with st.container(border=True):
        st.subheader("상수 입력")
        if CALIB:
            m = CALIB["metrics"]
            st.caption(f"기본값 = 실측 {m['n']}점으로 피팅한 값 ({', '.join(CALIB['fit'])}) · RMSE {m['rmse']:.3g}"
                       + (f" · 교차검증 {m['loco_rmse']:.3g}" if "loco_rmse" in m else ""))
        else:
            st.caption("기본값 = 문헌 수준의 초깃값 (아직 실측 보정 전). "
                       "데이터를 `calibrate_cvd.py`로 피팅하면 `cvd_calibrated.json`이 기본값이 됩니다.")
        with st.form("cvd_setup"):
            raw = {}
            groups = list(dict.fromkeys(c["g"] for c in CONSTS))
            cols = st.columns(len(groups))
            for col, grp in zip(cols, groups):
                with col:
                    st.markdown(f"**{grp}**")
                    for c in [c for c in CONSTS if c["g"] == grp]:
                        unit = f" [{c['unit']}]" if c["unit"] else ""
                        ss.setdefault(f"cvd_in_{c['id']}", f"{ss.cvd_consts.get(c['id'], DEFAULTS[c['id']]):.6g}")
                        raw[c["id"]] = st.text_input(c["label"] + unit, key=f"cvd_in_{c['id']}",
                                                     help=f"{c['hint']} · 허용 {c['min']:g} ~ {c['max']:g}")
            b1, b2 = st.columns([1, 1])
            submitted = b1.form_submit_button("이 값으로 시뮬레이션 시작", type="primary")
            reset = b2.form_submit_button("기본값으로 되돌리기")

        if reset:
            ss.cvd_consts = dict(DEFAULTS)
            for c in CONSTS:
                ss.pop(f"cvd_in_{c['id']}", None)
            st.rerun()

        vals, errors = {}, []
        for c in CONSTS:
            s = raw[c["id"]].strip().replace(",", "")
            try:
                v = float(s)
                if not math.isfinite(v):
                    raise ValueError
            except ValueError:
                errors.append(f"**{c['label']}**: 숫자를 입력하세요. 예: 1e7")
                continue
            if v < c["min"] or v > c["max"]:
                errors.append(f"**{c['label']}**: {c['min']:g} 이상 {c['max']:g} 이하로 입력하세요.")
                continue
            vals[c["id"]] = v

        if {"k0", "Ea"} <= set(vals):
            cc = {**DEFAULTS, **vals}
            ks = [float(M.k_s(T + 273.15, cc)) for T in (400, 600, 800)]
            st.info("표면 반응 속도 k_s: " + " · ".join(f"{T} °C **{sig(k)}** cm/s" for T, k in zip((400, 600, 800), ks)))

        st.markdown(
            "<div class='model-box'>"
            "δ(x) = 5.0·√(ν(x+x₀)/u) &nbsp;→&nbsp; h_g = D_g/δ &nbsp;→&nbsp; "
            "k_eff = (1/h_g + 1/k_s)⁻¹, &nbsp;k_s = k₀·exp(−Eₐ/RT)<br>"
            "C_g(x) = C_g0·exp(−∫₀ˣ k_eff dx′ / (u·H)) &nbsp;→&nbsp; R = C_g·k_eff·M/ρ_film"
            "</div>", unsafe_allow_html=True)

        if submitted:
            if errors:
                for e in errors:
                    st.error(e)
            else:
                ss.cvd_consts = vals
                ss.cvd_stage = "sim"
                st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# 2단계: 사이드바 — 공정 조건
# ---------------------------------------------------------------------------
c = {**DEFAULTS, **ss.cvd_consts}

with st.sidebar:
    back_button(use_container_width=True)
    st.divider()
    st.subheader("공정 조건")
    st.slider("웨이퍼 온도 [°C]", 250, 1000, step=5, key="cvd_T_C", on_change=clear_preset,
              help="Hot-wall 가정: 기체 온도 = 웨이퍼 온도")
    st.caption(f"= {ss.cvd_T_C + 273.15:.0f} K")
    st.slider("챔버 압력 log₁₀(P / Torr)", -1.0, 2.88, step=0.01, key="cvd_logP", on_change=clear_preset,
              help="압력이 낮을수록 ν, D가 커짐 → 경계층 확산이 빨라짐")
    P_tot = 10 ** ss.cvd_logP
    st.caption(f"P = **{sig(P_tot)} Torr**")
    st.slider("전구체 분압 log₁₀(p / Torr)", -3.0, 1.5, step=0.01, key="cvd_logp", on_change=clear_preset,
              help="Inlet 농도 C_g0 = p / (R·T)")
    p_prec = 10 ** ss.cvd_logp
    st.caption(f"p_prec = **{sig(p_prec)} Torr**")
    if p_prec > P_tot:
        st.error("전구체 분압이 전체 압력보다 커요.")
    st.slider("유속 u [cm/s]", 0.5, 200.0, step=0.5, key="cvd_u", on_change=clear_preset)
    st.slider("증착 시간 [min]", 0.5, 60.0, step=0.5, key="cvd_t", on_change=clear_preset)
    st.button("공정 조건 초기화", on_click=apply_preset, args=(0,))

    st.divider()
    st.subheader("두께 분포")
    rm = RADIAL["metrics"]
    st.caption(f"t(r) = t_중심 · N(r) · 중심 두께는 위 물리 모델, N(r)은 논문 Fig. 2 "
               f"(300 mm, 32 rpm 회전)로 최적화 · 오차 RMSE {rm['fit']['rmse_pct']:.3f} % "
               f"(교차검증 {rm['block_cv']['rmse_pct']:.3f} %) · 반경은 웨이퍼 반경으로 정규화해서 옮김")

    st.divider()
    st.subheader("두께 맵 컬러 범위")
    st.checkbox("자동 (현재 웨이퍼 min–max)", value=True, key="cvd_cauto")
    if not ss.cvd_cauto:
        ss.setdefault("cvd_cmin", 0.0)
        ss.setdefault("cvd_cmax", 100.0)
        cc1, cc2 = st.columns(2)
        cc1.number_input("최소 [nm]", min_value=0.0, step=1.0, format="%.1f", key="cvd_cmin")
        cc2.number_input("최대 [nm]", min_value=0.01, step=1.0, format="%.1f", key="cvd_cmax")

    st.divider()
    st.subheader("고정 상수")
    st.table({
        "상수": [x["label"] for x in CONSTS],
        "값": [x["fmt"].format(c[x["id"]]) + (f" {x['unit']}" if x["unit"] else "") for x in CONSTS],
    })
    if st.button("상수 다시 입력"):
        ss.cvd_stage = "setup"
        st.rerun()

pcols = st.columns([1.2] + [1.3] * len(PRESETS) + [2.0])
pcols[0].markdown("<div style='padding-top:6px;opacity:.6;font-size:.8rem'>예시 조건</div>", unsafe_allow_html=True)
for i, (name, _) in enumerate(PRESETS):
    pcols[i + 1].button(name, key=f"cvd_preset_{i}", on_click=apply_preset, args=(i,),
                        type="primary" if ss.cvd_preset == i else "secondary", use_container_width=True)

# ---------------------------------------------------------------------------
# 계산
# ---------------------------------------------------------------------------
T = ss.cvd_T_C + 273.15
u = float(ss.cvd_u)
t_min = float(ss.cvd_t)


RP = RADIAL["params"]
# 웨이퍼 중심 두께 = 사용자 물리 모델 (cvd_model)로 웨이퍼 중심 좌표에서 계산
t_center = float(M.thickness(T, P_tot, p_prec, u, t_min, M.WC[0], M.WC[1], c))


def thk_xy(x, y):
    return RAD.thickness_map(t_center, x, y, M.WC, M.WR, RP)


# 평가 지점: 웨이퍼 중심을 지나는 직경 위 등간격 18점 (N(r)이 축대칭이라 방향은 상관없음)
pts_pos = np.linspace(-M.WR, M.WR, N_DIAM)                         # 웨이퍼 중심 기준 위치 [cm]
pts_xy = np.column_stack([M.WC[0] + pts_pos, np.full(N_DIAM, M.WC[1])])
pts_r = RAD.r_paper(pts_xy[:, 0], pts_xy[:, 1], M.WC, M.WR, RP)   # 논문 웨이퍼 반경 [mm]
pts_v = thk_xy(pts_xy[:, 0], pts_xy[:, 1])
mean = float(pts_v.mean())
sd = float(pts_v.std(ddof=1))
nu_s = sd / mean * 100 if mean > 0 else float("nan")
nu_r = (pts_v.max() - pts_v.min()) / (2 * mean) * 100 if mean > 0 else float("nan")

d_center = float(np.hypot(*M.WC))
reg = float(M.regime(d_center, T, P_tot, u, c))
delta_c = float(M.delta(d_center, T, P_tot, u, c))
nu_gas, D_gas = (float(v) for v in M.gas_props(T, P_tot, c))
Re = u * d_center / nu_gas

if reg < 0.1:
    regime_txt, regime_kind = f"표면 반응 율속 (k_s/h_g = {reg:.2g})", "success"
elif reg > 10:
    regime_txt, regime_kind = f"물질 전달 율속 (k_s/h_g = {reg:.3g})", "error"
else:
    regime_txt, regime_kind = f"전환 구간 (k_s/h_g = {reg:.2g})", "warning"

# ---------------------------------------------------------------------------
# KPI
# ---------------------------------------------------------------------------
k = st.columns(6)
k[0].metric("평균 두께", f"{sig(mean)} nm", help=f"직경 {N_DIAM}점 평균")
k[1].metric("불균일도 1σ", f"{nu_s:.1f} %", help=f"직경 방향 등간격 {N_DIAM}점의 σ / 평균 × 100")
k[2].metric("불균일도 (max−min)/2·mean", f"±{nu_r:.1f} %")
k[3].metric("평균 증착률", f"{sig(mean / t_min)} nm/min")
k[4].metric("k_s / h_g (웨이퍼 중심)", f"{reg:.3g}")
r_full = np.linspace(0, RAD.R_DATA_MAX_MM, 600)
nu_full = RAD.uniformity(r_full, RAD.norm_dr(r_full, RP))[1]
k[5].metric("중심 두께 (물리 모델)", f"{sig(t_center)} nm",
            help=f"웨이퍼 전체(면적 가중) 불균일도 1σ = {nu_full:.3f} %")

if delta_c > c["H"]:
    st.warning(f"웨이퍼 중심에서 경계층 δ = {delta_c:.2f} cm가 통로 높이 H = {c['H']:.2f} cm보다 두꺼워요. "
               "평판 경계층 식이 맞지 않는 영역(저압·저유속)이라 h_g를 과소평가할 수 있어요.")

if ss.cvd_cauto:
    pad = max((pts_v.max() - pts_v.min()) * 0.15, mean * 0.01, 1e-6)
    c_lo, c_hi = float(pts_v.min() - pad), float(pts_v.max() + pad)
else:
    c_lo, c_hi = float(ss.cvd_cmin), float(ss.cvd_cmax) if ss.cvd_cmax > ss.cvd_cmin else float(ss.cvd_cmin) + 1


# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------
def wafer_figure():
    N = 200
    xs = np.linspace(M.WC[0] - M.WR, M.WC[0] + M.WR, N)
    ys = np.linspace(M.WC[1] - M.WR, M.WC[1] + M.WR, N)
    X, Y = np.meshgrid(xs, ys)
    inside = np.hypot(X - M.WC[0], Y - M.WC[1]) <= M.WR
    Z = np.where(inside, thk_xy(X, Y), np.nan)
    Rmm = RAD.r_paper(X, Y, M.WC, M.WR, RP)
    custom = np.stack([Rmm, RAD.norm_dr(Rmm, RP), RAD.delta_T(Rmm, RP), RAD.phi_edge(Rmm, RP)], -1)
    hover = ("x %{x:.2f}, y %{y:.2f} cm · 논문 반경 %{customdata[0]:.1f} mm<br>"
             "두께 <b>%{z:.3f}</b> nm · N(r) <b>%{customdata[1]:.5f}</b><br>"
             "ΔT %{customdata[2]:+.2f} K · φ(TCS) %{customdata[3]:.3f}<extra></extra>")

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=Z, customdata=custom, colorscale=COLORSCALE, zmin=c_lo, zmax=c_hi,
        colorbar=dict(title="두께 [nm]", thickness=12, len=0.6),
        hovertemplate=hover,
    ))
    fig.add_shape(type="circle", x0=M.WC[0] - M.WR, y0=M.WC[1] - M.WR, x1=M.WC[0] + M.WR, y1=M.WC[1] + M.WR,
                  line=dict(color="#4a525e", width=1.5))
    fig.add_trace(go.Scatter(
        x=pts_xy[:, 0], y=pts_xy[:, 1], mode="markers",
        marker=dict(size=7, color="white", line=dict(color="#141820", width=1.2)),
        customdata=np.stack([np.arange(1, N_DIAM + 1), pts_r, pts_v], -1),
        hovertemplate="Point %{customdata[0]:.0f} · 논문 반경 %{customdata[1]:.1f} mm<br>"
                      "두께 <b>%{customdata[2]:.2f}</b> nm<extra></extra>",
        showlegend=False))
    # 점마다 두께 값 표시 — 18점이 촘촘해서 위·아래로 번갈아 놓는다
    for i, ((px, py), v) in enumerate(zip(pts_xy, pts_v)):
        fig.add_annotation(x=px, y=py, text=sig(v), showarrow=False, yshift=18 if i % 2 == 0 else -18,
                           font=dict(family="ui-monospace, Menlo, monospace", size=11, color="#141820"),
                           bgcolor="rgba(255,255,255,0.9)", borderpad=2)
    fig.update_xaxes(range=[-5.4, 5.4], visible=False)
    fig.update_yaxes(range=[M.WC[1] - 5.4, M.WC[1] + 5.4], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def arrhenius_figure():
    TT = np.linspace(250, 1000, 200) + 273.15
    inv = 1000 / TT
    fig = go.Figure()
    lim_rxn = M.c_g0(TT, p_prec) * M.k_s(TT, c) * c["M"] / c["rho"] * 1e7 * 60
    lim_mt = M.c_g0(TT, p_prec) * M.h_g(d_center, TT, P_tot, u, c) * c["M"] / c["rho"] * 1e7 * 60
    fig.add_trace(go.Scatter(x=inv, y=lim_rxn, name="반응 한계 (k_s만)", line=dict(color=REF, dash="dot", width=1.3),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=inv, y=lim_mt, name="물질 전달 한계 (h_g만, 중심)",
                             line=dict(color=REF, dash="dash", width=1.3), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=inv, y=M.rate(d_center, TT, P_tot, p_prec, u, c), name="웨이퍼 중심 증착률",
                             line=dict(color=ACCENT, width=2), customdata=TT - 273.15,
                             hovertemplate="%{customdata:.0f} °C · %{y:.3g} nm/min"))
    fig.add_trace(go.Scatter(x=[1000 / T], y=[t_center / t_min], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(size=11, color=ACCENT, line=dict(color="white", width=2))))
    top = float(np.nanmax(M.rate(d_center, TT, P_tot, p_prec, u, c)))
    fig.update_xaxes(title="1000 / T [1/K]")
    fig.update_yaxes(title="증착률 [nm/min]", type="log", range=[math.log10(top) - 4, math.log10(top) + 0.5])
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.18, x=0, font=dict(size=11)))
    return fig


def radial_figure():
    """N(r) (모델 · 실험 · 논문 COMSOL) + 추정 온도 편차 ΔT(r) (오른쪽 축)"""
    rr = np.linspace(0, RAD.R_DATA_MAX_MM, 600)
    fig = go.Figure()
    if RADIAL_DATA_PATH.exists():
        import pandas as pd
        dd = pd.read_csv(RADIAL_DATA_PATH)
        fig.add_trace(go.Scatter(x=dd.r_mm, y=dd.norm_dr_exp, mode="markers", name="실험 (Fig. 2)",
                                 marker=dict(size=4, color="#1f2328"), hovertemplate="%{y:.5f}"))
        ms = dd.norm_dr_sim_paper.notna()
        fig.add_trace(go.Scatter(x=dd.r_mm[ms], y=dd.norm_dr_sim_paper[ms], mode="markers", name="논문 COMSOL",
                                 marker=dict(size=3, color=S2, opacity=0.6), hovertemplate="%{y:.5f}"))
    fig.add_trace(go.Scatter(x=rr, y=RAD.norm_dr(rr, RP), name="최적화 모델 N(r)", line=dict(color=ACCENT, width=2.2),
                             hovertemplate="%{y:.5f}"))
    fig.add_trace(go.Scatter(x=rr, y=RAD.delta_T(rr, RP), name="ΔT(r) [K] (오른쪽 축)", yaxis="y2",
                             line=dict(color=REF, width=1.4, dash="dot"), hovertemplate="%{y:+.2f} K"))
    fig.update_xaxes(title="논문 웨이퍼 반경 r [mm] (300 mm 웨이퍼, 앱 웨이퍼 반경으로 정규화)", range=[0, 150])
    fig.update_yaxes(title="정규화 증착률 N(r)", tickformat=".3f")
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="ΔT [K]", showgrid=False),
                      height=340, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.2, x=0, font=dict(size=11)))
    return fig


# ---------------------------------------------------------------------------
# 레이아웃
# ---------------------------------------------------------------------------
left, right = st.columns([1.05, 1], gap="medium")
with left:
    with st.container(border=True):
        h1, h2 = st.columns([1, 1.3])
        h1.markdown(f"##### 두께 맵 · 직경 {N_DIAM}점")
        getattr(h2, regime_kind)(regime_txt)
        st.plotly_chart(wafer_figure(), use_container_width=True)
        st.caption(f"두께 [nm] · 컬러 범위 {sig(max(c_lo, 0))} ~ {sig(c_hi)} nm")
        with st.expander(f"직경 {N_DIAM}점 값 표"):
            st.dataframe({
                "Point": list(range(1, N_DIAM + 1)), "위치 x [cm]": np.round(pts_pos, 3),
                "논문 반경 [mm]": np.round(pts_r, 1),
                "N(r)": np.round(RAD.norm_dr(pts_r, RP), 5),
                "ΔT [K]": np.round(RAD.delta_T(pts_r, RP), 3),
                "증착률 [nm/min]": np.round(pts_v / t_min, 4),
                "두께 [nm]": np.round(pts_v, 3),
            }, hide_index=True, use_container_width=True)

with right:
    with st.container(border=True):
        st.markdown(f"##### 아레니우스 플롯 · P = {sig(P_tot)} Torr, u = {u:g} cm/s")
        st.plotly_chart(arrhenius_figure(), use_container_width=True)
        st.caption("저온(오른쪽): 기울기 = −Eₐ/R, 반응 율속 · 고온(왼쪽): 확산(물질 전달) 율속 · "
                   f"k_s = {sig(float(M.k_s(T, c)))} cm/s · h_g = {sig(float(M.h_g(d_center, T, P_tot, u, c)))} cm/s · "
                   f"ν = {sig(nu_gas)}, D_g = {sig(D_gas)} cm²/s · δ = {delta_c:.2f} cm · Re = {sig(Re)} · "
                   f"C_g0 = {float(M.c_g0(T, p_prec)):.3e} mol/cm³")
    with st.container(border=True):
        st.markdown("##### 반경 프로파일")
        st.plotly_chart(radial_figure(), use_container_width=True)
        st.caption(f"t(r) = t_중심 {sig(t_center)} nm × N(r) · 오차율 RMSE {rm['fit']['rmse_pct']:.3f} % "
                   f"(최대 {rm['fit']['max_abs_pct']:.2f} %, 블록 교차검증 {rm['block_cv']['rmse_pct']:.3f} %) · "
                   f"논문 COMSOL {rm['paper_comsol']['rmse_pct']:.3f} % · "
                   f"ΔT(r): 램프 가열 불균일, φ(r): 가장자리 TCS 저하 (λ = {RP['lam_edge']:.2f} mm)")

# ---------------------------------------------------------------------------
# 실측 데이터 비교 (data/cvd_data.csv 가 있을 때)
# ---------------------------------------------------------------------------
if DATA_PATH.exists():
    import pandas as pd

    from calibrate_cvd import load, predict

    @st.cache_data
    def load_data():
        a = CALIB["assumed"] if CALIB else M.DEFAULT_PROCESS
        return load(DATA_PATH, {k: a[k] for k in ("P", "p_prec", "u", "t_min")})

    try:
        d, kind = load_data()
    except SystemExit as e:   # 열 이름이 맞지 않을 때
        st.error(f"data/cvd_data.csv를 읽지 못했어요: {e}")
        st.stop()
    d = d.assign(pred=predict(c, d), T_C=(d["T"] - 273.15).round(1))
    err = d.pred - d.y_obs
    unit = "nm" if kind == "thickness_nm" else "nm/min"
    st.divider()
    st.subheader("실측 데이터와 비교")
    st.caption(f"`data/cvd_data.csv` {len(d)}점 · 현재 상수로 예측 · RMSE **{float(np.sqrt(np.mean(err ** 2))):.3g} {unit}**")
    q1, q2 = st.columns([1, 1.15], gap="medium")
    with q1:
        with st.container(border=True):
            pf = go.Figure()
            pf.add_trace(go.Scatter(x=d.y_obs, y=d.pred, mode="markers", marker=dict(size=7, color=d.T_C,
                                    colorscale="Plasma", colorbar=dict(title="°C", thickness=10)),
                                    customdata=d.T_C, hovertemplate="%{customdata} °C<br>실측 %{x:.3g} · 예측 %{y:.3g}<extra></extra>"))
            hi = float(max(d.y_obs.max(), d.pred.max()) * 1.05)
            pf.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color=REF, dash="dash", width=1),
                                    hoverinfo="skip", showlegend=False))
            pf.update_xaxes(title=f"실측 [{unit}]", range=[0, hi])
            pf.update_yaxes(title=f"예측 [{unit}]", range=[0, hi], scaleanchor="x")
            pf.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            st.plotly_chart(pf, use_container_width=True)
    with q2:
        with st.container(border=True):
            temps = sorted(d.T_C.unique())
            t_sel = st.selectbox("온도 [°C]", temps, index=len(temps) // 2, key="cvd_data_T")
            sel = d[d.T_C == t_sel].assign(dist=lambda z: np.hypot(z.x, z.y)).sort_values("dist")
            bf = go.Figure()
            lbl = [f"{v:.1f} cm" for v in sel.dist]
            bf.add_trace(go.Bar(x=lbl, y=sel.y_obs, name="실측", marker_color=REF))
            bf.add_trace(go.Bar(x=lbl, y=sel.pred, name="예측", marker_color=ACCENT))
            bf.update_xaxes(title="Inlet 거리")
            bf.update_yaxes(title=unit)
            bf.update_layout(height=300, barmode="group", margin=dict(l=10, r=10, t=10, b=10),
                             legend=dict(orientation="h", y=1.12, x=0))
            st.plotly_chart(bf, use_container_width=True)
