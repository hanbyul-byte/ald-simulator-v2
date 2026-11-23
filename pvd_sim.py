"""
PVD 시뮬레이터 페이지 (app.py에서 실행됨)

평면 원형 타깃 마그네트론 스퍼터링 · 중심 기준 상대 두께 분포
- 산란 원자 도달 비율 γ 기본값은 논문 Fig. 2(c, d) 실측으로 보정한 값 (calibrate_pvd.py → pvd_calibrated.json)
- 모델 식은 pvd_model.py (Streamlit 의존성 없음)
- st.set_page_config는 app.py에서 이미 호출하므로 여기서 다시 부르지 않는다.
- 다른 페이지와 session_state 키가 겹치지 않도록 PVD 쪽 키는 "pvd_" 접두어를 쓴다.
"""
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import pvd_model as P
from nav import back_button
from wafer_zoom import HINT as ZOOM_HINT, wafer_chart

ACCENT = "#0d7f74"
REF = "#868d98"
THK_RAMP = ["#d7191c", "#ff8c00", "#ffff00", "#22c55e", "#00e5ff", "#1e40ff", "#0000c8"]
COLORSCALE = [[i / (len(THK_RAMP) - 1), c] for i, c in enumerate(THK_RAMP)]

WAFERS = {"2-inch (R = 2.54 cm)": 2.54, "4-inch (R = 5.08 cm)": 5.08, "6-inch (R = 7.62 cm)": 7.62}
D0 = P.DEFAULT_PARAMS
CTRL_DEFAULTS = {
    "pvd_a": float(D0["a"]),
    "pvd_H": float(D0["H"]),
    "pvd_lam": float(D0["lam"]),
    "pvd_gamma": float(D0["gamma"]),
    "pvd_Tref": float(D0["T_ref"]),
    "pvd_wafer": list(WAFERS)[0],
}
# 논문 Figure 2(a) 이론 곡선 조건 (H = 10 cm, λ = 1.6 cm, 타깃 반경만 바꿈)
PRESETS = [(f"a = {a:g} cm", {"pvd_a": float(a)}) for a in P.FIG2A["a_list"]]
N_THEORY = len(PRESETS)

# 논문 Figure 2(c, d) 실측 조건 (H = 9 cm) + 실측 18점
HERE = Path(__file__).parent
EXP = P.FIG2CD_EXPERIMENT
try:
    _csv = np.genfromtxt(HERE / "data" / "pvd_fig2cd_thickness.csv", delimiter=",", skip_header=1)
    MEASURED = {N_THEORY: _csv[:, 1], N_THEORY + 1: _csv[:, 2]}
except OSError:
    MEASURED = {}
try:
    CAL = json.loads((HERE / "pvd_calibrated.json").read_text(encoding="utf-8"))
except (OSError, ValueError):
    CAL = None
for _a in EXP["a_list"]:
    PRESETS.append((f"a = {_a * 10:g} mm", {"pvd_a": float(_a), "pvd_H": float(EXP["H"])}))


def sig(v):
    if v >= 100:
        return f"{v:.0f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', ui-monospace, Menlo, monospace; font-size: 1.45rem; }
    .model-box { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.8rem; line-height: 1.8; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)

ss = st.session_state
for k, v in CTRL_DEFAULTS.items():
    ss.setdefault(k, v)
ss.setdefault("pvd_preset", 2)   # a = 5 cm


def apply_preset(i):
    for k, v in CTRL_DEFAULTS.items():
        ss[k] = v
    for k, v in PRESETS[i][1].items():
        ss[k] = v
    if i in MEASURED:
        # 중심 두께: 모델 18점이 실측 18점에 최소제곱으로 맞도록 (모델은 상대 분포만 계산)
        m = P.nonuniformity_diameter(ss.pvd_a, ss.pvd_H, ss.pvd_lam, gamma=ss.pvd_gamma)[1]
        ss.pvd_Tref = float(round((m @ MEASURED[i]) / (m @ m), 1))
    ss.pvd_preset = i


def clear_preset():
    ss.pvd_preset = None


st.title("PVD 웨이퍼 시뮬레이터")
st.caption("평면 원형 타깃 마그네트론 스퍼터링 (d = 0) · 타깃 ∥ 웨이퍼 · "
           "코사인 방출 + 거리 감소 + Ar 충돌(직진 감쇠 + 산란 원자 도달) → 중심 기준 상대 두께")

# ---------------------------------------------------------------------------
# 사이드바 — 조건
# ---------------------------------------------------------------------------
with st.sidebar:
    back_button(use_container_width=True)
    st.divider()
    st.subheader("기하")
    st.slider("타깃 반경 a [cm]", 1.0, 12.0, step=0.1, key="pvd_a", on_change=clear_preset)
    st.slider("타깃–웨이퍼 거리 H [cm]", 3.0, 20.0, step=0.1, key="pvd_H", on_change=clear_preset)
    st.selectbox("웨이퍼", list(WAFERS), key="pvd_wafer", on_change=clear_preset)

    st.divider()
    st.subheader("기체 산란")
    st.slider("평균자유행로 λ [cm]", 0.3, 10.0, step=0.1, key="pvd_lam", on_change=clear_preset,
              help="스퍼터 원자가 Ar과 충돌하지 않고 가는 평균 거리. 논문: 0.4 Pa에서 1.6 cm")
    st.caption(f"λ ∝ 1/P 로 환산하면 Ar 압력 ≈ **{P.P_REF * P.LAM_REF / ss.pvd_lam:.2f} Pa**")
    st.slider("산란 원자 도달 비율 γ", 0.0, 1.0, step=0.05, key="pvd_gamma", on_change=clear_preset,
              help="Ar과 충돌한 원자가 확산해서 웨이퍼에 닿는 비율. 0이면 논문 식(직진 성분만), "
                   "1이 논문 Fig. 2(c, d) 실측으로 보정한 값")

    st.divider()
    st.subheader("기준 두께")
    st.number_input("웨이퍼 중심 두께 T_ref [nm]", min_value=1.0, max_value=5000.0, step=10.0, format="%.1f",
                    key="pvd_Tref", on_change=clear_preset,
                    help="모델은 상대 분포만 계산해요. 중심 두께는 직접 넣는 기준값입니다.")
    st.button("조건 초기화", on_click=apply_preset, args=(2,))

    st.divider()
    st.subheader("두께 맵 컬러 범위")
    st.checkbox("자동 (현재 웨이퍼 min–max)", value=True, key="pvd_cauto")
    if not ss.pvd_cauto:
        ss.setdefault("pvd_cmin", 0.0)
        ss.setdefault("pvd_cmax", 200.0)
        cc1, cc2 = st.columns(2)
        cc1.number_input("최소 [nm]", min_value=0.0, step=1.0, format="%.1f", key="pvd_cmin")
        cc2.number_input("최대 [nm]", min_value=0.01, step=1.0, format="%.1f", key="pvd_cmax")

pcols = st.columns([1.5] + [1.0] * N_THEORY + [1.7] + [1.0] * (len(PRESETS) - N_THEORY))
pcols[0].markdown("<div style='padding-top:6px;opacity:.6;font-size:.8rem'>Fig. 2(a) 조건 (H = 10 cm)</div>",
                  unsafe_allow_html=True)
pcols[N_THEORY + 1].markdown("<div style='padding-top:6px;opacity:.6;font-size:.8rem;text-align:right'>"
                             "Fig. 2(c, d) 실측 (H = 9 cm)</div>", unsafe_allow_html=True)
for i, (name, _) in enumerate(PRESETS):
    pcols[i + 1 + (i >= N_THEORY)].button(name, key=f"pvd_preset_{i}", on_click=apply_preset, args=(i,),
                                          type="primary" if ss.pvd_preset == i else "secondary",
                                          use_container_width=True)

# ---------------------------------------------------------------------------
# 계산
# ---------------------------------------------------------------------------
a, H, lam, gamma = float(ss.pvd_a), float(ss.pvd_H), float(ss.pvd_lam), float(ss.pvd_gamma)
measured = MEASURED.get(ss.pvd_preset)     # 실측 조건 프리셋일 때만 실측 18점을 같이 그림
T_ref = float(ss.pvd_Tref)
Rw = WAFERS[ss.pvd_wafer]

s = P.summary(T_ref, a, H, lam, Rw=Rw, gamma=gamma)
pos, pts_v, mean18, sd18, eta18 = P.nonuniformity_diameter(a, H, lam, Rw=Rw, T_ref=T_ref, gamma=gamma)
r_line, t_line = P.radial_profile(a, H, lam, Rw=Rw, n=201, T_ref=1.0, gamma=gamma)

# 감쇠 항이 분포에 주는 영향 (λ → ∞ 와 비교)
edge_no_scatter = float(P.t_relative(Rw, a, H, np.inf))
edge_paper = float(P.t_relative(Rw, a, H, lam))        # 논문 식 (γ = 0)

k = st.columns(5)
k[0].metric("중심 두께 (기준값)", f"{sig(s['T_center'])} nm")
k[1].metric("가장자리 두께", f"{sig(s['T_edge'])} nm")
k[2].metric("가장자리 / 중심", f"{s['edge_ratio']:.4f}",
            help=f"Ar 충돌이 없다면(λ → ∞) {edge_no_scatter:.4f} · 논문 식(γ = 0)이면 {edge_paper:.4f}")
k[3].metric("불균일도 η (직경 18점)", f"{s['eta_18pt']:.2f} %",
            help="논문 Eq. (1): σ_T / 평균 × 100, 직경 방향 등간격 18점")
k[4].metric("불균일도 η (면적 가중)", f"{s['eta_area']:.2f} %",
            help="웨이퍼 전체 면적 가중. 직경 18점 값과 달라서 논문 값과 직접 비교하지 않아요.")

if ss.pvd_cauto:
    pad = max((s["T_center"] - s["T_edge"]) * 0.05, T_ref * 0.002)
    c_lo, c_hi = s["T_edge"] - pad, s["T_center"] + pad
else:
    c_lo = float(ss.pvd_cmin)
    c_hi = float(ss.pvd_cmax) if ss.pvd_cmax > ss.pvd_cmin else c_lo + 1.0


# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------
def wafer_figure():
    n = 201
    xs = np.linspace(-Rw, Rw, n)
    X, Y = np.meshgrid(xs, xs)
    Z = P.thickness_map(X, Y, T_ref, a, H, lam, Rw=Rw, gamma=gamma)
    custom = np.stack([np.hypot(X, Y), Z / T_ref], -1)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xs, y=xs, z=Z, customdata=custom, colorscale=COLORSCALE, zmin=c_lo, zmax=c_hi,
        colorbar=dict(title="두께 [nm]", thickness=12, len=0.6),
        hovertemplate="x %{x:.2f}, y %{y:.2f} cm · r %{customdata[0]:.2f} cm<br>"
                      "두께 <b>%{z:.2f}</b> nm · 상대 두께 <b>%{customdata[1]:.4f}</b><extra></extra>",
    ))
    fig.add_shape(type="circle", x0=-Rw, y0=-Rw, x1=Rw, y1=Rw, line=dict(color="#4a525e", width=1.5))
    fig.add_trace(go.Scatter(
        x=pos, y=np.zeros_like(pos), mode="markers",
        marker=dict(size=7, color="white", line=dict(color="#141820", width=1.2)),
        customdata=np.stack([np.arange(1, len(pos) + 1), pts_v], -1),
        hovertemplate="Point %{customdata[0]:.0f} · x %{x:.2f} cm<br>"
                      "두께 <b>%{customdata[1]:.2f}</b> nm<extra></extra>",
        showlegend=False))
    # 점마다 두께 값 표시 — 18점이 촘촘해서 위·아래로 번갈아 놓는다
    fmt = "{:.1f}" if T_ref < 1000 else "{:.0f}"
    for i, (px, v) in enumerate(zip(pos, pts_v)):
        fig.add_annotation(x=px, y=0, text=fmt.format(v), showarrow=False, yshift=18 if i % 2 == 0 else -18,
                           font=dict(family="ui-monospace, Menlo, monospace", size=11, color="#141820"),
                           bgcolor="rgba(255,255,255,0.9)", borderpad=2)
    lim = Rw * 1.08
    fig.update_xaxes(range=[-lim, lim], visible=False)
    fig.update_yaxes(range=[-lim, lim], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def radial_figure():
    fig = go.Figure()
    for ai in P.FIG2A["a_list"]:
        _, ti = P.radial_profile(ai, H, lam, Rw=Rw, n=101, T_ref=1.0, gamma=gamma)
        fig.add_trace(go.Scatter(x=np.linspace(0, Rw, 101), y=ti, name=f"a = {ai:g} cm",
                                 line=dict(color=REF, width=1.1, dash="dot"), hovertemplate="%{y:.4f}"))
    fig.add_trace(go.Scatter(x=r_line, y=t_line, name=f"현재 (a = {a:g} cm)", line=dict(color=ACCENT, width=2.4),
                             hovertemplate="%{y:.4f}"))
    fig.add_trace(go.Scatter(x=np.abs(pos), y=pts_v / T_ref, mode="markers", name="직경 18점",
                             marker=dict(size=7, color="white", line=dict(color=ACCENT, width=1.5)),
                             hovertemplate="%{y:.4f}"))
    if measured is not None:
        fig.add_trace(go.Scatter(x=np.abs(pos), y=measured / T_ref, mode="markers", name="논문 실측 18점",
                                 marker=dict(size=8, color="#5b63b7"), hovertemplate="%{y:.4f}"))
    fig.update_xaxes(title="웨이퍼 반경 r [cm]", range=[0, Rw])
    fig.update_yaxes(title="상대 두께 T(r) / T(0)", tickformat=".3f")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.22, x=0, font=dict(size=11)))
    return fig


def sweep_figure():
    """타깃 반경에 따른 불균일도 (현재 H, λ, 웨이퍼)"""
    aa = np.linspace(1.0, 12.0, 45)
    eta = [P.nonuniformity_diameter(x, H, lam, Rw=Rw, gamma=gamma)[4] for x in aa]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=aa, y=eta, line=dict(color=ACCENT, width=2), showlegend=False,
                             hovertemplate="a = %{x:.2f} cm · η = %{y:.2f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=[a], y=[eta18], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(size=11, color=ACCENT, line=dict(color="white", width=2))))
    fig.update_xaxes(title="타깃 반경 a [cm]")
    fig.update_yaxes(title="불균일도 η [%] (직경 18점)", rangemode="tozero")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10))
    return fig


# ---------------------------------------------------------------------------
# 레이아웃
# ---------------------------------------------------------------------------
left, right = st.columns([1.05, 1], gap="medium")
with left:
    with st.container(border=True):
        st.markdown("##### 두께 맵 · 직경 18점")
        wafer_chart(wafer_figure())
        st.caption(f"두께 [nm] · 컬러 범위 {sig(max(c_lo, 0))} ~ {sig(c_hi)} nm · "
                   f"축대칭 이론 환산맵 (실제 2D 측정 웨이퍼맵이 아님) · 🔍 {ZOOM_HINT}")
        with st.expander("직경 18점 값 표"):
            st.dataframe({
                "Point": list(range(1, len(pos) + 1)),
                "위치 x [cm]": np.round(pos, 3),
                "상대 두께": np.round(pts_v / T_ref, 5),
                "두께 [nm]": np.round(pts_v, 3),
            }, hide_index=True, use_container_width=True)
            st.caption(f"평균 {mean18:.2f} nm · σ {sd18:.2f} nm · η {eta18:.2f} % · "
                       "점 위치는 −R ~ +R 등간격 (논문의 실제 측정 좌표와 다를 수 있음)")

with right:
    with st.container(border=True):
        st.markdown(f"##### 반경 프로파일 · H = {H:g} cm, λ = {lam:g} cm, γ = {gamma:g}")
        st.plotly_chart(radial_figure(), use_container_width=True)
        st.caption("점선: 같은 H, λ, γ에서 타깃 반경 3–7 cm (논문 Fig. 2(a)의 타깃 반경 범위)")
        if measured is not None:
            m_mean, _, m_eta = P.nonuniformity(measured)
            rmse = float(np.sqrt(np.mean((pts_v - measured) ** 2)) / m_mean * 100)
            st.caption(f"논문 실측: 평균 {m_mean:.1f} nm, η {m_eta:.2f} % · 모델: 평균 {mean18:.1f} nm, η {eta18:.2f} % · "
                       f"18점 RMSE {rmse:.2f} % (실측 평균 대비) · 중심 두께는 실측에 맞춘 값")
    with st.container(border=True):
        st.markdown("##### 타깃 반경에 따른 불균일도")
        st.plotly_chart(sweep_figure(), use_container_width=True)

st.markdown(
    "<div class='model-box'>"
    "L = √(H² + r² + q² − 2rq·cos φ) &nbsp;→&nbsp; cos α = cos β = H/L &nbsp;→&nbsp; S(L) = exp(−L/λ)<br>"
    "F(r) = ∫₀ᵃ∫₀²ᵖⁱ [H²/L⁴ · S + γ · H/(2L³) · (1 − S)] · q dφ dq &nbsp;→&nbsp; T(r) = T_ref · F(r)/F(0)"
    "</div>", unsafe_allow_html=True)
st.caption("Z. Cheng et al., Cryst. Growth Des. 24, 339–346 (2024)의 Eq. (2)–(4), (7)–(9)에 평면 타깃(d = 0) 조건 적용 · "
           "산란 성분(γ)은 이 프로젝트에서 넣은 확장이고 γ = 1은 논문 Fig. 2(c, d) 실측 36점으로 보정한 값"
           + (f" (18점 모양 RMSE {CAL['metrics']['before']['rmse_pct']:.1f} % → {CAL['metrics']['after']['rmse_pct']:.1f} %)"
              if CAL else "")
           + " · 절대 증착량은 계산하지 않고, 중심 두께는 입력한 기준값 · 타깃 레이스트랙 침식은 반영하지 않음")
