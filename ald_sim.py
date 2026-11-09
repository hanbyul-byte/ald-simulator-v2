"""
ALD 시뮬레이터 페이지 (app.py에서 실행됨)
TDMA-Hf 전구체 · Cross-flow 반응기 · Langmuir 포화 성장 모델

실행:  streamlit run app.py
"""
import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from nav import back_button
from model import PTS, WC, WR, f_T, gpc, gpc_max0, p_local, p_vap, stick, thickness

HERE = Path(__file__).parent
CALIB_PATH = HERE / "calibrated.json"
DATA_PATH = HERE / "data" / "hfox_tox_suppfig1.csv"
CALIB = json.loads(CALIB_PATH.read_text(encoding="utf-8")) if CALIB_PATH.exists() else None
ZKEYS = ("z_tilt_y", "z_tilt_x", "z_defocus")

# 색
ACCENT = "#0d7f74"
S2 = "#eb6834"
REF = "#868d98"
RAMP = ["#ecf8f6", "#b4e4dd", "#6fc7bb", "#1f9e8f", "#137a6e", "#0b4f48"]
# 두께 맵 컬러: 얇음 = 빨강 → 주황 → 노랑 → 초록 → 하늘 → 파랑 = 두꺼움
THK_RAMP = ["#d7191c", "#ff8c00", "#ffff00", "#22c55e", "#00e5ff", "#1e40ff", "#0000c8"]
COLORSCALE = [[i / (len(THK_RAMP) - 1), c] for i, c in enumerate(THK_RAMP)]

# ---------------------------------------------------------------------------
# 입력 정의
# ---------------------------------------------------------------------------
CONSTS = [
    dict(g="반응 속도론", id="sref", label="점착 계수 s₀ (300 K 기준)", unit="", min=1e-8, max=1.0, v=1e-3,
         fmt="{:.2e}", hint="분자가 표면에 부딪혔을 때 달라붙을 확률. 1e-3처럼 입력 가능"),
    dict(g="반응 속도론", id="Ea", label="활성화 에너지 Eₐ", unit="kJ/mol", min=0.0, max=200.0, v=10.0,
         fmt="{:.1f}", hint="클수록 저온에서 반응이 느려짐"),
    dict(g="반응 속도론", id="Ton", label="탈수 시작 온도 T_on", unit="K", min=300.0, max=900.0, v=550.0,
         fmt="{:.0f}", hint="이 온도부터 표면 –OH가 줄어 GPC_max가 감소"),
    dict(g="반응 속도론", id="Tscale", label="감소 폭 T_scale", unit="K", min=1.0, max=2000.0, v=150.0,
         fmt="{:.0f}", hint="작을수록 T_on 이후 GPC_max가 급격히 떨어짐"),
    dict(g="기하 · 분자", id="Ldep", label="전구체 고갈 길이 L_dep", unit="cm", min=0.01, max=1000.0, v=8.0,
         fmt="{:.2f}", hint="빈 표면 위에서 전구체가 소모되는 길이. 작을수록 공급 부족 시 Inlet 쪽만 두꺼워짐 "
                            "(상류가 포화되면 전구체가 더 멀리 감: Bohart–Adams 모델)"),
    dict(g="기하 · 분자", id="r", label="전구체 유효 반경 r", unit="nm", min=0.1, max=2.0, v=0.45,
         fmt="{:.3f}", hint="TDMA-Hf 분자 크기. GPC_max를 결정"),
    dict(g="전구체 공급", id="Pref", label="75 °C 전구체 분압 P_ref", unit="Torr", min=1e-4, max=10.0, v=0.1,
         fmt="{:.3g}", hint="'전구체 온도' 입력 모드에서 사용. 캐니스터 75 °C일 때 Inlet 분압"),
    dict(g="전구체 공급", id="dHvap", label="기화 엔탈피 ΔH_vap", unit="kJ/mol", min=10.0, max=300.0, v=100.0,
         fmt="{:.1f}", hint="Clausius–Clapeyron: 전구체 온도가 낮아질 때 분압이 떨어지는 정도"),
]

# 공정 조건 기본값 (dose 시간은 log10 슬라이더)
CTRL_DEFAULTS = {"P0": 0.05, "log_t": -1.0, "T": 525, "n": 100, "supply": "분압 직접 입력", "Tp": 75}
PRESETS = [
    ("포화 (기본)", {}),
    ("짧은 dose", {"log_t": math.log10(0.005)}),
    ("저온", {"T": 320, "log_t": -2.0}),
    ("고온 · 탈수", {"T": 620}),
]


def sig(v):
    if v >= 1:
        return f"{v:.2f}"
    if v >= 0.1:
        return f"{v:.3f}"
    return f"{v:.2g}"


# ---------------------------------------------------------------------------
# 페이지 설정 · 상태
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', ui-monospace, Menlo, monospace; font-size: 1.45rem; }
    .model-box { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 0.8rem; line-height: 1.8; opacity: 0.85; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 기본 상수 = 논문 데이터로 보정한 값 (calibrated.json). 파일이 없으면 CONSTS의 v 값
DEFAULTS = {c["id"]: (float(f"{CALIB['consts'][c['id']]:.4g}") if CALIB else c["v"]) for c in CONSTS}
ZERN = {k: (CALIB["consts"].get(k, 0.0) if CALIB else 0.0) for k in ZKEYS}

ss = st.session_state
ss.setdefault("stage", "setup")
ss.setdefault("consts", dict(DEFAULTS))
for k, v in CTRL_DEFAULTS.items():
    ss.setdefault(k, v)
ss.setdefault("preset", 0)


def apply_preset(i):
    for k, v in CTRL_DEFAULTS.items():
        ss[k] = v
    for k, v in PRESETS[i][1].items():
        ss[k] = v
    ss.preset = i


def clear_preset():
    ss.preset = None


def apply_paper_condition(td, tp):
    """논문 실험 조건을 시뮬레이터 공정 조건으로 적용"""
    ss.T = int(round(td + 273.15))
    ss.supply = "전구체 온도"
    ss.Tp = int(tp)
    ss.n = int(CALIB["assumed"]["cycles"])
    ss.log_t = math.log10(CALIB["assumed"]["dose_s"])
    ss.preset = None


if ss.stage == "setup":
    back_button()
st.title("ALD 웨이퍼 시뮬레이터" if ss.stage == "sim" else "ALD 시뮬레이터")
st.caption("TDMA-Hf 전구체 · Cross-flow 반응기 · Langmuir 포화 성장 모델")

# ---------------------------------------------------------------------------
# 1단계: 상수 입력
# ---------------------------------------------------------------------------
if ss.stage == "setup":
    with st.container(border=True):
        st.subheader("상수 입력")
        if CALIB:
            m = CALIB["metrics"]
            st.caption(
                f"기본값 = HfOx 9-point 실측 315점(증착 100–250 °C × 전구체 55–75 °C)으로 피팅한 값 · "
                f"RMSE {m['rmse_phys']:.2f} nm"
                + (f" · 교차검증 {m['loco_rmse_phys']:.2f} nm" if "loco_rmse_phys" in m else "")
                + f" · 논문 공정 조건: {CALIB['assumed']['cycles']} cycle, pulse {CALIB['assumed']['dose_s']} s"
            )
        with st.form("setup"):
            raw = {}
            groups = list(dict.fromkeys(c["g"] for c in CONSTS))
            cols = st.columns(len(groups))
            for col, grp in zip(cols, groups):
                with col:
                    st.markdown(f"**{grp}**")
                    for c in [c for c in CONSTS if c["g"] == grp]:
                        unit = f" [{c['unit']}]" if c["unit"] else ""
                        ss.setdefault(f"in_{c['id']}", str(ss.consts.get(c["id"], DEFAULTS[c["id"]])))
                        raw[c["id"]] = st.text_input(
                            c["label"] + unit,
                            help=f"{c['hint']} · 허용 {c['min']} ~ {c['max']}",
                            key=f"in_{c['id']}",
                        )
            b1, b2 = st.columns([1, 1])
            submitted = b1.form_submit_button("이 값으로 시뮬레이션 시작", type="primary")
            reset = b2.form_submit_button("기본값으로 되돌리기")

        if reset:
            ss.consts = dict(DEFAULTS)
            for c in CONSTS:
                ss.pop(f"in_{c['id']}", None)
            st.rerun()

        # 검증
        vals, errors = {}, []
        for c in CONSTS:
            s = raw[c["id"]].strip().replace(",", "")
            try:
                v = float(s)
                if not math.isfinite(v):
                    raise ValueError
            except ValueError:
                errors.append(f"**{c['label']}**: 숫자를 입력하세요. 예: 0.001 또는 1e-3")
                continue
            if v < c["min"] or v > c["max"]:
                errors.append(f"**{c['label']}**: {c['min']} 이상 {c['max']} 이하로 입력하세요.")
                continue
            vals[c["id"]] = v

        # 파생값
        if "r" in vals:
            g0, n0 = gpc_max0(vals["r"])
            st.info(
                f"흡착 자리 밀도 N_max **{n0:.3f}** /nm²  ·  "
                f"저온 한계 GPC_max,0 **{g0:.3f}** Å/cycle  ·  "
                f"분자 1개 면적 **{math.pi * vals['r'] ** 2:.3f}** nm²"
            )

        if submitted:
            if errors:
                for e in errors:
                    st.error(e)
            else:
                ss.consts = vals
                ss.stage = "sim"
                st.rerun()
    st.stop()

# ---------------------------------------------------------------------------
# 2단계: 사이드바 — 공정 조건 · 고정 상수
# ---------------------------------------------------------------------------
c = {**DEFAULTS, **ss.consts, **ZERN}
zero_z = all(abs(ZERN[k]) < 1e-12 for k in ZKEYS)

with st.sidebar:
    back_button(use_container_width=True)
    st.divider()
    st.subheader("공정 조건")
    st.radio("전구체 공급 입력", ["분압 직접 입력", "전구체 온도"], key="supply", horizontal=True,
             on_change=clear_preset)
    if ss.supply == "전구체 온도":
        st.slider("전구체(캐니스터) 온도 [°C]", 40, 95, step=1, key="Tp", on_change=clear_preset,
                  help="Clausius–Clapeyron으로 Inlet 분압 계산 (P_ref, ΔH_vap 상수 사용)")
        P0_now = float(p_vap(ss.Tp + 273.15, c))
        st.caption(f"→ Inlet 분압 P₀ = **{sig(P0_now)} Torr**")
    else:
        st.slider("압력 P₀ [Torr]", 0.005, 0.5, step=0.005, format="%.3f", key="P0",
                  help="Inlet에서의 전구체 분압", on_change=clear_preset)
        P0_now = float(ss.P0)
    st.slider("Dose 시간 log₁₀(t / s)", -3.0, 1.0, step=0.01, key="log_t",
              help="한 사이클에서 전구체를 흘리는 시간 (로그 스케일)", on_change=clear_preset)
    t_dose = 10 ** ss.log_t
    st.caption(f"Dose 시간 = **{sig(t_dose)} s**")
    st.slider("웨이퍼 온도 [K]", 300, 700, step=5, key="T", on_change=clear_preset)
    st.caption(f"= {ss.T - 273.15:.0f} °C")
    st.slider("사이클 수", 10, 500, step=10, key="n", on_change=clear_preset)
    st.button("공정 조건 초기화", on_click=apply_preset, args=(0,))

    st.divider()
    st.subheader("두께 맵 컬러 범위")
    ss.setdefault("cmin", 0.0)
    ss.setdefault("cmax", 10.0)
    cc1, cc2 = st.columns(2)
    cc1.number_input("최소 [nm]", min_value=0.0, step=0.5, format="%.2f", key="cmin")
    cc2.number_input("최대 [nm]", min_value=0.01, step=0.5, format="%.2f", key="cmax",
                     help="사이클 수·온도와 관계없이 이 범위로 색이 고정됩니다. 범위를 넘는 값은 양 끝 색으로 표시돼요.")
    if ss.cmax <= ss.cmin:
        st.error("최대는 최소보다 커야 해요.")

    st.divider()
    st.subheader("고정 상수")
    st.table({
        "상수": [x["label"] for x in CONSTS],
        "값": [x["fmt"].format(c[x["id"]]) + (f" {x['unit']}" if x["unit"] else "") for x in CONSTS],
    })
    if not zero_z:
        st.caption("장비 패턴 보정 적용 중 (저차 Zernike) · tilt_y {:+.3f} · tilt_x {:+.3f} · "
                   "defocus {:+.3f}".format(*(ZERN[k] for k in ZKEYS)))
    if st.button("상수 다시 입력"):
        ss.stage = "setup"
        st.rerun()

# 예시 조건 (프리셋)
pcols = st.columns([1.2] + [1] * len(PRESETS) + [0.4, 2.6])
pcols[0].markdown("<div style='padding-top:6px;opacity:.6;font-size:.8rem'>예시 조건</div>", unsafe_allow_html=True)
for i, (name, _) in enumerate(PRESETS):
    pcols[i + 1].button(name, key=f"preset_{i}", on_click=apply_preset, args=(i,),
                        type="primary" if ss.preset == i else "secondary", use_container_width=True)
# 역설계 버튼 → 팝업에서 ① 조건 입력 → ② 결과 (구현은 파일 맨 아래)
open_inv = pcols[-1].button("🎯 역설계 · 목표 두께·균일도 → 공정 조건 추천", key="inv_open_btn",
                            use_container_width=True)
inv_slot = st.container()   # 적용 메시지 · (팝업을 못 쓰는 Streamlit 버전에서는) 역설계 화면이 들어갈 자리

# ---------------------------------------------------------------------------
# 계산
# ---------------------------------------------------------------------------
P0, T, n = P0_now, float(ss.T), ss.n
use_z = not zero_z   # 장비 패턴 보정은 항상 적용
g_gpc, g_n = gpc_max0(c["r"])
g_now = g_gpc * float(f_T(T, c))


def p_loc(d):
    """dose 종료 시점 국소 분압 [Torr]"""
    return p_local(T, P0, t_dose, d, c)


def thk_xy(x, y):  # nm
    return thickness(T, P0, t_dose, n, x, y, c, use_zernike=use_z)


t_sat = max(g_now * n / 10, 1e-9)
pts_xy = np.array(PTS, dtype=float)
pts_d = np.hypot(pts_xy[:, 0], pts_xy[:, 1])
pts_v = thk_xy(pts_xy[:, 0], pts_xy[:, 1])
pts_th = gpc(T, P0, t_dose, pts_d, c) / max(g_now, 1e-12)   # 피복률 θ
mean = pts_v.mean()
sd = pts_v.std(ddof=1)
nu_s = sd / mean * 100 if mean > 0 else float("nan")
nu_r = (pts_v.max() - pts_v.min()) / (2 * mean) * 100 if mean > 0 else float("nan")
s0 = float(stick(T, c))
exposure_L = P0 * t_dose * 1e6
d_near, d_far = pts_d.min(), pts_d.max()

# 두께 맵 컬러 범위 (고정값, 사이클 수와 무관)
c_lo = float(ss.cmin)
c_hi = float(ss.cmax) if ss.cmax > ss.cmin else c_lo + 1.0

# ---------------------------------------------------------------------------
# KPI
# ---------------------------------------------------------------------------
k = st.columns(6)
k[0].metric("평균 두께", f"{mean:.2f} nm")
k[1].metric("불균일도 1σ", f"{nu_s:.1f} %")
k[2].metric("불균일도 (max−min)/2·mean", f"±{nu_r:.1f} %")
k[3].metric("GPC_max(T)", f"{g_now:.3f} Å/cyc")
k[4].metric("점착 계수 s₀(T)", f"{s0:.2e}")
k[5].metric("전구체 노출량", f"{round(exposure_L):,} L")

# ---------------------------------------------------------------------------
# 웨이퍼 맵
# ---------------------------------------------------------------------------
def wafer_figure():
    N = 220
    xs = np.linspace(WC[0] - WR, WC[0] + WR, N)
    ys = np.linspace(WC[1] - WR, WC[1] + WR, N)
    X, Y = np.meshgrid(xs, ys)
    inside = np.hypot(X - WC[0], Y - WC[1]) <= WR
    D = np.hypot(X, Y)
    Z = np.where(inside, thk_xy(X, Y), np.nan)
    TH = gpc(T, P0, t_dose, D, c) / max(g_now, 1e-12)
    custom = np.stack([D, TH, p_loc(D)], axis=-1)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=Z, customdata=custom, colorscale=COLORSCALE, zmin=c_lo, zmax=c_hi,
        colorbar=dict(title="두께 [nm]", thickness=12, len=0.6),
        hovertemplate=("x %{x:.2f}, y %{y:.2f} cm · Inlet %{customdata[0]:.2f} cm<br>"
                       "두께 <b>%{z:.3f}</b> nm · θ <b>%{customdata[1]:.3f}</b><br>"
                       "국소 압력 <b>%{customdata[2]:.4f}</b> Torr<extra></extra>"),
    ))

    # 등두께선 (Inlet 중심 동심원, 웨이퍼 안쪽만)
    ang = np.linspace(0, 2 * np.pi, 400)
    for d in np.arange(3, 13, 1.0):
        cx, cy = d * np.cos(ang), d * np.sin(ang)
        mask = np.hypot(cx - WC[0], cy - WC[1]) <= WR
        cx, cy = np.where(mask, cx, np.nan), np.where(mask, cy, np.nan)
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines", line=dict(color="rgba(255,255,255,0.5)", width=1),
                                 hoverinfo="skip", showlegend=False))

    # 웨이퍼 외곽 + 노치
    fig.add_shape(type="circle", x0=WC[0] - WR, y0=WC[1] - WR, x1=WC[0] + WR, y1=WC[1] + WR,
                  line=dict(color="#4a525e", width=1.5))
    fig.add_shape(type="circle", x0=-0.12, y0=WC[1] + WR - 0.12, x1=0.12, y1=WC[1] + WR + 0.12,
                  fillcolor="white", line=dict(color="#4a525e", width=1))

    # Inlet + 흐름 화살표
    fig.add_shape(type="rect", x0=-0.6, y0=0.3, x1=0.6, y1=-0.15, fillcolor="#141820", line_width=0)
    fig.add_annotation(x=0, y=2.6, ax=0, ay=0.45, xref="x", yref="y", axref="x", ayref="y",
                       showarrow=True, arrowhead=2, arrowwidth=2.5, arrowcolor=REF, text="")
    fig.add_annotation(x=0.85, y=0.08, text="<b>Gas inlet</b>", showarrow=False, xanchor="left")
    fig.add_annotation(x=0.35, y=1.5, text="흐름 방향", showarrow=False, xanchor="left",
                       font=dict(color=REF, size=11))

    # 9-Point
    fig.add_trace(go.Scatter(
        x=pts_xy[:, 0], y=pts_xy[:, 1], mode="markers",
        marker=dict(size=9, color="white", line=dict(color="#141820", width=1.5)),
        text=[f"{v:.2f}" for v in pts_v],
        customdata=np.stack([np.arange(1, 10), pts_d, pts_th], axis=-1),
        hovertemplate="Point %{customdata[0]:.0f} · Inlet %{customdata[1]:.2f} cm<br>"
                      "두께 <b>%{text}</b> nm · θ %{customdata[2]:.3f}<extra></extra>",
        showlegend=False,
    ))

    for (px, py), v in zip(pts_xy, pts_v):
        fig.add_annotation(x=px, y=py, text=f"{v:.2f}", showarrow=False, yshift=16,
                           font=dict(family="ui-monospace, Menlo, monospace", size=12, color="#141820"),
                           bgcolor="rgba(255,255,255,0.9)", borderpad=2)

    fig.update_xaxes(range=[-5.4, 5.4], visible=False)
    fig.update_yaxes(range=[-0.9, 13.0], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=600, margin=dict(l=10, r=10, t=10, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ---------------------------------------------------------------------------
# 선 그래프
# ---------------------------------------------------------------------------
def line_fig(x, near, far, ref, cur_x, cur_near, cur_far, xlabel, log_x, x_fmt, band=None):
    ymax = max(g_gpc * 1.2, 0.05)
    fig = go.Figure()
    if band:
        fig.add_vrect(x0=band[0], x1=band[1], fillcolor=ACCENT, opacity=0.07, line_width=0)
    fig.add_vline(x=cur_x, line=dict(color=REF, width=1), opacity=0.6)
    fig.add_trace(go.Scatter(x=x, y=ref, name="GPC_max(T)", line=dict(color=REF, dash="dash", width=1.4),
                             customdata=x_fmt, hovertemplate="%{y:.3f} Å"))
    fig.add_trace(go.Scatter(x=x, y=near, name=f"가장 가까운 지점 ({d_near:.1f} cm)",
                             line=dict(color=ACCENT, width=2), hovertemplate="%{y:.3f} Å"))
    fig.add_trace(go.Scatter(x=x, y=far, name=f"가장 먼 지점 ({d_far:.1f} cm)",
                             line=dict(color=S2, width=2), hovertemplate="%{y:.3f} Å"))
    fig.add_trace(go.Scatter(x=[cur_x, cur_x], y=[cur_near, cur_far], mode="markers",
                             marker=dict(size=11, color=[ACCENT, S2], line=dict(color="white", width=2)),
                             hoverinfo="skip", showlegend=False))
    fig.update_xaxes(title=xlabel, type="log" if log_x else "linear", showgrid=True)
    fig.update_yaxes(title="GPC [Å/cycle]", range=[0, ymax], showgrid=True)
    fig.update_layout(height=290, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)))
    return fig


def at(d, TT, tt):
    return gpc(TT, P0, tt, d, c)


t_axis = np.logspace(-3, 1, 160)
T_axis = np.linspace(300, 700, 160)

fig_sat = line_fig(
    t_axis, at(d_near, T, t_axis), at(d_far, T, t_axis), np.full_like(t_axis, g_now),
    t_dose, float(at(d_near, T, t_dose)), float(at(d_far, T, t_dose)),
    "Dose 시간 [s]", True, [f"dose {sig(v)} s" for v in t_axis])
fig_win = line_fig(
    T_axis, at(d_near, T_axis, t_dose), at(d_far, T_axis, t_dose), g_gpc * f_T(T_axis, c),
    T, float(at(d_near, T, t_dose)), float(at(d_far, T, t_dose)),
    "온도 [K]", False, [f"{v:.0f} K" for v in T_axis], band=(300, min(700, c["Ton"])))

# ---------------------------------------------------------------------------
# 레이아웃
# ---------------------------------------------------------------------------
left, right = st.columns([1.05, 1], gap="medium")

with left:
    with st.container(border=True):
        h1, h2 = st.columns([1, 1])
        h1.markdown("##### 두께 맵 · 9-Point")
        th_min = pts_th.min()
        if th_min >= 0.95:
            h2.success("전 지점 포화")
        elif th_min >= 0.5:
            h2.warning(f"부분 포화 · 최소 θ {th_min:.2f}")
        else:
            h2.error(f"미포화 · 최소 θ {th_min:.2f}")
        st.plotly_chart(wafer_figure(), use_container_width=True)
        st.caption(f"두께 [nm] · 컬러 범위 {c_lo:.2f} ~ {c_hi:.2f} nm 고정 "
                   f"(참고: 현재 조건의 완전 포화 두께 {t_sat:.2f} nm)")

        with st.expander("9-Point 측정값 표"):
            st.dataframe(
                {
                    "Point": list(range(1, 10)),
                    "x [cm]": pts_xy[:, 0], "y [cm]": pts_xy[:, 1],
                    "Inlet 거리 [cm]": np.round(pts_d, 2),
                    "국소 압력 [Torr]": np.round(p_loc(pts_d), 4),
                    "두께 [nm]": np.round(pts_v, 3),
                    "θ": np.round(pts_th, 3),
                },
                hide_index=True, use_container_width=True,
            )

with right:
    with st.container(border=True):
        st.markdown(f"##### 포화 곡선 · T = {T:.0f} K")
        st.plotly_chart(fig_sat, use_container_width=True)
    with st.container(border=True):
        st.markdown(f"##### ALD 온도 창 · dose = {sig(t_dose)} s")
        st.plotly_chart(fig_win, use_container_width=True)

# ---------------------------------------------------------------------------
# 논문 실측 데이터 비교
# ---------------------------------------------------------------------------
if DATA_PATH.exists():
    import pandas as pd

    @st.cache_data
    def load_data():
        return pd.read_csv(DATA_PATH)

    df = load_data()
    cyc0 = CALIB["assumed"]["cycles"] if CALIB else 100
    dose0 = CALIB["assumed"]["dose_s"] if CALIB else 0.2

    def predict_df(d_):
        return thickness(d_.T_dep_C.values + 273.15, p_vap(d_.T_prec_C.values + 273.15, c), dose0, cyc0,
                         d_.x_cm.values, d_.y_cm.values, c, use_zernike=use_z)

    st.divider()
    st.subheader("논문 실측 데이터와 비교")
    st.caption(f"HfOx 두께 9-point 맵 (Supplementary Fig. 1a) · 현재 상수로 예측 · "
               f"논문 공정 조건 {cyc0} cycle, TDMA-Hf pulse {dose0} s · 분압은 전구체 온도에서 계산")
    df = df.assign(pred=predict_df(df))
    err = df.pred - df.Tox_nm
    rm = float(np.sqrt(np.mean(err**2)))
    starved = df.T_prec_C <= 55

    q1, q2 = st.columns([1, 1.15], gap="medium")
    with q1:
        with st.container(border=True):
            m1, m2, m3 = st.columns(3)
            m1.metric("전체 RMSE", f"{rm:.2f} nm")
            m2.metric("전구체 충분 (≥60 °C)", f"{np.sqrt(np.mean(err[~starved] ** 2)):.2f} nm")
            m3.metric("전구체 부족 (55 °C)", f"{np.sqrt(np.mean(err[starved] ** 2)):.2f} nm")
            pf = go.Figure()
            for msk, name, col in [(~starved, "전구체 ≥ 60 °C", ACCENT), (starved, "전구체 55 °C", S2)]:
                s_ = df[msk]
                pf.add_trace(go.Scatter(
                    x=s_.Tox_nm, y=s_.pred, mode="markers", name=name, marker=dict(size=6, color=col, opacity=0.8),
                    customdata=np.stack([s_.T_dep_C, s_.T_prec_C, s_.point_no], -1),
                    hovertemplate="증착 %{customdata[0]} °C · 전구체 %{customdata[1]} °C · P%{customdata[2]}<br>"
                                  "실측 %{x:.1f} · 예측 %{y:.2f} nm<extra></extra>"))
            hi = float(max(df.Tox_nm.max(), df.pred.max()) * 1.05)
            pf.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color=REF, dash="dash", width=1),
                                    hoverinfo="skip", showlegend=False))
            pf.update_xaxes(title="실측 두께 [nm]", range=[0, hi])
            pf.update_yaxes(title="예측 두께 [nm]", range=[0, hi], scaleanchor="x")
            pf.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                             legend=dict(orientation="h", y=1.08, x=0))
            st.plotly_chart(pf, use_container_width=True)

    with q2:
        with st.container(border=True):
            s1, s2, s3 = st.columns([1, 1, 1.2])
            td_sel = s1.selectbox("증착 온도 [°C]", sorted(df.T_dep_C.unique()), index=3)
            tp_sel = s2.selectbox("전구체 온도 [°C]", sorted(df.T_prec_C.unique()), index=0)
            s3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            s3.button("이 조건을 시뮬레이터에 적용", on_click=apply_paper_condition, args=(td_sel, tp_sel),
                      disabled=CALIB is None, use_container_width=True)
            sel = df[(df.T_dep_C == td_sel) & (df.T_prec_C == tp_sel)].sort_values("point_no")
            bf = go.Figure()
            bf.add_trace(go.Bar(x=[f"P{i}" for i in sel.point_no], y=sel.Tox_nm, name="실측",
                                marker_color=REF))
            bf.add_trace(go.Bar(x=[f"P{i}" for i in sel.point_no], y=sel.pred, name="예측",
                                marker_color=ACCENT))
            bf.update_yaxes(title="두께 [nm]")
            bf.update_layout(height=290, barmode="group", margin=dict(l=10, r=10, t=10, b=10),
                             legend=dict(orientation="h", y=1.12, x=0))
            st.plotly_chart(bf, use_container_width=True)
            e_ = sel.pred - sel.Tox_nm
            st.caption(f"이 조건 RMSE {np.sqrt(np.mean(e_ ** 2)):.2f} nm · 평균 실측 {sel.Tox_nm.mean():.2f} / "
                       f"예측 {sel.pred.mean():.2f} nm · P1 = 웨이퍼 위(Inlet에서 가장 먼 점), P9 = 아래")



# ---------------------------------------------------------------------------
# 역설계: 목표 사양 → 공정 조건 추천 (계산은 inverse.py)
# 위쪽 '🎯 역설계' 버튼 → 팝업에서 ① 조건 입력 → ② 결과 → 'N순위 적용'으로 사이드바에 반영
# ---------------------------------------------------------------------------
import pandas as pd  # noqa: E402

import inverse as INV  # noqa: E402

INV_SIG = (tuple(sorted(c.items())), use_z)   # 상수가 바뀌면 이전 결과는 무효
INV_STEPS = ("① 조건 입력", "② 결과")


def apply_inverse(row):
    """추천 조건을 사이드바 공정 조건에 적용 (탐색 격자가 슬라이더 눈금과 같아서 그대로 들어감)"""
    ss.T = int(round(row["T_K"]))
    ss.supply = "전구체 온도"
    ss.Tp = int(round(row["Tp"]))
    ss.log_t = float(round(row["log_t"], 2))
    ss.n = int(round(row["n"]))
    ss.preset = None
    ss.inv_applied = (f"역설계 추천 적용: {row['T_K']:.0f} K ({row['T_K'] - 273.15:.0f} °C) · 전구체 {row['Tp']:.0f} °C · "
                      f"dose {row['t']:.3g} s · {row['n']:.0f} cycle")


def inv_back():
    ss.inv_step = "input"


def inverse_inputs():
    """① 조건 입력. 제출하면 계산해서 ss.inv에 저장하고 True"""
    st.caption("원하는 두께·균일도를 넣으면 현재 모델(상수 + 장비 패턴 보정)로 후보 공정 조건을 격자로 **전부** 계산해서, "
               "조건을 만족하는 것 중 목표에 가장 좋은 조건을 추천합니다. 격자는 사이드바 눈금과 같아서"
               "(웨이퍼 온도 5 K, 전구체 온도 1 °C, 사이클 10) 추천값을 그대로 적용할 수 있습니다.")
    d = ss.get("inv_last", {})   # 팝업을 닫았다 열어도 마지막 입력값 유지
    with st.form("inv_form", border=False):
        f1, f2, f3 = st.columns(3, gap="medium")
        with f1:
            st.markdown("**목표 사양**")
            st.number_input("목표 평균 두께 [nm]", 0.5, 100.0, d.get("target", 10.0), step=0.5, key="invf_target")
            st.number_input("허용 오차 ± [nm]", 0.01, 10.0, d.get("tol", 0.3), step=0.1, key="invf_tol")
            st.number_input("최대 불균일도 1σ [%]", 0.0, 100.0, d.get("nu_max", 4.0), step=0.5, key="invf_numax")
            st.number_input("최소 포화도 θ (1 = 완전 포화)", 0.0, 1.0, d.get("theta_min", 0.9), step=0.05, key="invf_thmin")
        with f2:
            st.markdown("**평가 방식**")
            st.selectbox("최적화 목표", list(INV.OBJECTIVES), format_func=INV.OBJECTIVES.get, key="invf_obj",
                         index=list(INV.OBJECTIVES).index(d.get("objective", "nu")))
            st.selectbox("평가 영역", ["9pt", "wafer"], key="invf_region", index=["9pt", "wafer"].index(d.get("region", "9pt")),
                         format_func={"9pt": "9-point (측정 위치)", "wafer": "웨이퍼 전체 (9점 바깥은 외삽)"}.get)
            st.slider("공정 마진: 공급량 변동 ± [%]", 0, 30, int(round(d.get("margin", 0.1) * 100)), step=5, key="invf_margin",
                      help="전구체 공급량(dose)이 이만큼 늘거나 줄어도 조건을 계속 만족하는 후보만 추천합니다. 0이면 검사하지 않습니다.")
        with f3:
            st.markdown("**탐색 범위**")
            st.slider("증착 온도 [°C]", 27, 427, tuple(d.get("T_C_rng", (100, 250))), key="invf_T")
            st.slider("전구체 온도 [°C]", 40, 95, tuple(d.get("Tp_rng", (55, 75))), key="invf_Tp")
            st.checkbox("사이클 수로 두께 맞추기", value=d.get("n_free", True), key="invf_nfree",
                        help=f"끄면 사이클 수를 현재 사이드바 값({ss.n} cycle)으로 고정합니다.")
            st.checkbox("dose 시간도 조정", value=d.get("t_free", False), key="invf_tfree",
                        help="끄면 논문 공정 조건 0.2 s로 고정합니다 (실측 데이터가 있는 조건).")
            st.select_slider("dose 시간 범위 [s]", options=[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                             value=tuple(d.get("t_rng", (0.05, 1.0))), key="invf_trng")
        go_ = st.form_submit_button("추천 조건 찾기 →", type="primary", use_container_width=True)
    if not go_:
        return False
    if ss.invf_tfree and ss.invf_trng[0] >= ss.invf_trng[1]:
        st.error("dose 시간 범위의 최솟값이 최댓값보다 작아야 해요.")
        return False
    opts = dict(target=ss.invf_target, tol=ss.invf_tol, nu_max=ss.invf_numax, theta_min=ss.invf_thmin,
                objective=ss.invf_obj, region=ss.invf_region, T_C_rng=ss.invf_T, Tp_rng=ss.invf_Tp,
                t_fixed=0.2, t_free=ss.invf_tfree, t_rng=ss.invf_trng, n_fixed=ss.n, n_free=ss.invf_nfree,
                margin=ss.invf_margin / 100)
    with st.spinner("후보 조건 계산 중…"):
        ss.inv = {"res": INV.search(c, use_z, **opts), "opts": opts, "sig": INV_SIG}
    ss.inv_last = opts
    return True


def inverse_results():
    """② 결과: 추천 표 + 적용 버튼 + 그래프 + 주의할 점"""
    R, o = ss.inv["res"], ss.inv["opts"]
    allv, top = R["all"], R["top"]
    sat = allv[allv.thmin > 0.995]
    sat_floor = float(sat.nu.min()) if len(sat) else float("nan")
    mtxt = f"공급량 ±{o['margin'] * 100:.0f} % 변동 포함" if o["margin"] > 0 else "공정 마진 검사 없음"
    st.caption(f"목표 {o['target']:g} ± {o['tol']:g} nm · 불균일도 ≤ {o['nu_max']:g} % · θ ≥ {o['theta_min']:g} · "
               f"{INV.OBJECTIVES[o['objective']]} · {mtxt}")
    if R["status"] == "ok":
        st.success(f"후보 {R['n_cand']:,}개 × 평가 지점 {R['n_points']}개 계산 · {R['n_feas']:,}개가 조건 만족")
    else:
        tips = []
        if np.isfinite(sat_floor) and o["nu_max"] < sat_floor:
            tips.append(f"이 모델에서 완전 포화 영역의 불균일도는 약 {sat_floor:.1f} %입니다(장비 패턴 보정에서 옴). "
                        "최대 불균일도를 그 이상으로 올리거나 공정 마진을 줄여 보세요.")
        if not o["n_free"]:
            tips.append("두께가 문제라면 '사이클 수로 두께 맞추기'를 켜 보세요.")
        st.error(f"조건을 모두 만족하는 후보가 없습니다 ({R['n_cand']:,}개 계산). "
                 "아래는 제약 위반이 가장 작은 절충안입니다. " + " ".join(tips))

    show = pd.DataFrame({
        "순위": np.arange(1, len(top) + 1),
        "웨이퍼 온도": [f"{k:.0f} K ({k - 273.15:.0f} °C)" for k in top.T_K],
        "전구체 °C": top.Tp.round(0).astype(int), "dose [s]": top.t, "사이클": top.n.astype(int),
        "평균 두께 [nm]": top["mean"], "불균일도 1σ [%]": top.nu, "변동 시 최악 [%]": top.nu_worst,
        "최소 θ": top.thmin, "전구체 사용량": top.usage,
        "데이터 근거": np.where(top.in_range, "측정 범위 안", "외삽"),
        "조건 만족": np.where(top.feasible, "✓", "✕"),
    })
    st.dataframe(show.style.format({"dose [s]": "{:.3g}", "평균 두께 [nm]": "{:.2f}", "불균일도 1σ [%]": "{:.2f}",
                                    "변동 시 최악 [%]": "{:.2f}", "최소 θ": "{:.2f}", "전구체 사용량": "×{:.3f}"}),
                 hide_index=True, use_container_width=True)
    bcols = st.columns(max(len(top), 1))
    for i, (bc, (_, row)) in enumerate(zip(bcols, top.iterrows())):
        if bc.button(f"{i + 1}순위 적용", key=f"inv_apply_{i}", type="primary" if i == 0 else "secondary",
                     use_container_width=True):
            apply_inverse(row.to_dict())
            ss.inv_open = False
            st.rerun()          # 전체 화면을 다시 그림 → 팝업이 닫히고 사이드바·두께 맵에 반영
    st.caption("전구체 사용량 = 분압(전구체 온도) × dose 시간 × 사이클, 논문 조건(75 °C · 0.2 s · 100 cycle) = 1. "
               "'변동 시 최악'은 공급량이 ±마진만큼 변했을 때 가장 나쁜 불균일도입니다.")

    tab1, tab2, tab3 = st.tabs(["균일도 vs 전구체 사용량", "가능 영역", "해석할 때 주의할 점"])
    with tab1:
        thk = allv[allv.ok_thk & allv.ok_th]
        cloud = thk.iloc[:: max(1, len(thk) // 3000)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cloud.usage, y=cloud.nu_worst, mode="markers", name="후보",
                                 marker=dict(color="#c3c8cf", size=5), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=R["pareto"].usage, y=R["pareto"].nu_worst, mode="lines+markers",
                                 name="Pareto 최적", line=dict(color=ACCENT, width=2), marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=top.usage, y=top.nu_worst, mode="markers+text", name="추천",
                                 text=[str(i + 1) for i in range(len(top))], textposition="top center",
                                 marker=dict(color=S2, size=11)))
        fig.add_hline(y=o["nu_max"], line=dict(color="#c23b3a", dash="dash", width=1.2),
                      annotation_text="불균일도 상한", annotation_position="top left")
        fig.update_xaxes(title="전구체 사용량 (논문 조건 = 1, 로그 축)", type="log")
        fig.update_yaxes(title="불균일도 1σ, 변동 시 최악 [%]", rangemode="tozero")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1, x=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("두께·포화도 조건을 만족하는 후보(회색)와 Pareto 최적선. 왼쪽 아래로 갈수록 좋습니다.")
    with tab2:
        TKs, Tps, LTs = R["axes"]
        li = int(np.argmin(np.abs(LTs - (top.log_t.iloc[0] if len(top) else LTs[0]))))
        cube = lambda col: allv[col].to_numpy().reshape(len(TKs), len(Tps), len(LTs))[:, :, li].T  # noqa: E731
        nu_g, ok_g = cube("nu_worst"), cube("feasible").astype(bool)
        zmax = max(o["nu_max"] * 2, (sat_floor if np.isfinite(sat_floor) else 0) + 1)
        fig = go.Figure()
        fig.add_trace(go.Heatmap(x=TKs - 273.15, y=Tps, z=nu_g, zmin=0, zmax=zmax,
                                 colorscale=[[0, "#1e40ff"], [0.3, "#22c55e"], [0.55, "#ffff00"],
                                             [0.8, "#ff8c00"], [1, "#d7191c"]],
                                 colorbar=dict(title="1σ %", thickness=12, len=0.7),
                                 hovertemplate="증착 %{x:.0f} °C · 전구체 %{y} °C<br>불균일도(최악) %{z:.2f} %<extra></extra>"))
        fig.add_trace(go.Heatmap(x=TKs - 273.15, y=Tps, z=np.where(ok_g, np.nan, 1.0), showscale=False,
                                 hoverinfo="skip", colorscale=[[0, "rgba(120,124,130,0.68)"], [1, "rgba(120,124,130,0.68)"]]))
        if DATA_PATH.exists():
            mc = pd.read_csv(DATA_PATH)[["T_dep_C", "T_prec_C"]].drop_duplicates()
            fig.add_trace(go.Scatter(x=mc.T_dep_C, y=mc.T_prec_C, mode="markers", name="실측 조건",
                                     marker=dict(color="black", size=5)))
        on = np.abs(top.log_t.to_numpy() - LTs[li]) < 1e-9 if len(top) else np.zeros(0, bool)
        fig.add_trace(go.Scatter(x=top.T_K[on] - 273.15, y=top.Tp[on], mode="markers+text", name="추천",
                                 text=[str(i + 1) for i in np.flatnonzero(on)], textfont=dict(size=11, color="#141820"),
                                 marker=dict(color="white", size=17, line=dict(color=S2, width=2))))
        fig.update_xaxes(title="증착 온도 [°C]")
        fig.update_yaxes(title="전구체 온도 [°C]")
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1, x=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"dose {10 ** LTs[li]:.3g} s 단면 · 색 = 불균일도(공급량 변동 시 최악), 회색 = 조건 불만족, "
                   "검은 점 = 논문 실측 조건, 숫자 = 추천 순위")
    with tab3:
        loco = CALIB["metrics"].get("loco_rmse_zernike" if use_z else "loco_rmse_phys") if CALIB else None
        notes = [
            "추천은 **모델 예측**입니다." + (f" 조건 1개씩 뺀 교차검증에서 모델 예측 오차는 약 {loco:.2f} nm였으므로, "
                                         "이보다 작은 차이로 갈리는 추천끼리는 실제로 구분되지 않을 수 있습니다." if loco else ""),
            (f"완전 포화 영역(θ > 0.995)의 최소 불균일도는 약 {sat_floor:.1f} %입니다. 이는 장비 패턴 보정(Zernike tilt)에서 나옵니다. "
             "이보다 낮은 불균일도는 포화 직전 조건에서 전구체 고갈(먼 쪽이 얇아짐)과 장비 패턴(먼 쪽이 두꺼워짐)이 상쇄될 때만 "
             "나오는데, 이런 조건은 공급량이 조금만 변해도 깨집니다. 공정 마진 검사가 이런 조건을 걸러냅니다."
             if np.isfinite(sat_floor) and use_z else
             "장비 패턴 보정이 꺼져 있으면 포화 영역의 예측 불균일도가 0에 가까워집니다. 실제 장비 편차는 반영되지 않습니다."),
            "모델에서 전구체 온도와 dose 시간은 노출량(분압 × 시간)으로만 작용해서 서로 대체할 수 있습니다. "
            "전구체 온도 → 분압 환산은 피팅된 ΔH_vap에 기반합니다.",
            "사이클 수는 두께 ∝ 사이클로 맞춥니다(모델 정의). 핵생성 지연은 반영하지 않으며, 실측은 100 cycle 조건뿐입니다.",
            ("평가 영역이 9-point라 웨이퍼 가장자리(9점 바깥)는 반영되지 않습니다." if o["region"] == "9pt" else
             "웨이퍼 전체 평가는 9점 바깥까지 모델과 장비 패턴 보정(정규화 반경 3 cm)을 외삽한 결과입니다."),
        ]
        st.markdown("\n".join(f"- {t}" for t in notes))


def inverse_flow():
    """① 조건 입력 → ② 결과 (팝업 안 또는 화면 위쪽 패널에서 공통으로 사용)"""
    if ss.get("inv") and ss.inv["sig"] != INV_SIG:     # 상수가 바뀌면 처음부터
        ss.pop("inv")
        ss.inv_step = "input"
    step = ss.get("inv_step", "input") if ss.get("inv") else "input"
    head = st.empty()
    if step == "input":
        body = st.empty()
        with body.container():
            done = inverse_inputs()
        if done:
            body.empty()                  # 입력 폼을 지우고 같은 자리에 결과 표시
            step = ss.inv_step = "result"
    if step == "result":
        cb, _ = st.columns([1, 3])
        cb.button("← 조건 다시 입력", on_click=inv_back, use_container_width=True)
        inverse_results()
    head.markdown(" → ".join(f"**{s}**" if i == INV_STEPS.index("② 결과" if step == "result" else "① 조건 입력")
                             else f"<span style='opacity:.45'>{s}</span>" for i, s in enumerate(INV_STEPS)),
                  unsafe_allow_html=True)


# 팝업(st.dialog)으로 띄우기. 팝업이 없는 옛 Streamlit이면 위쪽 패널로 펼침
_dialog = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)
if _dialog is not None:
    @_dialog("🎯 역설계 · 목표 사양 → 공정 조건 추천", width="large")
    def inverse_dialog():
        inverse_flow()

    if open_inv:
        inverse_dialog()
else:
    if open_inv:
        ss.inv_open = not ss.get("inv_open", False)
    if ss.get("inv_open"):
        with inv_slot:
            with st.container(border=True):
                hc1, hc2 = st.columns([5, 1])
                hc1.markdown("#### 🎯 역설계 · 목표 사양 → 공정 조건 추천")
                hc2.button("닫기", key="inv_close", on_click=lambda: ss.update(inv_open=False), use_container_width=True)
                inverse_flow()

if ss.get("inv_applied"):
    with inv_slot:
        st.success(ss.pop("inv_applied") + " · 사이드바와 두께 맵에 반영됐습니다. 🎯 역설계 버튼을 다시 누르면 결과를 다시 볼 수 있어요.")
