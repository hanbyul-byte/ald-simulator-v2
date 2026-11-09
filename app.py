"""
ALD 시뮬레이터 (Streamlit 버전)
TDMA-Hf 전구체 · Cross-flow 반응기 · Langmuir 포화 성장 모델

실행:  streamlit run app.py
"""
import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

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
st.set_page_config(page_title="ALD 시뮬레이터", page_icon="🧪", layout="wide")

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
ss.setdefault("stage", "setup")
ss.setdefault("consts", {c["id"]: c["v"] for c in CONSTS})
for k, v in CTRL_DEFAULTS.items():
    ss.setdefault(k, v)
ss.setdefault("preset", 0)
ss.setdefault("zern", {k: 0.0 for k in ZKEYS})
ss.setdefault("use_z", False)


def apply_preset(i):
    for k, v in CTRL_DEFAULTS.items():
        ss[k] = v
    for k, v in PRESETS[i][1].items():
        ss[k] = v
    ss.preset = i


def clear_preset():
    ss.preset = None


def load_calibration():
    """논문 데이터로 보정한 상수를 입력칸에 채움"""
    for c in CONSTS:
        v = CALIB["consts"][c["id"]]
        ss[f"in_{c['id']}"] = f"{v:.4g}"
    ss.zern = {k: CALIB["consts"].get(k, 0.0) for k in ZKEYS}


def apply_paper_condition(td, tp):
    """논문 실험 조건을 시뮬레이터 공정 조건으로 적용"""
    ss.T = int(round(td + 273.15))
    ss.supply = "전구체 온도"
    ss.Tp = int(tp)
    ss.n = int(CALIB["assumed"]["cycles"])
    ss.log_t = math.log10(CALIB["assumed"]["dose_s"])
    ss.preset = None


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
            b0, b1 = st.columns([1, 2.4])
            b0.button("논문 데이터 보정값 불러오기", on_click=load_calibration, type="secondary")
            b1.caption(
                f"HfOx 9-point 실측 315점(증착 100–250 °C × 전구체 55–75 °C)으로 피팅한 값 · "
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
                        ss.setdefault(f"in_{c['id']}", str(ss.consts.get(c["id"], c["v"])))
                        raw[c["id"]] = st.text_input(
                            c["label"] + unit,
                            help=f"{c['hint']} · 허용 {c['min']} ~ {c['max']}",
                            key=f"in_{c['id']}",
                        )
            b1, b2 = st.columns([1, 1])
            submitted = b1.form_submit_button("이 값으로 시뮬레이션 시작", type="primary")
            reset = b2.form_submit_button("기본값으로 되돌리기")

        if reset:
            ss.consts = {c["id"]: c["v"] for c in CONSTS}
            ss.zern = {k: 0.0 for k in ZKEYS}
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
c = {**{x["id"]: x["v"] for x in CONSTS}, **ss.consts, **ss.zern}

with st.sidebar:
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
    zero_z = all(abs(ss.zern[k]) < 1e-12 for k in ZKEYS)
    st.toggle("장비 패턴 보정 (저차 Zernike)", key="use_z", disabled=zero_z,
              help="논문 데이터에서 추정한 장비 고유 두께 패턴(위쪽이 두꺼움)을 곱해 줌. "
                   "'논문 데이터 보정값 불러오기'로 상수를 넣었을 때만 켤 수 있어요.")

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
        st.caption("Zernike 계수 · tilt_y {:+.3f} · tilt_x {:+.3f} · defocus {:+.3f}".format(
            *(ss.zern[k] for k in ZKEYS)))
    if st.button("상수 다시 입력"):
        ss.stage = "setup"
        st.rerun()

# 예시 조건 (프리셋)
pcols = st.columns([1.2] + [1] * len(PRESETS) + [3])
pcols[0].markdown("<div style='padding-top:6px;opacity:.6;font-size:.8rem'>예시 조건</div>", unsafe_allow_html=True)
for i, (name, _) in enumerate(PRESETS):
    pcols[i + 1].button(name, key=f"preset_{i}", on_click=apply_preset, args=(i,),
                        type="primary" if ss.preset == i else "secondary", use_container_width=True)

# ---------------------------------------------------------------------------
# 계산
# ---------------------------------------------------------------------------
P0, T, n = P0_now, float(ss.T), ss.n
use_z = bool(ss.use_z) and not zero_z
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
