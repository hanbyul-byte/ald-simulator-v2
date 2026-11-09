"""
박막 공정 시뮬레이터 — 시작 화면 (공정 선택)

실행:  streamlit run app.py

화면 구성
- 공정 선택 화면: ALD / CVD 중 하나를 고름
- ALD → ald_sim.py  (TDMA-Hf · Langmuir 포화 + 전구체 소모 모델, 논문 데이터 보정)
- CVD → cvd_sim.py  (준비 중)
주소 뒤에 ?process=ald 또는 ?process=cvd 를 붙이면 해당 시뮬레이터로 바로 열림
"""
import runpy
from pathlib import Path

import streamlit as st

from nav import PROCESSES, go

HERE = Path(__file__).parent
PAGES = {"ald": HERE / "ald_sim.py", "cvd": HERE / "cvd_sim.py"}

st.set_page_config(page_title="박막 공정 시뮬레이터", page_icon="🧪", layout="wide")

ss = st.session_state
if "process" not in ss:
    q = st.query_params.get("process")
    ss.process = q if q in PROCESSES else None

# ---------------------------------------------------------------------------
# 각 시뮬레이터 화면
# ---------------------------------------------------------------------------
if ss.process in PAGES:
    runpy.run_path(str(PAGES[ss.process]), run_name="__main__")
    st.stop()

# ---------------------------------------------------------------------------
# 공정 선택 화면
# ---------------------------------------------------------------------------
st.title("박막 공정 시뮬레이터")
st.caption("시뮬레이션할 증착 공정을 고르세요.")

CARDS = [
    dict(
        key="ald",
        name="ALD",
        full="Atomic Layer Deposition · 원자층 증착",
        lines=[
            "TDMA-Hf / H₂O · HfOx · Cross-flow 반응기",
            "Langmuir 포화 + 흐름 방향 전구체 소모 모델",
            "논문 실측 315점으로 상수 보정 (RMSE 0.95 nm)",
        ],
        ready=True,
    ),
    dict(
        key="cvd",
        name="CVD",
        full="Chemical Vapor Deposition · 화학 기상 증착",
        lines=["준비 중"],
        ready=False,
    ),
]

cols = st.columns(len(CARDS), gap="large")
for col, card in zip(cols, CARDS):
    with col:
        with st.container(border=True):
            st.subheader(card["name"])
            st.caption(card["full"])
            for line in card["lines"]:
                st.markdown(f"- {line}")
            st.button(
                f"{card['name']} 시뮬레이터 열기" + ("" if card["ready"] else " (준비 중)"),
                key=f"open_{card['key']}",
                on_click=go,
                args=(card["key"],),
                type="primary" if card["ready"] else "secondary",
                use_container_width=True,
            )
