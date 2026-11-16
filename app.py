"""
박막 공정 시뮬레이터 — 시작 화면 (공정 선택)

실행:  streamlit run app.py

화면 구성
- 공정 선택 화면: ALD / CVD / PVD 중 하나를 고름
- ALD → ald_sim.py  (TDMA-Hf · Langmuir 포화 + 전구체 소모 모델, 논문 데이터 보정)
- CVD → cvd_sim.py  (중심 두께: 경계층 물질 전달 + 아레니우스 표면 반응 · 반경 분포: 논문 보정 프로파일)
- PVD → pvd_sim.py  (평면 타깃 스퍼터링 · 코사인 방출 + 거리 감소 + Ar 충돌 감쇠 → 중심 기준 상대 두께)
주소 뒤에 ?process=ald · ?process=cvd · ?process=pvd 를 붙이면 해당 시뮬레이터로 바로 열림
"""
import runpy
from pathlib import Path

import streamlit as st

from nav import PROCESSES, go

HERE = Path(__file__).parent
PAGES = {"ald": HERE / "ald_sim.py", "cvd": HERE / "cvd_sim.py", "pvd": HERE / "pvd_sim.py"}

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

CARDS = [
    dict(key="ald", name="ALD", ready=True),
    dict(key="cvd", name="CVD", ready=True),
    dict(key="pvd", name="PVD", ready=True),
]

cols = st.columns(len(CARDS), gap="large")
for col, card in zip(cols, CARDS):
    with col:
        with st.container(border=True):
            st.subheader(card["name"])
            st.button(
                f"{card['name']} 시뮬레이터 열기" + ("" if card["ready"] else " (준비 중)"),
                key=f"open_{card['key']}",
                on_click=go,
                args=(card["key"],),
                type="primary" if card["ready"] else "secondary",
                use_container_width=True,
            )
