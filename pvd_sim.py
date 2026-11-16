"""
PVD 시뮬레이터 페이지 (app.py에서 실행됨) — 준비 중

물리 모델과 데이터가 정해지면 이 파일에 화면을 만든다.
- 모델 식은 pvd_model.py로 따로 두고(Streamlit 의존성 없이) 화면과 보정 스크립트가 같이 쓰게 할 예정 (ALD·CVD와 같은 구조)
- st.set_page_config는 app.py에서 이미 호출하므로 여기서 다시 부르지 않는다.
- 다른 페이지와 session_state 키가 겹치지 않도록 PVD 쪽 키는 "pvd_" 접두어를 쓴다.
"""
import streamlit as st

from nav import back_button

back_button()
st.title("PVD 시뮬레이터")
st.info("준비 중이에요. 물리 모델과 데이터가 들어오면 이 화면에 시뮬레이터가 열립니다.")
