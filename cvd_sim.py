"""
CVD 시뮬레이터 페이지 (app.py에서 실행됨)

아직 모델 코드가 없어서 자리만 만들어 둔 화면.
CVD 코드를 받으면 이 파일에 구현한다.
- st.set_page_config는 app.py에서 이미 호출하므로 여기서 다시 부르지 않는다.
- ALD 페이지와 session_state 키가 겹치지 않도록 CVD 쪽 키는 "cvd_" 접두어를 쓴다.
"""
import streamlit as st

from nav import back_button

with st.sidebar:
    back_button(use_container_width=True)

back_button()
st.title("CVD 시뮬레이터")
st.caption("Chemical Vapor Deposition · 화학 기상 증착")
st.info("CVD 시뮬레이터는 준비 중이에요. 모델 코드가 들어오면 이 화면에 연결됩니다.")
