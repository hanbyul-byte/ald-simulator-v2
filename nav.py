"""화면 전환 도우미 (공정 선택 ↔ 각 시뮬레이터)"""
import streamlit as st

PROCESSES = ("ald", "cvd")


def go(process=None):
    """process: 'ald' | 'cvd' | None(공정 선택 화면)"""
    st.session_state.process = process
    if process:
        st.query_params["process"] = process
    else:
        st.query_params.clear()


def back_button(where=st, **kw):
    where.button("← 공정 선택으로", on_click=go, args=(None,), **kw)
