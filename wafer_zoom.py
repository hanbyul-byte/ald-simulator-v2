"""
웨이퍼 두께 맵 표시 (ALD · CVD · PVD 공통) — 맵을 클릭하면 전체화면으로 확대

Streamlit Plotly 차트에 원래 있는 '전체화면' 기능(차트에 마우스를 올리면 오른쪽 위 툴바에 나오는 ⛶ 버튼)을
그대로 쓴다. 맵을 클릭하면 아래 스크립트가 그 버튼을 대신 눌러 준다.
- 전체화면에서는 Streamlit이 그림 크기를 브라우저 창 크기에 맞춰 다시 그림 (hover · 확대도 그대로 됨)
- 닫기: Esc 또는 툴바의 '전체화면 닫기' 버튼
- 드래그(영역 확대)나 툴바 버튼 클릭은 전체화면을 열지 않음

사용:  wafer_chart(wafer_figure())
"""
import inspect

import streamlit as st

WAFER_META = "wafer-map"     # 이 표시(layout.meta)가 붙은 Plotly 그림만 클릭 → 전체화면
HINT = "맵을 클릭하면 전체화면 (Esc로 닫기)"

# 앱 화면(W)에서 웨이퍼 맵을 찾아 클릭 이벤트를 붙이는 스크립트.
# __WIN__ / __HOST__ 는 넣는 방식에 따라 바뀜 (st.html: 앱 화면에서 바로 실행, iframe: 부모 화면을 조작)
_JS = r"""
(function () {
  var W = __WIN__;
  var D = W.document;
  var KEY = "__waferFullscreen";
  var META = "__META__";
  var ID = Math.random().toString(36).slice(2);

  // 스크립트를 담은 빈 요소는 숨김 (맵 아래에 빈 칸이 생기지 않게)
  try {
    var host = __HOST__;
    var box = host && host.closest ? host.closest(".element-container") : null;
    if (box) box.style.display = "none";
  } catch (e) {}

  // 다시 설치될 때(코드 변경 등) 이전 타이머 정리
  if (W[KEY] && W[KEY].timer) W.clearInterval(W[KEY].timer);
  W[KEY] = { id: ID, timer: null };

  if (!D.getElementById("wafer-fs-style")) {
    var css = D.createElement("style");
    css.id = "wafer-fs-style";
    css.textContent = ".wafer-fs-ready .nsewdrag { cursor: zoom-in !important; }";
    D.head.appendChild(css);
  }

  // Streamlit의 전체화면 버튼 찾기
  function fsButton(gd) {
    // Streamlit 1.35 이후: Plotly 툴바(modebar) 안의 'Fullscreen' 버튼 (전체화면 중에는 'Close fullscreen')
    var b = gd.querySelector('.modebar-btn[data-title="Fullscreen"]');
    if (b) return b;
    // 예전 Streamlit: 차트 바깥에 있는 'View fullscreen' 버튼
    var box = gd.closest(".element-container");
    if (!box) return null;
    var cands = box.querySelectorAll("button");
    for (var i = 0; i < cands.length; i++) {
      var t = (cands[i].getAttribute("title") || "") + " " + (cands[i].getAttribute("aria-label") || "");
      if (/full\s*screen/i.test(t) && !/close|exit/i.test(t)) return cands[i];
    }
    return null;
  }

  var last = 0;
  function open(gd) {
    var now = Date.now();
    if (now - last < 600) return;          // 한 번 클릭에 이벤트가 두 번 와도 한 번만
    var b = fsButton(gd);                   // 이미 전체화면이면 없음 → 아무 일도 안 함
    if (b) { last = now; b.click(); }
  }

  function hook(gd) {
    if (!gd.on || !gd.layout || gd.layout.meta !== META) return;
    gd.classList.toggle("wafer-fs-ready", !!fsButton(gd));   // 클릭 가능할 때만 돋보기 커서

    var h = gd[KEY];
    if (h && h.id !== ID) {                 // 이전 설치에서 붙인 이벤트 떼기
      try { gd.removeListener("plotly_click", h.click); } catch (e) {}
      gd.removeEventListener("click", h.native);
      h = null;
    }
    if (!h) {
      h = gd[KEY] = {
        id: ID,
        bound: false,
        // 웨이퍼 위(데이터가 있는 곳) 클릭
        click: function () { open(gd); },
        // 웨이퍼 바깥 빈 곳 클릭. 드래그 확대 · 툴바 클릭은 제외
        native: function (ev) {
          if (gd._dragged) return;
          var t = ev.target;
          if (t && t.closest && t.closest(".modebar")) return;
          open(gd);
        }
      };
      gd.addEventListener("click", h.native);
    }
    // Plotly가 그림을 새로 만들면 plotly_click 이벤트가 지워질 수 있어서 매번 확인
    var ls = gd._ev && gd._ev.listeners ? gd._ev.listeners("plotly_click") : null;
    if (!h.bound || (ls && ls.indexOf(h.click) < 0)) {
      gd.on("plotly_click", h.click);
      h.bound = true;
    }
  }

  function scan() {
    if (!W[KEY] || W[KEY].id !== ID) return;
    var gds = D.querySelectorAll(".js-plotly-plot");
    for (var i = 0; i < gds.length; i++) {
      try { hook(gds[i]); } catch (e) {}
    }
  }

  scan();
  W[KEY].timer = W.setInterval(scan, 500);   // 다시 그려진 차트에도 붙도록 주기적으로 확인
})();
"""


def _script(win, host):
    return _JS.replace("__WIN__", win).replace("__HOST__", host).replace("__META__", WAFER_META)


def _install():
    """클릭 → 전체화면 스크립트를 화면에 넣음 (내용이 같으면 다시 실행되지 않음)"""
    html = getattr(st, "html", None)
    if html is not None and "unsafe_allow_javascript" in inspect.signature(html).parameters:
        # 최신 Streamlit: 앱 화면에서 바로 실행
        html(f"<script>{_script('window', 'document.currentScript')}</script>", unsafe_allow_javascript=True)
    else:
        # 예전 Streamlit: 높이 0인 iframe에서 부모(앱) 화면을 조작
        import streamlit.components.v1 as components
        components.html(f"<script>{_script('window.parent', 'window.frameElement')}</script>", height=0)


def wafer_chart(fig):
    """웨이퍼 두께 맵을 그리고, 클릭하면 전체화면으로 확대되게 함"""
    fig.update_layout(meta=WAFER_META)
    st.plotly_chart(fig, use_container_width=True)
    _install()
