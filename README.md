# 박막 공정 시뮬레이터 (ALD · CVD)

첫 화면에서 공정을 고르면 해당 시뮬레이터가 열린다. 주소 뒤에 `?process=ald` / `?process=cvd`를 붙이면 바로 열림.

- **ALD:** TDMA-Hf 전구체 · Cross-flow 반응기 · Langmuir 포화 + 흐름 방향 전구체 소모 모델
- **CVD:** 준비 중

```
streamlit run app.py
```

## 파일

| 파일 | 내용 |
|---|---|
| `app.py` | 시작 화면 (ALD / CVD 선택) |
| `ald_sim.py` | ALD 시뮬레이터 화면 |
| `cvd_sim.py` | CVD 시뮬레이터 화면 (준비 중) |
| `nav.py` | 화면 전환 도우미 |
| `model.py` | 성장 모델 (화면과 보정 스크립트가 같이 씀) |
| `inverse.py` | 역설계: 목표 사양 → 공정 조건 추천 (ALD 화면 위쪽 🎯 역설계 버튼 → 팝업에서 조건 입력 → 결과) |
| `calibrate.py` | 실측 데이터로 상수 피팅 → `calibrated.json`, `calibration.png` |
| `data/hfox_tox_suppfig1.csv` | 논문 Supplementary Fig. 1(a)에서 읽은 HfOx 두께 315점 |

## 모델

- 포화 GPC: `GPC_max(T) = GPC_max,0(r) · f(T)` (T_on 이후 지수 감소, –OH 탈수)
- Inlet 노출량: `τ = s(T)·Φ(P0,T)·t / (N_max·f(T))`
- 흐름 방향 소모(Bohart–Adams): `ξ = d/L_dep · s(T)/s₀ · √(300/T)`, `θ = (e^τ − 1)/(e^τ + e^ξ − 1)`
  - Inlet(d→0)에서는 기존 Langmuir `θ = 1 − e^(−τ)`와 같음
  - 상류 표면이 포화되면 전구체가 하류로 더 멀리 감 → 공급 부족 시 포화 '전선'이 웨이퍼 중간에서 멈춤
  - 기존 `P = P0·exp(−d/L_dep)` 고정 감쇠로는 55 °C 데이터의 급격한 두께 기울기와 55→60 °C 사이의 급변을 재현하지 못해서 바꿈
- 전구체 온도 → 분압: `P0 = P_ref · exp(−ΔH_vap/R · (1/T_p − 1/348.15 K))`
- (선택) 장비 패턴: 저차 Zernike 곱셈 보정 `1 + a·Y + b·X + c·(X²+Y²)`

## 역설계 (inverse.py)

목표 평균 두께 ± 허용 오차, 최대 불균일도 1σ, 최소 포화도를 넣으면 공정 조건을 추천한다.

- 탐색: 웨이퍼 온도(5 K) × 전구체 온도(1 °C) × dose 시간(선택) 격자를 전부 계산. 격자 = 사이드바 눈금이라 추천을 그대로 '적용' 가능
- 사이클 수: 두께 ∝ 사이클로 목표 두께에 맞춤 (10 cycle 단위, 끄면 현재 값 고정)
- 목표: 불균일도 / 전구체 사용량 / 증착 온도 / 사이클 수 최소
- 공정 마진: 공급량이 ±마진(기본 10 %) 변해도 조건을 만족하는 후보만 추천
  - 이 검사가 없으면 '포화 직전'에서 전구체 고갈(먼 쪽 얇음)과 장비 패턴(Zernike tilt, 먼 쪽 두꺼움)이 상쇄되는
    불안정한 조건이 최적으로 뽑힘 (예: 불균일도 2.1 % → 공급 10 % 변동 시 3.3 %)
- 전구체 사용량 = 분압 × dose 시간 × 사이클, 논문 조건(75 °C · 0.2 s · 100 cycle) = 1
- 실측 범위(증착 100–250 °C, 전구체 55–75 °C, dose 0.2 s) 밖의 추천은 '외삽'으로 표시

## 보정 (calibrate.py)

```
pip install scipy matplotlib pandas
python calibrate.py --cv                      # 논문 조건: 100 cycle, TDMA-Hf pulse 0.2 s (P_ref 0.1 Torr 고정)
```

데이터: 증착온도 100–250 °C(7) × 전구체온도 55–75 °C(5) × 9-point = 315점

논문 공정 조건 (Lee et al., Commun. Mater. 7, 243 (2026), Methods):
6-inch ALD (Lucida D100, NCD), 4-inch Si 웨이퍼를 Inlet 쪽에 배치, TDMA-Hf/H₂O,
pulse/purge 0.2/20 s, 100 cycle, ~310 mTorr, Ar carrier 40 sccm.
측정 좌표는 Inlet 쪽 (0, 0) 기준 → 앱 기하(Inlet 원점, 웨이퍼 중심 (0, 7.5))와 동일

| | RMSE (nm) |
|---|---|
| 물리 모델 (전체 / 전구체 ≥60 °C / 55 °C) | 0.95 / 0.66 / 1.68 |
| + 저차 Zernike | 0.88 / 0.51 / 1.68 |
| 교차검증: 조건 1개씩 제외(35-fold), 물리 / +Zernike | 1.11 / 1.05 |

식별 가능성 주의
- `s₀ · P_ref · dose`는 곱으로만 들어감 → `P_ref`는 0.1 Torr로 고정하고 `s₀`만 피팅
- `r`(GPC_max,0)과 사이클 수는 두께 스케일을 함께 정함 → 사이클 수를 바꾸면 `r`이 달라짐
- 9-point 좌표: 논문 그림 위쪽(P1) = (0, 10.5), 아래쪽(P9) = (0, 4.5).
  논문 Supp. Fig. 7·8의 좌표 표기와 본문 "Inlet 근처가 두꺼움"으로 확인
