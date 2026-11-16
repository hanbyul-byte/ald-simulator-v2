# 박막 공정 시뮬레이터 (ALD · CVD)

첫 화면에서 공정을 고르면 해당 시뮬레이터가 열린다. 주소 뒤에 `?process=ald` / `?process=cvd`를 붙이면 바로 열림.

- **ALD:** TDMA-Hf 전구체 · Cross-flow 반응기 · Langmuir 포화 + 흐름 방향 전구체 소모 모델
- **CVD:** Cross-flow 반응기 · 평판 경계층 물질 전달 + 아레니우스 표면 반응(직렬 저항) + 흐름 방향 전구체 고갈

```
streamlit run app.py
```

## 파일

| 파일 | 내용 |
|---|---|
| `app.py` | 시작 화면 (ALD / CVD 선택) |
| `ald_sim.py` | ALD 시뮬레이터 화면 |
| `cvd_sim.py` | CVD 시뮬레이터 화면 |
| `cvd_model.py` | CVD 성장 모델 (화면과 `calibrate_cvd.py`가 같이 씀) |
| `calibrate_cvd.py` | CVD 실측 데이터로 상수 피팅 → `cvd_calibrated.json`, `cvd_calibration.png` |
| `nav.py` | 화면 전환 도우미 |
| `model.py` | 성장 모델 (화면과 보정 스크립트가 같이 씀) |
| `inverse.py` | 역설계: 목표 사양 → 공정 조건 추천 (ALD 화면 위쪽 🎯 역설계 버튼 → 팝업에서 조건 입력 → 결과) |
| `calibrate.py` | 실측 데이터로 상수 피팅 → `calibrated.json`, `calibration.png` |
| `data/hfox_tox_suppfig1.csv` | 논문 Supplementary Fig. 1(a)에서 읽은 HfOx 두께 315점 |

## ALD 모델

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

## CVD 모델 (cvd_model.py)

| 단계 | 식 | 의미 |
|---|---|---|
| 경계층 | `δ(x) = 5.0·√(ν(x+x₀)/u)` | Inlet에서 멀수록 두꺼워짐 |
| 물질 전달 | `h_g(x) = D_g/δ(x)` | ∝ 1/√x |
| 표면 반응 | `k_s(T) = k₀·exp(−Eₐ/RT)` | 아레니우스 |
| 직렬 저항 | `k_eff = (1/h_g + 1/k_s)⁻¹` | 느린 쪽이 율속 |
| 전구체 고갈 | `C_g(x) = C_g0·exp(−∫₀ˣ k_eff dx′ / (u·H))` | 앞쪽에서 소모된 만큼 뒤쪽 농도 감소 |
| 증착률 | `R = C_g·k_eff·M/ρ_film` | nm/min, × 시간 = 두께 |

- `k_eff`가 x에 따라 변하므로 고갈 식의 지수는 `k_eff·x` 대신 적분으로 둠 (k_eff가 상수면 원래 식과 같음).
  `1/k_eff = b·√(x+x₀) + 1/k_s` 꼴이라 해석해가 있고, 반응 율속에서 생기는 상쇄 오차를 없앤 형태로 계산 (수치 적분과 1e-8 이내 일치)
- 기체 물성: 1 atm·300 K 기준값에서 `ν, D ∝ T^α · (760/P)`로 환산 (α = 0이면 온도 보정 없음)
- Hot-wall 가정: 기체 온도 = 웨이퍼 온도, `C_g0 = p_prec/(R·T)`
- 기하·9-point 좌표는 ALD와 같음 (Inlet 원점, x = Inlet에서의 거리)
- 평판 경계층 식은 `δ < H`에서 성립 → 넘으면 화면에 경고
- 기본 상수는 보정 전 초깃값(상압 CVD, HfO₂ 막). `cvd_calibrated.json`이 있으면 그 값을 기본으로 씀

화면: 두께 맵 · 율속 단계 판정(`k_s/h_g`) · 아레니우스 플롯(반응/물질 전달 한계선 포함) · 흐름 방향 증착률·농도 프로파일 ·
`data/cvd_data.csv`가 있으면 실측 비교

## CVD 보정 (calibrate_cvd.py)

```
python calibrate_cvd.py --data data/cvd_data.csv --cv                 # 기본 피팅: k0, Ea, D0, x0
python calibrate_cvd.py --data data/cvd_data.csv --fit k0,Ea,D0,x0,H  # 고갈 정도(H)까지
python calibrate_cvd.py --demo                                        # 합성 데이터로 정답 복원 테스트
```

데이터 CSV (한 행 = 측정점)
- 필수: `T_C`(또는 `T_K`) · `x_cm, y_cm`(또는 Inlet 거리 `d_cm`) · `thickness_nm`(또는 `rate_nm_min`)
- 선택: `P_Torr, p_prec_Torr, u_cm_s, time_min` — 없으면 `--P --p-prec --u --time` 값 사용

피팅
- `k0` 대신 '데이터 평균 온도에서의 k_s'를 피팅 → k₀·Eₐ 상관 완화. 다중 시작 least squares, 파라미터 1σ 출력
- `--demo` 결과 (잡음 2 %, 450–700 °C × 9-point): k₀ 3.37e7 (정답 3e7) · Eₐ 135.8 (135) · D₀ 0.0697 (0.07) · x₀ 2.51 (2.5)

식별 가능성 주의
- `D₀`와 `ν₀`는 `D₀/√ν₀` 묶음으로만 들어감 → `ν₀` 고정
- 반응 율속(저온)만 있으면 `D₀, x₀`가, 물질 전달 율속(고온)만 있으면 `k₀, Eₐ`가 정해지지 않음 → 전환 구간을 걸치는 온도 데이터가 필요
- 두께 스케일은 `p_prec·M/ρ`와 곱으로 묶임 → 전구체 분압이 부정확하면 `k₀`가 그만큼 흡수
