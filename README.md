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
