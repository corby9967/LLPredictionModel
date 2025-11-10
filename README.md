# 수술 후 요추 전만각(Post-Surgery Lumbar Lordosis) 예측 모델

이 저장소는 수술 전 영상 및 임상 지표를 기반으로 **수술 후 요추 전만각(LL)** 을 예측하기 위한 딥러닝 회귀 모델을 포함합니다.

모델은 **다중 출력 구조(Multi-Head MLP)** 로 설계되었으며, 학습 손실은 **MAE(L1Loss)** 로 구성되어 있습니다.

---

## 1. 모델 개요

모델은 **공유 MLP 백본(Shared MLP Backbone)** 을 중심으로 두 개의 출력 헤드를 사용합니다.

* **ΔSL Head (5채널)** : 각 요추 레벨별 Segmental Lordosis 변화를 예측합니다.
* **ΔLL_aux Head (1채널)** : 전체 LL 변화량(ΔLL)을 보조적으로 학습합니다.

**출력 계산식:**

```
SL_post_pred = ΔSL + SL_pre
LL_post_pred = ΣΔSL + LL_pre
```

---

## 2. 입력 특징 (Input Features)

* **Segment-level features (5×12 = 60차원)**
  SL_pre, 수술 접근법(op_tlif, op_plif, op_olif, op_alif),
  Cage 크기(cage_w, cage_l, cage_h, cage_d),
  Cage 위치(cagepos_anterior, cagepos_center, cagepos_posterior)

* **Global features (4차원)**
  Age, BMI, Sex(0/1), SS_pre
  → 총 64차원 입력이며, `replicate_globals` 옵션은 비활성화되었습니다.

---

## 3. 모델 구조

<img width="1858" height="744" alt="스크린샷 2025-11-10 오후 7 44 57" src="https://github.com/user-attachments/assets/2d081397-60cc-4c18-a344-95d8944a5f83" />

| 구성 요소   | 내용                                                                                                                                         |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| 입력 차원   | 64                                                                                                                                         |
| 백본      | Linear(64→128) → ReLU → Dropout(0.2) → Linear(128→128) → ReLU → Dropout(0.2) → Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→32) → ReLU |
| 출력 헤드   | ΔSL Head: Linear(32→5), ΔLL_aux Head: Linear(32→1)                                                                                         |
| 활성화 함수  | ReLU                                                                                                                                       |
| 가중치 초기화 | Xavier(Glorot) uniform                                                                                                                     |
| 입력 정규화  | z-score (표준편차 0일 경우 1.0으로 대체)                                                                                                              |

---

## 4. 학습 설정

| 항목                | 값                                                   |
| ----------------- | --------------------------------------------------- |
| Optimizer         | AdamW (lr=0.001, weight_decay=1e-4)                 |
| Batch Size        | 64                                                  |
| Epochs            | 최대 1000 (EarlyStopping: patience=50, min_delta=0.1) |
| Gradient Clipping | 5.0                                                 |
| Dropout           | 0 (비활성화)                                            |

### 손실 함수 구성

$$
L_{total} = 0.6L_{SL} + 0.3L_{LL} + 0.1L_{cons} + 0.02L_{tv}
$$

* **L_SL:** 예측된 SL_post와 실제 SL_post 간의 MAE (레벨별 λ 가중치 적용)
  λ = [0.133, 0.114, 0.197, 0.208, 0.348]
* **L_LL:** 예측된 LL_post와 실제 LL_post 간의 MAE
* **L_cons:** ΔLL_aux와 ΣΔSL 간의 일관성 손실 (detach 적용)
* **L_tv:** ΔSL 간 인접 smoothness 손실 (0.02 적용)

---

## 5. 실험 결과

### 5.1 최적 성능 (Test MAE 최소 기준)

| 항목                     | 결과                                  |
| ---------------------- | ----------------------------------- |
| **Test MAE (LL_post)** | **6.26°**                           |
| Test MAE (SL_post_ALL) | 3.50°                               |
| Segment별 MAE (SL₁~SL₅) | [2.67°, 2.72°, 3.66°, 3.72°, 4.73°] |
| Test Total Loss        | 4.195                               |
| Best Validation Loss   | 4.499                               |
| 최적 Epoch               | 21                                  |

### 5.2 하이퍼파라미터

| 항목                       | 값                                   |
| ------------------------ | ----------------------------------- |
| Input Dimension          | 64                                  |
| replicate_globals        | False                               |
| Learning Rate            | 0.001                               |
| Alpha                    | 0.01                                |
| Batch Size               | 64                                  |
| Dropout                  | 0                                   |
| w_SL / w_LL / w_C / w_TV | 0.6 / 0.3 / 0.1 / 0.02              |
| λ (Segment 가중치)          | [0.133, 0.114, 0.197, 0.208, 0.348] |

---

## 6. RMSE → MAE 전환 이유

1. **이상치에 대한 민감도 완화**
   RMSE는 제곱 오차로 인해 일부 큰 오차가 전체 지표를 과도하게 왜곡할 수 있습니다. MAE는 모든 샘플에 균등 가중치를 부여하여 보다 안정적인 학습을 제공합니다.

2. **학습 안정성 향상**
   다중 손실 구조(ΔSL, ΔLL consistency) 환경에서 L1Loss(MAE)는 RMSE보다 수렴이 빠르고 학습이 안정적입니다.

3. **임상적 해석 용이성**
   임상에서는 각도의 절댓값 오차(°)로 해석하기 때문에 MAE가 더 직관적입니다.

---

## 7. X-ray 촬영 각도 문제 및 평가 정책

* **문제 인식:** 일부 X-ray 데이터는 수술 전후 촬영 각도 차이로 인해 동일한 뼈 구조임에도 각도 차이가 발생했습니다.
* **원인:** 이는 촬영자의 세팅 오류로 발생한 데이터 획득상의 문제이며, 모델이 학습으로 교정할 수 없는 외생적 요인입니다.
* **가정:** 모델 학습 및 평가 시, 수술 전후 촬영 각도가 동일하다고 가정합니다.
* **실험적 보정 시도:** 과거 절댓값 기반 오차 제거 방식을 통해 2.2° 결과를 얻은 적이 있으나, ±오차의 방향성을 제거하여 논리적으로 잘못된 접근임이 확인되었습니다.
* **최종 정책:**

  1. 촬영각도 보정(절댓값 필터링)은 적용하지 않음.
  2. 모든 결과는 원본 측정값 기준 MAE로 계산.
  3. 촬영오차 문제는 향후 별도의 데이터 수집 및 분석을 통해 통제 예정.

---

