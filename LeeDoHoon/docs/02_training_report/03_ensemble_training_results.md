# 03. 앙상블 모델 학습 결과 (Ensemble Training Results)

> **작성자**: 이도훈 (LDH)  
> **작성일**: 2025-12-18  
> **버전**: v1.0

---

## 1. 학습 개요

### 1.1 앙상블 구성
| 구성 요소 | 설명 |
|----------|------|
| **Base Model 1** | LightGBM (GBDT) |
| **Base Model 2** | CatBoost (Ordered Boosting) |
| **앙상블 방법** | Soft Voting, Weighted Average, Stacking |

### 1.2 앙상블 방법 설명
| 방법 | 설명 |
|------|------|
| **Soft Voting** | 두 모델의 예측 확률을 단순 평균 |
| **Weighted Average** | Validation AUC 기반 가중 평균 |
| **Stacking** | Logistic Regression 메타러너로 예측 결합 |

---

## 2. 개별 모델 성능 (Base Models)

### 2.1 Validation Set 성능

| 지표 | LightGBM | CatBoost | 우수 모델 |
|------|----------|----------|-----------|
| **ROC-AUC** | 0.9884 | 0.9881 | LightGBM ✅ |
| **PR-AUC** | 0.9279 | 0.9241 | LightGBM ✅ |
| **Recall** | 0.9407 | 0.9424 | CatBoost ✅ |
| **Precision** | 0.6257 | 0.6130 | LightGBM ✅ |
| **F1-Score** | 0.7515 | 0.7428 | LightGBM ✅ |

### 2.2 Test Set 성능

| 지표 | LightGBM | CatBoost | 우수 모델 |
|------|----------|----------|-----------|
| **ROC-AUC** | 0.9887 | 0.9884 | LightGBM ✅ |
| **PR-AUC** | 0.9277 | 0.9237 | LightGBM ✅ |
| **Recall** | 0.9413 | 0.9448 | CatBoost ✅ |
| **Precision** | 0.6199 | 0.6061 | LightGBM ✅ |
| **F1-Score** | 0.7475 | 0.7384 | LightGBM ✅ |

---

## 3. 앙상블 모델 성능 (Test Set)

### 3.1 앙상블 방법별 성능 비교

| 지표 | Soft Voting | Weighted Avg | Stacking | 최고 성능 |
|------|-------------|--------------|----------|-----------|
| **ROC-AUC** | 0.9886 | 0.9886 | 0.9886 | Stacking ✅ |
| **PR-AUC** | 0.9266 | 0.9266 | 0.9266 | - |
| **Recall** | 0.9442 | 0.9442 | 0.9474 | - |
| **Precision** | 0.6144 | 0.6144 | 0.5979 | - |
| **F1-Score** | 0.7444 | 0.7444 | 0.7331 | - |

### 3.2 앙상블 가중치 정보

| 앙상블 방법 | LightGBM 가중치 | CatBoost 가중치 |
|------------|-----------------|-----------------|
| Soft Voting | 0.5000 | 0.5000 |
| Weighted Average | 0.9884 | 0.9881 |
| Stacking (Meta Coef) | 4.1052 | 3.9750 |

---

## 4. 전체 모델 성능 비교 (Test Set)

| 순위 | 모델 | ROC-AUC | PR-AUC | Recall | Precision | F1-Score |
|------|------|---------|--------|--------|-----------|----------|
| 1 🥇 | **LightGBM** | 0.9887 | 0.9277 | 0.9413 | 0.6199 | 0.7475 |
| 2 🥈 | **Stacking** | 0.9886 | 0.9266 | 0.9474 | 0.5979 | 0.7331 |
| 3 🥉 | **Weighted Average** | 0.9886 | 0.9266 | 0.9442 | 0.6144 | 0.7444 |
| 4  | **Soft Voting** | 0.9886 | 0.9266 | 0.9442 | 0.6144 | 0.7444 |
| 5  | **CatBoost** | 0.9884 | 0.9237 | 0.9448 | 0.6061 | 0.7384 |

---

## 5. Confusion Matrix (Test Set)

### 5.1 LightGBM

```
              Predicted
              0        1
Actual  0    166,644    10,082
        1    1,026    16,440
```

### 5.2 CatBoost

```
              Predicted
              0        1
Actual  0    166,000    10,726
        1    964    16,502
```

### 5.3 Stacking (Best Ensemble)

```
              Predicted
              0        1
Actual  0    165,599    11,127
        1    919    16,547
```

---

## 6. 결론

### 6.1 최종 모델 선정

| 항목 | 값 |
|------|-----|
| **최고 성능 모델** | LightGBM |
| **ROC-AUC** | 0.9887 |
| **PR-AUC** | 0.9277 |
| **Recall** | 0.9413 |

### 6.2 앙상블 효과 분석

| 비교 항목 | 값 |
|----------|-----|
| LightGBM (단독) ROC-AUC | 0.9887 |
| CatBoost (단독) ROC-AUC | 0.9884 |
| 최고 앙상블 (Stacking) ROC-AUC | 0.9886 |
| **앙상블 향상도** | -0.00%p |

### 6.3 권장 사항

- ⚠️ **단일 모델 사용 권장**: LightGBM이 앙상블보다 우수하거나 동등한 성능
- 앙상블의 추가적인 복잡성 대비 성능 향상이 미미함

---

## 7. 저장된 파일

| 파일 | 경로 | 설명 |
|------|------|------|
| LightGBM | `models/ensemble_lightgbm.txt` | Base Model 1 |
| CatBoost | `models/ensemble_catboost.cbm` | Base Model 2 |
| 앙상블 결과 | `models/ensemble_results.json` | 전체 결과 |
| 리포트 | `docs/02_training_report/03_ensemble_training_results.md` | 이 문서 |

---

> **📌 다음 단계**: Risk Score 기반 비즈니스 액션 매핑 또는 SHAP 분석
