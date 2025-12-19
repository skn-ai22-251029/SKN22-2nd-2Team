# 06. 최종 모델 선정 결과

> **작성자**: 이도훈 (LDH)  
> **작성일**: 2025-12-19  
> **목적**: 평가 지표 기준 최적 모델 선정 및 배포 준비

---

## 1. 모델 선정 개요

### 1.1 선정 기준

| 평가 지표 | 가중치 | 목적 |
|-----------|--------|------|
| **ROC-AUC** | 높음 | 전체적인 분류 성능 |
| **PR-AUC** | 높음 | 불균형 데이터에서의 성능 |
| **Recall** | **최우선** | 이탈 고객 최대한 탐지 (비즈니스 목표) |
| **Precision** | 중간 | False Positive 최소화 |
| **F1-Score** | 중간 | Recall과 Precision의 균형 |

### 1.2 비교 대상 모델

| 모델 | 학습 데이터 | 최적화 목표 |
|------|------------|------------|
| Logistic Regression | v3 | Baseline |
| LightGBM | v3 | 일반 성능 |
| CatBoost | v3 | Recall 최적화 |
| Ensemble (LightGBM + CatBoost) | v3 | 앙상블 성능 |

---

## 2. 모델별 성능 비교 (Test Set)

### 2.1 전체 지표 비교

| 모델 | ROC-AUC | PR-AUC | Recall | Precision | F1-Score | 선택 |
|------|---------|--------|--------|-----------|----------|------|
| **Logistic Regression** | 0.9474 | 0.7498 | 0.8843 | 0.5134 | 0.6496 | |
| **LightGBM** | 0.9887 | 0.9277 | 0.9413 | 0.6199 | 0.7475 | |
| **CatBoost (Recall 최적화)** | 0.9883 | 0.9222 | **0.9514** | 0.5818 | 0.7220 | ✅ |
| **Ensemble (Stacking)** | 0.9886 | 0.9266 | 0.9474 | 0.5979 | 0.7331 | |

### 2.2 Recall 기준 비교

| 모델 | Recall | 이탈 고객 탐지율 | 비고 |
|------|--------|-----------------|------|
| Logistic Regression | 0.8843 | 88.4% | Baseline |
| LightGBM | 0.9413 | 94.1% | 일반 최적화 |
| **CatBoost (Recall 최적화)** | **0.9514** | **95.1%** | **최고 성능** ✅ |
| Ensemble (Stacking) | 0.9474 | 94.7% | 앙상블 |

---

## 3. 최종 모델 선정

### 3.1 선정 모델

**CatBoost (Recall 최적화 모델)**

### 3.2 선정 사유

1. **Recall 최고 성능**: 95.1%로 모든 모델 중 가장 높음
   - 이탈 고객의 95.1%를 사전에 탐지 가능
   - 비즈니스 목표(이탈 고객 최대한 탐지)에 가장 부합

2. **ROC-AUC 우수**: 0.9883으로 LightGBM(0.9887)과 거의 동등
   - 전체적인 분류 성능이 우수함

3. **PR-AUC 우수**: 0.9222로 불균형 데이터에서도 안정적 성능

4. **앙상블 대비 단순성**: 앙상블 대비 성능 차이가 미미하며, 단일 모델로 운영이 간단함

### 3.3 Trade-off 분석

| 항목 | 장점 | 단점 |
|------|------|------|
| **Recall 95.1%** | 이탈 고객 최대한 탐지 | False Positive 증가 |
| **Precision 58.2%** | - | 마케팅 비용 증가 가능 |
| **Threshold 0.58** | Recall/Precision 균형 | 비즈니스 요구에 따라 조정 필요 |

---

## 4. 최종 모델 성능 상세

### 4.1 Test Set 성능 (Threshold = 0.58)

| 지표 | 값 |
|------|-----|
| **ROC-AUC** | 0.9883 |
| **PR-AUC** | 0.9222 |
| **Recall** | **0.9514** |
| **Precision** | 0.5818 |
| **F1-Score** | 0.7220 |
| **Specificity** | 0.9324 |
| **Accuracy** | 0.9351 |

### 4.2 Confusion Matrix

```
              Predicted
              0        1
Actual  0    164,779    11,947
        1    848    16,618
```

**해석**:
- **True Positive**: 16,618명 (이탈 예측 → 실제 이탈)
- **False Negative**: 848명 (유지 예측 → 실제 이탈) ← 놓친 이탈 고객
- **False Positive**: 11,947명 (이탈 예측 → 실제 유지) ← 오탐
- **True Negative**: 164,779명 (유지 예측 → 실제 유지)

### 4.3 비즈니스 임팩트

- **탐지된 이탈자**: 16,618명 / 17,466명 (95.1%)
- **놓친 이탈자**: 848명 (4.9%)
- **오탐 고객**: 11,947명 (마케팅 비용 발생)

---

## 5. 모델 배포 정보

### 5.1 저장 위치

```
03_trained_model/
├── catboost_recall_selected.cbm    # 모델 파일
├── feature_cols.json                # 피처 목록
├── recall_selected_results.json    # 학습 결과
├── inference.py                     # 추론 파이프라인
└── model_metadata.md               # 모델 메타데이터
```

### 5.2 모델 하이퍼파라미터

| 파라미터 | 값 |
|----------|-----|
| `learning_rate` | 0.05 |
| `depth` | 6 |
| `l2_leaf_reg` | 3.0 |
| `iterations` | 500 |
| `best_iteration` | 493 |
| `scale_pos_weight` | 15.18 |
| `optimal_threshold` | 0.58 |

### 5.3 피처 정보

- **총 피처 수**: 35개
- **제거된 중복 피처**: 2개 (`recency_secs_ratio`, `recency_songs_ratio`)
- **주요 피처**: `days_to_expire` (23.24), `payment_method_last` (14.88), `cancel_count` (11.91)

---

## 6. 사용 가이드

### 6.1 Inference 코드 사용법

```python
from inference import load_predictor
import pandas as pd

# 모델 로드
predictor = load_predictor()

# 데이터 로드
df = pd.read_parquet("data/kkbox_train_feature_v3.parquet")

# 예측 실행
results = predictor.predict_with_metadata(df)

# 결과 확인
print(results.head())
```

### 6.2 Threshold 조정

```python
# Recall 우선 (더 많은 이탈 고객 탐지)
results = predictor.predict_with_metadata(df, threshold=0.5)

# Precision 우선 (정확도 향상)
results = predictor.predict_with_metadata(df, threshold=0.65)
```

---

## 7. 모델 비교 요약

### 7.1 각 모델의 강점

| 모델 | 강점 | 적합한 시나리오 |
|------|------|----------------|
| **Logistic Regression** | 해석 가능성, 빠른 학습 | Baseline, 빠른 프로토타이핑 |
| **LightGBM** | 높은 ROC-AUC, 균형잡힌 성능 | 일반적인 이탈 예측 |
| **CatBoost (선정)** | **최고 Recall, 안정적 성능** | **이탈 고객 최대한 탐지** ✅ |
| **Ensemble** | 다양한 모델 조합 | 성능 향상이 필요한 경우 |

### 7.2 최종 권장사항

1. **프로덕션 배포**: CatBoost (Recall 최적화) 모델 사용
2. **Threshold 조정**: 비즈니스 요구사항에 따라 0.5~0.65 범위에서 조정
3. **모니터링**: False Positive 비율과 마케팅 비용을 지속적으로 모니터링

---

## 8. 결론

### 8.1 최종 선정 모델

**CatBoost (Recall 최적화 모델)**

### 8.2 핵심 성과

- ✅ **Recall 95.1%**: 이탈 고객의 95.1%를 사전에 탐지
- ✅ **ROC-AUC 0.9883**: 우수한 전체 분류 성능
- ✅ **PR-AUC 0.9222**: 불균형 데이터에서도 안정적 성능
- ✅ **배포 준비 완료**: Inference 파이프라인 및 문서화 완료

### 8.3 다음 단계

1. ✅ 모델 파일 저장 (`03_trained_model/`)
2. ✅ Inference 파이프라인 작성 (`inference.py`)
3. ✅ 모델 메타데이터 문서화 (`model_metadata.md`)
4. ⏭️ Streamlit App 개발
5. ⏭️ 최종 문서화 및 정리

---

> **핵심 메시지**: CatBoost (Recall 최적화) 모델을 선택하여 이탈 고객의 **95.1%**를 탐지할 수 있습니다. 비즈니스 목표에 맞게 threshold를 조정하여 사용하세요.

