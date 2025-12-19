# 모델 메타데이터 (Model Metadata)

> **작성자**: 이도훈 (LDH)  
> **작성일**: 2025-12-19  
> **모델 버전**: v1.0 (Recall 최적화)

---

## 1. 모델 개요

### 1.1 기본 정보

| 항목 | 값 |
|------|-----|
| **모델 유형** | CatBoost Classifier |
| **목적** | 고객 이탈 예측 (Churn Prediction) |
| **최적화 목표** | Recall 최대화 (이탈 고객 최대한 탐지) |
| **모델 파일** | `catboost_recall_selected.cbm` |
| **학습 데이터** | kkbox_train_feature_v3 (중복 피처 제거) |
| **학습 일자** | 2025-12-18 |

### 1.2 성능 지표 (Test Set)

| 지표 | 값 |
|------|-----|
| **ROC-AUC** | 0.9883 |
| **PR-AUC** | 0.9222 |
| **Recall** | **0.9514** (95.1%) |
| **Precision** | 0.5818 |
| **F1-Score** | 0.7220 |
| **Specificity** | 0.9324 |
| **최적 Threshold** | 0.58 |

---

## 2. 모델 하이퍼파라미터

| 파라미터 | 값 |
|----------|-----|
| `loss_function` | Logloss |
| `eval_metric` | AUC |
| `learning_rate` | 0.05 |
| `depth` | 6 |
| `l2_leaf_reg` | 3.0 |
| `iterations` | 500 |
| `early_stopping_rounds` | 50 |
| `scale_pos_weight` | 15.18 |
| `random_seed` | 42 |
| `best_iteration` | 493 |

---

## 3. 피처 정보

### 3.1 피처 개수
- **총 피처 수**: 35개
- **제거된 중복 피처**: 2개
  - `recency_secs_ratio` (= `secs_trend_w7_w30`)
  - `recency_songs_ratio` (= `songs_trend_w7_w30`)

### 3.2 피처 목록
피처 목록은 `feature_cols.json` 파일에 저장되어 있습니다.

### 3.3 주요 피처 (Top 10)

| 순위 | 피처 | Importance |
|------|------|------------|
| 1 | `days_to_expire` | 23.24 |
| 2 | `payment_method_last` | 14.88 |
| 3 | `cancel_count` | 11.91 |
| 4 | `total_payment` | 11.55 |
| 5 | `auto_renew_rate` | 10.02 |
| 6 | `has_cancelled` | 6.74 |
| 7 | `tenure_days` | 2.94 |
| 8 | `avg_payment` | 2.90 |
| 9 | `transaction_count` | 2.74 |
| 10 | `avg_list_price` | 2.72 |

---

## 4. 사용 방법

### 4.1 Python 코드 예제

```python
from pathlib import Path
from inference import load_predictor
import pandas as pd

# 1. 모델 로드
predictor = load_predictor()

# 2. 데이터 로드 (전처리된 데이터)
df = pd.read_parquet("data/kkbox_train_feature_v3.parquet")

# 3. 예측 실행
results = predictor.predict_with_metadata(df)

# 4. 결과 확인
print(results.head())
```

### 4.2 배치 예측

```python
from inference import predict_batch

# 배치 예측
results = predict_batch(df, threshold=0.58)
```

### 4.3 커스텀 Threshold 사용

```python
# 기본 threshold (0.58) 대신 다른 값 사용
results = predictor.predict_with_metadata(df, threshold=0.5)
```

---

## 5. 입력 데이터 요구사항

### 5.1 필수 컬럼
모델은 다음 35개 피처를 필요로 합니다 (자세한 목록은 `feature_cols.json` 참조):

- User Logs 피처: `total_songs`, `total_secs`, `num_25_sum`, `num_50_sum`, `num_75_sum`, `num_985_sum`, `num_100_sum`, `num_unq_sum`, `active_days`, `skip_ratio`, `complete_ratio`, `partial_ratio`, `avg_songs_per_day`, `avg_secs_per_day`, `listening_variety`, `avg_song_length`
- Transaction 피처: `total_payment`, `avg_payment`, `avg_list_price`, `cancel_count`, `auto_renew_rate`, `avg_discount_rate`, `transaction_count`, `is_auto_renew_last`, `plan_days_last`, `payment_method_last`, `days_to_expire`, `has_cancelled`
- Member 피처: `city`, `age`, `registered_via`, `tenure_days`, `gender_female`, `gender_male`, `gender_unknown`

### 5.2 데이터 전처리
- 입력 데이터는 학습 시 사용된 전처리 파이프라인을 거쳐야 합니다.
- 결측치 처리, 범주형 인코딩 등이 일관되게 적용되어야 합니다.
- 자세한 전처리 방법은 `src/preprocessing_ldh.py` 참조

### 5.3 데이터 형식
- **입력**: pandas DataFrame
- **msno 컬럼**: 선택사항 (있으면 결과에 포함)
- **is_churn 컬럼**: 선택사항 (예측 시 무시됨)

---

## 6. 출력 형식

### 6.1 예측 결과 컬럼

| 컬럼 | 설명 | 타입 |
|------|------|------|
| `msno` | 사용자 ID | str/int |
| `churn_proba` | 이탈 확률 (0~1) | float |
| `churn_pred` | 이탈 예측 (0: 유지, 1: 이탈) | int |

### 6.2 Threshold 해석

- **Threshold = 0.58** (기본값)
  - Recall: 95.1% (이탈 고객의 95.1% 탐지)
  - Precision: 58.2%
  - False Positive가 많아질 수 있으나, 이탈 고객을 최대한 놓치지 않음

- **Threshold 조정 가이드**
  - **Recall 우선**: Threshold 낮춤 (0.4~0.5) → 더 많은 이탈 고객 탐지
  - **Precision 우선**: Threshold 높임 (0.6~0.7) → 정확도 향상, 일부 이탈 고객 놓칠 수 있음

---

## 7. 의존성 (Dependencies)

### 7.1 필수 패키지

```
pandas >= 1.5.0
numpy >= 1.20.0
catboost >= 1.0.0
```

### 7.2 설치 방법

```bash
pip install pandas numpy catboost
```

---

## 8. 파일 구조

```
03_trained_model/
├── catboost_recall_selected.cbm    # 모델 파일
├── feature_cols.json                # 피처 목록
├── recall_selected_results.json    # 학습 결과 및 메타데이터
├── inference.py                     # 추론 파이프라인
└── model_metadata.md               # 이 문서
```

---

## 9. 주의사항

### 9.1 데이터 일관성
- 입력 데이터는 학습 시 사용된 전처리 파이프라인을 거쳐야 합니다.
- 피처 순서와 형식이 일치해야 합니다.

### 9.2 Threshold 선택
- 기본 threshold (0.58)는 Recall 95% 목표로 설정되었습니다.
- 비즈니스 요구사항에 따라 threshold를 조정할 수 있습니다.

### 9.3 성능 고려사항
- 모델은 Recall 최적화를 목표로 하므로 Precision이 상대적으로 낮습니다.
- False Positive가 많아질 수 있으므로, 마케팅 비용을 고려하여 threshold를 조정하세요.

---

## 10. 참고 자료

- 학습 리포트: `docs/02_training_report/05_recall_model_selection.md`
- 전처리 파이프라인: `src/preprocessing_ldh.py`
- 학습 스크립트: `src/train_recall_optimized.py`

---

## 11. 문의

모델 관련 문의사항이 있으시면 프로젝트 팀에 연락해주세요.

---

> **핵심 메시지**: 이 모델은 이탈 고객의 **95.1%**를 탐지할 수 있습니다. 비즈니스 목표에 맞게 threshold를 조정하여 사용하세요.

