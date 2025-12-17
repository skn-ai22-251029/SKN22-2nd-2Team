# GitHub Projects Issues 목록 (최종 플랜 기준)

본 문서는 **가입 고객 이탈 예측 프로젝트**를 GitHub Projects에 등록하기 위한  
최종 Issue(EPIC) 목록을 정리한 문서이다.

각 Issue는 **EPIC 1개 단위**로 구성되며,  
GitHub Issue 생성 시 제목과 설명을 그대로 복사하여 사용할 수 있다.

---

## Issue 1. Problem Definition & Prediction Frame

### Title
```
[EPIC] Problem Definition & Prediction Frame
```

### Description
```
- churn 예측 문제 정의
- 타겟 변수(is_churn) 정의
- 예측 시점(T) 정의
- 관측 윈도우 / 예측 윈도우 개념 및 기준 설정
- 데이터 누수 방지 관점에서 예측 프레임 명확화
```

### Deliverable
```
/01_preprocessing_report/00_problem_definition.md
```

---

## Issue 2. Data Schema Understanding & Minimal EDA

### Title
```
[EPIC] Data Schema Understanding & Minimal EDA
```

### Description
```
- members_v3 / user_logs_v2 / transactions_v2 / train_v2 스키마 정리
- msno 기준 테이블 관계 정의
- 테이블별 날짜 범위(min/max) 확인
- 타겟(is_churn) 분포 확인
- 전처리를 위한 최소 수준의 데이터 점검 수행
```

### Deliverable
```
/01_preprocessing_report/01_schema_overview.md
/01_preprocessing_report/02_data_dictionary.md
```

---

## Issue 3. Data Preprocessing & Feature Engineering

### Title
```
[EPIC] Data Preprocessing & Feature Engineering
```

### Description
```
- 결측치 및 이상치 처리 규칙 정의
- user_logs_v2 (30일 관측 윈도우) 집계 feature 생성
- transactions_v2 기반 결제/구독 feature 생성
- members_v3 정적 정보 병합 및 인코딩
- 학습용 데이터셋 생성 (train/valid/test)
- 전처리 결과 sanity check
```

### Deliverable
```
/01_preprocessing_report/03_preprocessing_summary.md
/01_preprocessing_report/04_feature_dictionary.md
전처리 완료 데이터셋
```

---

## Issue 4. Machine Learning Model Training & Evaluation

### Title
```
[EPIC] Machine Learning Model Training & Evaluation
```

### Description
```
- Logistic Regression baseline 모델 학습
- Tree 기반 모델(LightGBM 또는 CatBoost) 학습
- 동일한 데이터 분할 기준으로 모델 성능 비교
- ROC-AUC, PR-AUC, Recall 기반 평가
```

### Deliverable
```
/02_training_report/01_ml_training_results.md
모델별 평가 지표 결과
```

---

## Issue 5. Deep Learning Model Training & Evaluation

### Title
```
[EPIC] Deep Learning Model Training & Evaluation
```

### Description
```
- Tabular 데이터 기반 MLP 모델 설계 및 학습
- 정규화, 드롭아웃, 조기 종료 적용
- ML 모델과 동일한 지표 기준 성능 비교
```

### Deliverable
```
/02_training_report/02_dl_training_results.md
학습 곡선 및 평가 결과
```

---

## Issue 6. Best Model Selection & Model Packaging

### Title
```
[EPIC] Best Model Selection & Model Packaging
```

### Description
```
- 평가 지표 기준 최적 모델 선정
- 학습된 모델 저장
- 전처리부터 추론까지 일관된 inference 코드 작성
```

### Deliverable
```
/03_trained_model/model_file
/03_trained_model/inference.py
/03_trained_model/model_metadata.md
```

---

## Issue 7. Streamlit App Development

### Title
```
[EPIC] Streamlit App Development
```

### Description
```
- 학습된 모델 로드
- 입력 데이터 기반 churn 예측 수행
- 예측 확률 출력
```

### Deliverable
```
app.py
README.md (실행 방법 포함)
```

---

## Issue 8. Documentation & Finalization

### Title
```
[EPIC] Documentation & Finalization
```

### Description
```
- 프로젝트 전체 README 정리
- 분석 흐름 및 결과 요약
- 데이터 및 모델 한계 정리
```

### Deliverable
```
README.md
```

---

## Issue 진행 권장 순서

```
Issue 1
 → Issue 2
 → Issue 3
 → Issue 4
 → Issue 5
 → Issue 6
 → Issue 7
 → Issue 8
```

본 문서는 GitHub Projects 이슈 등록용 공식 문서로 사용한다.
