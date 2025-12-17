# 가입 고객 이탈 예측 프로젝트 플랜 (KKBox 데이터셋 기반, 최종 정제본)

본 프로젝트는 **KKBox Churn Prediction 데이터셋에 포함된 정보만 사용**하여  
가입 고객의 이탈(churn)을 예측하는 머신러닝/딥러닝 모델을 구축하고,  
학습·평가·최적 모델 선정·배포(Streamlit 앱)까지 수행하는 것을 목표로 한다.

탐색적 데이터 분석(EDA)은 별도 단계로 분리하지 않으며,  
**전처리·예측 프레임·피처 설계를 위한 최소한의 데이터 점검만 수행**한다.

---

## 프로젝트 목표

- 가입 고객 이탈(churn) 예측 문제 정의
- 데이터 전처리 및 feature engineering
- 머신러닝 모델과 딥러닝 모델 학습 및 성능 비교
- 최적 모델 선정 및 재현 가능한 형태로 저장
- 예측 결과를 확인할 수 있는 Streamlit 앱 구현

---

## 사용 데이터셋

- `train_v2.csv` : 사용자별 이탈 라벨 (`is_churn`)
- `user_logs_v2.csv` : 사용자 일별 서비스 이용 로그
- `transactions_v2.csv` : 결제 및 구독 이력
- `members_v3.csv` : 사용자 기본 정보

모든 분석과 모델링은 위 데이터셋의 컬럼과 시간 정보에 한정한다.

---

## EPIC 1. 문제 정의 및 예측 프레임 설정

### Objective
- churn 예측 문제를 데이터 관점에서 명확히 정의

### Tasks
- 타겟 변수 정의: `is_churn`
- 예측 시점(T) 정의
- 관측 윈도우 설정  
  - T 이전 일정 기간의 로그/거래 데이터를 집계하여 feature 생성
- 예측 윈도우 설정  
  - T 이후 일정 기간 동안의 이탈 여부를 기준으로 라벨 결정

### 포함되는 최소 데이터 점검
- 각 테이블의 시간 범위(min/max date) 확인
- 예측 시점 이후 정보가 feature로 포함되지 않는지 점검

### Deliverables
- `/01_preprocessing_report/00_problem_definition.md`

---

## EPIC 2. 데이터 구조 이해 및 스키마 정리 (최소 EDA 포함)

### Objective
- 데이터셋 구조와 컬럼 의미를 정확히 이해하고 전처리 기준 확정

### Tasks
- 각 테이블의 컬럼, 타입, 의미 정리
- `msno` 기준 테이블 조인 관계 정리
- 테이블별 데이터 기간 확인
- 결측치 및 이상치 구조 파악  
  - 예: `members.bd`, `gender` 결측 구조
- 타겟 분포(`is_churn`) 확인

### Deliverables
- `/01_preprocessing_report/01_schema_overview.md`
- `/01_preprocessing_report/02_data_dictionary.md`

---

## EPIC 3. 데이터 전처리 및 Feature Engineering

### Objective
- Raw 데이터를 학습 가능한 user-level feature table로 변환

### Tasks
- 결측치 및 이상치 처리 규칙 정의
- `user_logs_v2` 집계  
  - 최근 활동량, 재생 횟수, 총 재생 시간, 활동 일수 등
  - 대용량 로그는 user 단위로 선집계 후 병합
- `transactions_v2` 집계  
  - 결제 횟수, 최근 결제 정보, 자동 갱신 여부, 취소 이력
- `members_v3` 병합 및 범주형 인코딩
- 전처리 결과 sanity check  
  - 집계값 범위 확인
  - 비정상 값 여부 점검
- 학습 데이터 분리(train/validation/test)

### Deliverables
- `/01_preprocessing_report/03_preprocessing_summary.md`
- `/01_preprocessing_report/04_feature_dictionary.md`
- 전처리 완료 데이터셋

---

## EPIC 4. 머신러닝 모델 학습 및 평가

### Objective
- 전통적인 머신러닝 모델 기반 churn 예측

### Tasks
- Baseline 모델: Logistic Regression
- Tree 기반 모델: LightGBM 또는 CatBoost 중 선택
- 동일한 데이터 분할 기준으로 모델 학습 및 비교
- 평가 지표 계산  
  - ROC-AUC  
  - PR-AUC  
  - Recall

### Deliverables
- `/02_training_report/01_ml_training_results.md`
- 모델별 평가 지표 파일

---

## EPIC 5. 딥러닝 모델 학습 및 평가

### Objective
- Tabular 데이터 기반 딥러닝 모델 성능 확인

### Tasks
- MLP 기반 딥러닝 모델 구현
- 정규화, 드롭아웃, 조기 종료 적용
- 머신러닝 모델과 동일한 지표로 성능 비교

### Deliverables
- `/02_training_report/02_dl_training_results.md`
- 학습 곡선 및 평가 지표

---

## EPIC 6. 최적 모델 선정 및 저장

### Objective
- 가장 성능이 우수한 모델을 최종 모델로 확정

### Tasks
- 평가 지표 기준으로 최적 모델 선정
- 최종 모델 저장
- 전처리부터 추론까지 일관된 추론 코드 작성

### Deliverables
- `/03_trained_model/model_file`
- `/03_trained_model/inference.py`
- `/03_trained_model/model_metadata.md`

---

## EPIC 7. Streamlit 앱 구현

### Objective
- 학습된 모델을 활용해 예측 결과를 확인할 수 있는 앱 구현

### App Scope
- 입력: 사용자 feature 데이터
- 출력: churn 예측 확률
- 모델 및 전처리 파이프라인 로드 후 추론 수행

### Deliverables
- `app.py`
- 실행 방법이 포함된 `README.md`

---

## 산출물 디렉터리 구조 요약

```
/01_preprocessing_report/
  00_problem_definition.md
  01_schema_overview.md
  02_data_dictionary.md
  03_preprocessing_summary.md
  04_feature_dictionary.md

/02_training_report/
  01_ml_training_results.md
  02_dl_training_results.md

/03_trained_model/
  model_file
  inference.py
  model_metadata.md

/app.py
/README.md
```

---

## 범위 제한 원칙

- 데이터셋에 존재하지 않는 정보는 사용하지 않는다.
- 외부 데이터, 비즈니스 액션, 효과 추정은 수행하지 않는다.
- churn 예측 모델 구축과 성능 평가에만 집중한다.

---

## 결론

본 플랜은 KKBox 데이터셋에 포함된 정보만을 사용하여  
가입 고객 이탈 예측 모델을 설계, 학습, 평가, 배포하는 전 과정을 포함한다.  
EDA는 의사결정을 위한 최소한의 데이터 점검으로만 수행하며,  
재현 가능하고 명확한 예측 모델 구축을 목표로 한다.
