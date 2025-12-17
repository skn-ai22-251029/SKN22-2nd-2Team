
# KKBox Churn Prediction  
## user_logs_v2 / transactions_v2 집계 설계 문서

본 문서는 **KKBox Churn Prediction 프로젝트**에서  
`user_logs_v2.csv`, `transactions_v2.csv`를 **데이터 누수 없이, 실무/대회 기준에 맞게 집계**하기 위한  
표준화된 설계 플랜을 단계별로 정리한 문서이다.

---

## 1. 예측 프레임 정의 (Prediction Frame)

### 1.1 기준 시점 (T)

- **T = 2017-04-01**
- 모든 feature는 **T 이전 정보만 사용**
- `train_v2.is_churn`은 **T 이후 이탈 여부**를 나타냄

### 1.2 데이터 기간

| 데이터 | 기간 |
|---|---|
| user_logs_v2 | 2017-03-01 ~ 2017-03-31 |
| transactions_v2 | 2015-01-01 ~ 2017-03-31 |

---

## 2. 집계 원칙 (공통)

1. 모든 집계 결과는 **msno 기준 1 row**
2. 시간 조건은 반드시 `date <= T`
3. 집계 결과는 **user-level feature table**로 생성
4. 단일 집계보다 **최근성(Recency)과 변화량**을 중시

---

## 3. user_logs_v2 집계 설계

### 3.1 집계 목적

- 최근 서비스 이용 행태를 통해 **이탈 직전 행동 변화 포착**
- churn의 단기 신호(short-term signal) 추출

### 3.2 멀티 윈도우 전략

단일 30일 집계 대신 **rolling window 집계** 사용

| Window | 기간 |
|---|---|
| W7 | 2017-03-25 ~ 03-31 |
| W14 | 2017-03-18 ~ 03-31 |
| W21 | 2017-03-11 ~ 03-31 |
| W30 | 2017-03-01 ~ 03-31 |

### 3.3 기본 집계 피처 (각 window 공통)

- `num_days_active_wX` : 활동 일수
- `total_secs_wX` : 총 재생 시간
- `num_songs_wX` : 재생 곡 수
- `num_unq_wX` : 고유 곡 수

### 3.4 비율형 피처 (중요)

- `skip_ratio_wX`
- `short_play_ratio_wX`
- `completion_ratio_wX`

### 3.5 변화량 파생 피처

- `total_secs_w7 / total_secs_w30`
- `num_days_active_w7 / num_days_active_w14`

> 목적: **최근 이용 급감 여부 포착**

---

## 4. transactions_v2 집계 설계

### 4.1 집계 목적

- 구독/결제 히스토리를 통해 **고객의 구조적 상태 파악**
- churn의 중·장기 신호(long-term signal) 추출

### 4.2 집계 전략 개요

transactions는 로그와 달리 이벤트가 희소하므로  
**멀티 윈도우보다는 상태 + 누적 중심 집계**를 사용

---

### 4.3 상태 기반 피처 (가장 중요)

마지막 트랜잭션 기준

- `days_since_last_payment`
- `days_since_last_cancel`
- `is_auto_renew_last`
- `last_plan_days`
- `last_payment_method`

> 목적: **현재 이탈 직전 상태인지 판단**

---

### 4.4 누적 히스토리 피처 (전체 기간)

(2015-01-01 ~ 2017-03-31)

- `total_payment_count`
- `total_amount_paid`
- `avg_amount_per_payment`
- `unique_plan_count`
- `subscription_months_est`

---

### 4.5 제한적 Recency 집계 (선택)

- `payment_count_last_30d`
- `payment_count_last_90d`

> 주의: 7일, 14일 bin은 대부분 0 → 사용 비권장

---

## 5. 집계 순서 (권장 파이프라인)

1. **transactions_v2 집계**
   - 상태 → 누적 → recency 순
2. **user_logs_v2 멀티 윈도우 집계**
3. msno 기준 row 수 검증 (1:1 여부)
4. train_v2 + members_v3 병합
5. feature sanity check

---

## 6. 데이터 누수 체크리스트

- [ ] 모든 date / transaction_date <= 2017-03-31
- [ ] churn 이후 정보 사용 여부
- [ ] label(is_churn) 관련 컬럼 미사용
- [ ] window 정의가 미래를 포함하지 않는지

---

## 7. 최종 산출물

- user-level feature table
- 모든 row = 1 msno
- Tree 기반 모델(LightGBM / CatBoost) 학습 가능 구조

---

## 8. 요약

- user_logs_v2 → **멀티 윈도우 + 변화량**
- transactions_v2 → **상태 + 누적 중심**
- 핵심은 **최근성 + 행동 붕괴 신호**
- 데이터 누수 방지가 최우선

