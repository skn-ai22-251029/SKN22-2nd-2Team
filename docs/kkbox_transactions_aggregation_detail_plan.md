
# KKBox Churn Prediction  
## transactions_v2 집계 세부 설계 문서

본 문서는 **KKBox Churn Prediction 프로젝트**에서  
`transactions_v2.csv`를 **데이터 누수 없이, 일관된 기준으로 집계**하기 위한  
세부 규칙을 정의한 공식 설계 문서이다.

---

## 1. 공통 전제

### 1.1 기준 시점 (T)

- **T = 2017-04-01**
- 모든 집계는 반드시 다음 조건을 만족해야 함:
  - `transaction_date <= T`
  - T 이후 정보는 feature로 사용 금지

### 1.2 집계 단위

- **msno 기준 1 row**
- 집계 결과는 user-level feature table로 병합됨

---

## 2. 트랜잭션 기본 정의

### 2.1 결제(Payment)의 정의

다음 조건을 만족하는 트랜잭션만 **결제**로 간주한다.

- `actual_amount_paid > 0`

※ cancel, 무료 체험, 0원 결제는 결제 건수 및 금액 집계에서 제외

---

### 2.2 Cancel의 정의

다음 중 하나 이상을 만족하면 **cancel 이벤트**로 정의한다.

- `is_cancel == 1`
- 또는 자동 갱신 상태(`is_auto_renew`)가 해제된 트랜잭션

※ 프로젝트 내에서 cancel 정의는 반드시 고정되어야 함

---

## 3. 상태 기반 피처 (가장 중요)

> 목적: **현재 이탈 직전 상태인지 판단**

모든 상태 기반 피처는  
**transaction_date 기준 가장 마지막 트랜잭션(row)**을 기준으로 산출한다.

---

### 3.1 days_since_last_payment

**정의**
```
days_since_last_payment = T - 마지막 결제(transaction_date)
```

**집계 기준**
- `actual_amount_paid > 0` 조건 만족
- transaction_date 최대값 사용

**결측 발생 조건**
- 결제 이력이 전혀 없는 사용자

**권장 처리**
- 결측 → 999 (또는 충분히 큰 값)
- 보조 피처: `has_ever_paid` (0/1)

---

### 3.2 days_since_last_cancel

**정의**
```
days_since_last_cancel = T - 마지막 cancel 발생일
```

**집계 기준**
- 2.2 cancel 정의를 만족하는 트랜잭션 중
- transaction_date 최대값 사용

**결측 발생 조건**
- cancel 이력이 전혀 없는 사용자

**권장 처리**
- 결측 → 999
- 보조 피처: `has_ever_cancelled` (0/1)

※ cancel 계열 피처는 **결측 자체가 긍정적 신호**

---

### 3.3 is_auto_renew_last

**정의**
- 마지막 트랜잭션의 `is_auto_renew` 값

**결측 발생 조건**
- 트랜잭션 이력이 없는 사용자

**권장 처리**
- 결측 → 0
- 보조 피처: `has_transaction`

---

### 3.4 last_plan_days

**정의**
- 마지막 트랜잭션의 `payment_plan_days`

**결측 발생 조건**
- 결제 이력 없는 사용자

**권장 처리**
- 결측 → 0
- 보조 피처: `is_free_user`

---

### 3.5 last_payment_method

**정의**
- 마지막 트랜잭션의 `payment_method_id`

**결측 발생 조건**
- 결제 이력 없는 사용자

**권장 처리**
- 결측 → "NONE" 카테고리

---

## 4. 누적 히스토리 피처 (전체 기간)

> 기간: **2015-01-01 ~ 2017-03-31**

---

### 4.1 total_payment_count

**정의**
- 결제(`actual_amount_paid > 0`) 트랜잭션 수

**결측 여부**
- 없음 (0 가능)

---

### 4.2 total_amount_paid

**정의**
- `actual_amount_paid` 합계

**결측 여부**
- 없음 (0 가능)

---

### 4.3 avg_amount_per_payment

**정의**
```
avg_amount_per_payment = total_amount_paid / total_payment_count
```

**주의**
- `total_payment_count == 0` → division by zero

**권장 처리**
- count = 0 → 0으로 치환

---

### 4.4 unique_plan_count

**정의**
- `payment_plan_days` distinct count

**결측 여부**
- 없음 (0 가능)

---

### 4.5 subscription_months_est

**정의**
```
subscription_months_est = sum(payment_plan_days) / 30
```

**주의**
- 월 환산 기준(30일)을 문서상 고정

---

## 5. 제한적 Recency 집계 (선택)

> 이벤트 희소성을 고려하여 제한적으로 사용

---

### 5.1 payment_count_last_30d

**정의**
- 기간: `(T-30, T]`
- 조건: `actual_amount_paid > 0`

---

### 5.2 payment_count_last_90d

**정의**
- 기간: `(T-90, T]`
- 조건: `actual_amount_paid > 0`

---

## 6. 데이터 누수 체크리스트

- [ ] transaction_date <= 2017-03-31
- [ ] cancel 정의 일관성 유지
- [ ] churn 이후 정보 미사용
- [ ] msno 기준 1 row 보장

---

## 7. 요약

- 상태 기반 피처는 **가장 강력한 churn 신호**
- 결측치는 제거 대상이 아니라 **의미 있는 정보**
- transactions는 **상태 + 누적 중심 설계**
- 짧은 window bin은 사용 비권장

