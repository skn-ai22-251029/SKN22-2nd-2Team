# transactions_v2 집계 컬럼 명세서
(최소 탐색 / 검증용)

## 1. 개요

- 데이터 소스: `transactions_v2.csv`
- 집계 단위: user-level (`msno`)
- 기준 시점(T): 2017-04-01
- 사용 데이터 범위: `transaction_date < T`
- 목적: churn 예측을 위한 **결제/구독 상태 및 이력 피처 생성**
- 본 문서는 **EDA 목적이 아닌, 집계 결과 검증 및 설명용 명세서**임

---

## 2. Key Column

| 컬럼명 | 타입 | 설명 |
|------|------|------|
| msno | string | 사용자 고유 ID (join key) |

---

## 3. 상태 기반 피처 (Last Transaction 기준)

> churn 직전 상태를 가장 직접적으로 설명하는 핵심 피처

| 컬럼명 | 타입 | 설명 | 최소 검증 포인트 |
|------|------|------|----------------|
| days_since_last_payment | int | T 기준 마지막 결제까지 경과 일수 | 0 이상, 극단적으로 큰 값 존재 여부 |
| days_since_last_cancel | int | T 기준 마지막 취소까지 경과 일수 | 결측 비율 확인 |
| is_auto_renew_last | int (0/1) | 마지막 거래 기준 자동 갱신 여부 | 0/1 이외 값 존재 여부 |
| last_plan_days | int | 마지막 결제 플랜 기간(일) | 30/90/180 분포 확인 |
| last_payment_method | int | 마지막 결제 수단 ID | 범주 개수 과다 여부 |

---

## 4. 누적 히스토리 피처 (전체 기간)

> 사용자 결제 성향 및 장기 가치 설명용

| 컬럼명 | 타입 | 설명 | 최소 검증 포인트 |
|------|------|------|----------------|
| total_payment_count | int | 전체 결제 횟수 | 0 값 비율 |
| total_amount_paid | float | 전체 누적 결제 금액 | 극단값(outlier) |
| avg_amount_per_payment | float | 평균 결제 금액 | total / count 일관성 |
| unique_plan_count | int | 이용한 플랜 종류 수 | 상한선 확인 |
| subscription_months_est | float | 누적 구독 개월 추정치 | 0 또는 비정상 값 |

---

## 5. Recency 기반 피처 (선택)

> user_logs 윈도우와 직접 정합 목적이 아님  
> 결제 활동의 최근성만 보조적으로 반영

| 컬럼명 | 타입 | 설명 | 최소 검증 포인트 |
|------|------|------|----------------|
| payment_count_last_30d | int | 최근 30일 결제 횟수 | 대부분 0인지 확인 |
| payment_count_last_90d | int | 최근 90일 결제 횟수 | w30 대비 정보 증가 여부 |
| has_payment_last_30d | int (0/1) | 최근 30일 결제 존재 여부 | binary 정상 여부 |

---

## 6. 결측치 정책 요약

- `days_since_last_cancel`
  - 결측 = 취소 이력 없음
- recency 관련 피처
  - 결측 발생 시 0으로 대체
- 상태 기반 피처
  - msno 단위 집계 실패 시 해당 유저 제거 또는 별도 처리

---

## 7. 최소 Sanity Check 리스트 (EDA 대체)

- days_since_last_payment ≥ 0
- last_plan_days ∈ {30, 90, 180, 365}
- is_auto_renew_last ∈ {0, 1}
- total_payment_count ≥ 1 인 유저 비율 확인
- payment_count_last_30d ≠ payment_count_last_90d 인 유저 비율

---

## 8. 비고

- 본 집계는 **user_logs_v2와 다른 시간 해상도를 가짐**
- 윈도우 정합성보다 **T 기준 누수 방지**를 우선함
- 본 결과는 members, train_v2와 병합 후 모델 입력으로 사용됨