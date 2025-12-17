# EPIC 2. Data Schema Overview (KKBox Dataset)

본 문서는 KKBox 가입 고객 이탈 예측 프로젝트의 **EPIC 2 단계**로,  
사용되는 모든 데이터셋의 스키마와 각 컬럼의 의미를 정리한 문서이다.

본 문서는 이후 전처리, feature engineering, 모델 해석 단계에서  
**공식 컬럼 사전(Data Dictionary)** 으로 사용된다.

---

## members_v3.csv

사용자의 기본 정보(정적 속성)를 담고 있는 테이블이다.

| 컬럼 이름 | 설명 |
|----------|------|
| msno | 사용자 ID (User ID). 모든 데이터셋에서 사용자를 식별하는 고유 ID |
| city | 사용자가 거주하는 도시 코드 (익명 처리된 정수 값) |
| bd | 사용자의 나이(Birth Date). 실제 나이와 불일치하거나 이상치가 포함될 수 있음 |
| gender | 사용자의 성별 (male, female 또는 결측) |
| registered_via | 사용자가 KKBox 서비스에 등록한 방법(채널)을 나타내는 코드 (익명 처리된 정수 값) |
| registration_init_time | 사용자가 회원 가입을 한 날짜 (YYYYMMDD 형식) |

---

## user_logs_v2.csv

사용자의 일별 서비스 이용 로그를 기록한 테이블이다.

| 컬럼 이름 | 설명 |
|----------|------|
| msno | 사용자 ID (User ID) |
| date | 사용자 로그가 기록된 날짜 (YYYYMMDD 형식) |
| num_25 | 노래 길이의 25% 미만 재생된 곡의 수 (즉시 건너뛰기/조기 종료) |
| num_50 | 노래 길이의 25% ~ 50% 재생된 곡의 수 |
| num_75 | 노래 길이의 50% ~ 75% 재생된 곡의 수 |
| num_985 | 노래 길이의 75% ~ 98.5% 재생된 곡의 수 |
| num_100 | 노래 길이의 98.5% 이상(거의 끝까지) 재생된 곡의 수 (완료된 청취로 간주) |
| num_unq | 해당 날짜에 재생된 고유한(Unique) 곡의 수 |
| total_secs | 해당 날짜에 음악을 재생한 총 시간 (초 단위) |

---

## transactions_v2.csv

사용자의 결제 및 구독 이력을 기록한 테이블이다.

| 컬럼 이름 | 설명 |
|----------|------|
| msno | 사용자 ID (User ID) |
| payment_method_id | 결제 수단 코드 (익명 처리된 정수 값) |
| payment_plan_days | 결제한 멤버십 플랜 기간 (일 수, 예: 30일, 90일 등) |
| plan_list_price | 플랜의 정가 비용 (통화는 신대만 달러 NTD 기준) |
| actual_amount_paid | 사용자가 실제로 지불한 금액 (할인 등으로 plan_list_price와 다를 수 있음, NTD 기준) |
| is_auto_renew | 자동 갱신 설정 여부 (0 또는 1) |
| transaction_date | 결제(거래)가 이루어진 날짜 (YYYYMMDD 형식) |
| membership_expire_date | 해당 구독의 만료 날짜 (YYYYMMDD 형식) |
| is_cancel | 사용자가 해당 거래에서 멤버십을 취소했는지 여부 (0 또는 1) |

---

## train_v2.csv

모델 학습에 사용되는 이탈 타겟 라벨 테이블이다.

| 컬럼 이름 | 설명 |
|----------|------|
| msno | 사용자 ID (User ID) |
| is_churn | 이탈 여부 (타겟 변수) <br>• 1: 해당 사용자가 2017년 3월 이후 이탈했음을 의미 <br>• 0: 해당 사용자가 2017년 3월 이후에도 구독을 유지했음을 의미 |

---

## 참고 사항

- 모든 테이블은 `msno`를 기준으로 조인된다.
- 날짜 컬럼은 모두 YYYYMMDD 형식의 정수형 데이터이다.
- 본 문서에 정의된 컬럼 의미는 이후 단계에서 변경하지 않는다.
