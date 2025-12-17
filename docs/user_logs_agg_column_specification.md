# User Logs 집계 데이터 컬럼 명세서

> **작성자**: 이도훈 (LDH)  
> **작성일**: 2025-12-17  
> **파일**: `user_logs_aggregated_ldh.parquet`

---

## 개요

| 항목 | 값 |
|------|-----|
| 총 컬럼 수 | 67개 |
| 총 레코드 수 | 1,103,894 rows |
| 기준 시점 | 2017-03-31 |
| 관측 윈도우 | 2017-03-01 ~ 2017-03-31 (30일) |

---

## 윈도우 기준 정보

| 윈도우 | 기간 | 목적 |
|--------|------|------|
| **W7** | 2017-03-25 ~ 03-31 | 최근 7일 행동 패턴 |
| **W14** | 2017-03-18 ~ 03-31 | 최근 14일 행동 패턴 |
| **W21** | 2017-03-11 ~ 03-31 | 최근 21일 행동 패턴 |
| **W30** | 2017-03-01 ~ 03-31 | 전체 30일 행동 패턴 |

---

## 컬럼 상세 명세

### 1. 식별자 (1개)

| # | 컬럼명 | 타입 | 설명 | 예시 |
|---|--------|------|------|------|
| 0 | `msno` | object | 사용자 고유 ID (해시값) | `+++IZse...` |

---

### 2. 기본 집계 피처 (윈도우별 10개 × 4 = 40개)

#### 2.1 활동량 피처

| # | 컬럼명 패턴 | 타입 | 설명 | 단위 |
|---|-------------|------|------|------|
| 1-4 | `num_days_active_wX` | float64/int64 | 해당 윈도우 내 활동 일수 | 일 |
| 5-8 | `total_secs_wX` | float64 | 총 청취 시간 | 초 |
| 9-12 | `avg_secs_per_day_wX` | float64 | 일평균 청취 시간 | 초/일 |
| 13-16 | `std_secs_wX` | float64 | 청취 시간의 표준편차 | 초 |

#### 2.2 곡 재생 피처

| # | 컬럼명 패턴 | 타입 | 설명 | 단위 |
|---|-------------|------|------|------|
| 17-20 | `num_songs_wX` | float64/int64 | 총 재생 곡 수 | 곡 |
| 21-24 | `avg_songs_per_day_wX` | float64 | 일평균 재생 곡 수 | 곡/일 |
| 25-28 | `num_unq_wX` | float64/int64 | 고유 곡 수 (중복 제외) | 곡 |

#### 2.3 청취 완료도 피처

| # | 컬럼명 패턴 | 타입 | 설명 | 의미 |
|---|-------------|------|------|------|
| 29-32 | `num_25_wX` | float64/int64 | 25% 미만 청취 곡 수 | 스킵된 곡 |
| 33-36 | `num_100_wX` | float64/int64 | 100% 완주 청취 곡 수 | 끝까지 들은 곡 |
| 37-40 | `short_play_wX` | float64/int64 | 50% 미만 청취 곡 수 | 짧게 들은 곡 |

---

### 3. 비율형 피처 (윈도우별 4개 × 4 = 16개)

| # | 컬럼명 패턴 | 타입 | 계산식 | Churn 해석 |
|---|-------------|------|--------|-----------|
| 41-44 | `skip_ratio_wX` | float64 | num_25 / total_songs | 높으면 불만족 ⚠️ |
| 45-48 | `completion_ratio_wX` | float64 | num_100 / total_songs | 낮으면 불만족 ⚠️ |
| 49-52 | `short_play_ratio_wX` | float64 | short_play / total_songs | 높으면 불만족 ⚠️ |
| 53-56 | `variety_ratio_wX` | float64 | num_unq / total_songs | 탐색 성향 지표 |

---

### 4. 변화량/추세 피처 (10개)

#### 4.1 청취 시간 추세

| # | 컬럼명 | 타입 | 계산식 | Churn 신호 |
|---|--------|------|--------|-----------|
| 57 | `secs_trend_w7_w30` | float64 | total_secs_w7 / total_secs_w30 | < 0.1 이면 위험 🚨 |
| 58 | `secs_trend_w14_w30` | float64 | total_secs_w14 / total_secs_w30 | 낮으면 위험 |

#### 4.2 활동일 추세

| # | 컬럼명 | 타입 | 계산식 | Churn 신호 |
|---|--------|------|--------|-----------|
| 59 | `days_trend_w7_w14` | float64 | num_days_active_w7 / num_days_active_w14 | 낮으면 위험 |
| 60 | `days_trend_w7_w30` | float64 | num_days_active_w7 / num_days_active_w30 | 낮으면 위험 |

#### 4.3 곡 재생 추세

| # | 컬럼명 | 타입 | 계산식 | Churn 신호 |
|---|--------|------|--------|-----------|
| 61 | `songs_trend_w7_w30` | float64 | num_songs_w7 / num_songs_w30 | 낮으면 위험 |
| 62 | `songs_trend_w14_w30` | float64 | num_songs_w14 / num_songs_w30 | 낮으면 위험 |

#### 4.4 행동 변화 추세

| # | 컬럼명 | 타입 | 계산식 | Churn 신호 |
|---|--------|------|--------|-----------|
| 63 | `skip_trend_w7_w30` | float64 | skip_ratio_w7 - skip_ratio_w30 | > 0 이면 위험 🚨 |
| 64 | `completion_trend_w7_w30` | float64 | completion_ratio_w7 - completion_ratio_w30 | < 0 이면 위험 🚨 |

#### 4.5 최근성 지표

| # | 컬럼명 | 타입 | 계산식 | Churn 신호 |
|---|--------|------|--------|-----------|
| 65 | `recency_secs_ratio` | float64 | total_secs_w7 / total_secs_w30 | 낮으면 위험 🚨 |
| 66 | `recency_songs_ratio` | float64 | num_songs_w7 / num_songs_w30 | 낮으면 위험 |

---

## 전체 컬럼 목록 (인덱스순)

```
 0: msno
 1: num_days_active_w7
 2: total_secs_w7
 3: avg_secs_per_day_w7
 4: std_secs_w7
 5: num_songs_w7
 6: avg_songs_per_day_w7
 7: num_unq_w7
 8: num_25_w7
 9: num_100_w7
10: short_play_w7
11: skip_ratio_w7
12: completion_ratio_w7
13: short_play_ratio_w7
14: variety_ratio_w7
15: num_days_active_w14
16: total_secs_w14
17: avg_secs_per_day_w14
18: std_secs_w14
19: num_songs_w14
20: avg_songs_per_day_w14
21: num_unq_w14
22: num_25_w14
23: num_100_w14
24: short_play_w14
25: skip_ratio_w14
26: completion_ratio_w14
27: short_play_ratio_w14
28: variety_ratio_w14
29: num_days_active_w21
30: total_secs_w21
31: avg_secs_per_day_w21
32: std_secs_w21
33: num_songs_w21
34: avg_songs_per_day_w21
35: num_unq_w21
36: num_25_w21
37: num_100_w21
38: short_play_w21
39: skip_ratio_w21
40: completion_ratio_w21
41: short_play_ratio_w21
42: variety_ratio_w21
43: num_days_active_w30
44: total_secs_w30
45: avg_secs_per_day_w30
46: std_secs_w30
47: num_songs_w30
48: avg_songs_per_day_w30
49: num_unq_w30
50: num_25_w30
51: num_100_w30
52: short_play_w30
53: skip_ratio_w30
54: completion_ratio_w30
55: short_play_ratio_w30
56: variety_ratio_w30
57: secs_trend_w7_w30
58: secs_trend_w14_w30
59: days_trend_w7_w14
60: days_trend_w7_w30
61: songs_trend_w7_w30
62: songs_trend_w14_w30
63: skip_trend_w7_w30
64: completion_trend_w7_w30
65: recency_secs_ratio
66: recency_songs_ratio
```

---

## 기술 통계 (W30 기준)

| 피처 | Mean | Std | Min | 25% | 50% | 75% | Max |
|------|------|-----|-----|-----|-----|-----|-----|
| `num_days_active_w30` | 16.66 | 10.30 | 1 | 7 | 18 | 26 | 31 |
| `total_secs_w30` | 131,733 | 185,227 | 0.3 | 13,115 | 67,936 | 173,934 | 2,406,313 |
| `num_songs_w30` | 642 | 829 | 1 | 73 | 354 | 877 | 11,490 |
| `skip_ratio_w30` | 0.20 | 0.18 | 0 | 0.06 | 0.15 | 0.29 | 1.0 |
| `completion_ratio_w30` | 0.80 | 0.18 | 0 | 0.71 | 0.85 | 0.94 | 1.0 |

---

## Churn 예측 핵심 피처 추천

### 🎯 Top Priority 피처

1. **`recency_secs_ratio`** - 최근 7일 청취 비중 (급감 시 이탈 위험)
2. **`skip_trend_w7_w30`** - 스킵율 변화 (증가 시 불만족)
3. **`completion_trend_w7_w30`** - 완주율 변화 (감소 시 불만족)
4. **`secs_trend_w7_w30`** - 청취시간 추세 (< 0.1 이면 고위험)

### 📊 파생 피처 제안

```python
# 급감 플래그
df['usage_drop_flag'] = (df['secs_trend_w7_w30'] < 0.1).astype(int)

# 불만족 증가 플래그  
df['dissatisfaction_flag'] = (df['skip_trend_w7_w30'] > 0.1).astype(int)

# 고위험 사용자 플래그
df['high_risk_flag'] = (
    (df['recency_secs_ratio'] < 0.1) | 
    (df['completion_trend_w7_w30'] < -0.2)
).astype(int)
```

---

## 데이터 타입 정보

| 타입 | 컬럼 수 | 비고 |
|------|---------|------|
| object | 1개 | msno (사용자 ID) |
| int64 | 일부 | W30 원본 집계값 |
| float64 | 대부분 | 비율, 평균, 추세 피처 |

> ⚠️ W7, W14, W21 윈도우의 일부 컬럼은 결측치 처리로 인해 float64로 저장됨

---

## 결측치/이상치 처리

### 결측치 처리

| 상황 | 처리 방법 |
|------|----------|
| 특정 윈도우 활동 없음 | `fillna(0)` |
| 비율 계산 시 0 나눗셈 | `eps = 1e-9` 추가 |
| Inf 값 발생 | `replace([np.inf, -np.inf], 0)` |

### 이상치 처리

- **Percentile 클리핑**: 0.1% ~ 99.9%
- **대상 컬럼**: `total_secs`, `num_25`, `num_50`, `num_75`, `num_985`, `num_100`, `num_unq`

---

**End of Document**

