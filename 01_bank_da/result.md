# 탐색적 데이터 분석 (EDA) 결과
## 1. 데이터 기본 정보
- **데이터 크기**: 41188행 × 21열
- **중복 행 수**: 12개

## 2. 컬럼 정보
- **총 컬럼 수**: 21개

### 컬럼 목록:
- `age` (int64) - 결측치: 0개
- `job` (object) - 결측치: 0개
- `marital` (object) - 결측치: 0개
- `education` (object) - 결측치: 0개
- `default` (object) - 결측치: 0개
- `housing` (object) - 결측치: 0개
- `loan` (object) - 결측치: 0개
- `contact` (object) - 결측치: 0개
- `month` (object) - 결측치: 0개
- `day_of_week` (object) - 결측치: 0개
- `duration` (int64) - 결측치: 0개
- `campaign` (int64) - 결측치: 0개
- `pdays` (int64) - 결측치: 0개
- `previous` (int64) - 결측치: 0개
- `poutcome` (object) - 결측치: 0개
- `emp.var.rate` (float64) - 결측치: 0개
- `cons.price.idx` (float64) - 결측치: 0개
- `cons.conf.idx` (float64) - 결측치: 0개
- `euribor3m` (float64) - 결측치: 0개
- `nr.employed` (float64) - 결측치: 0개
- `y` (object) - 결측치: 0개

## 3. 결측치 현황
- 결측치가 없습니다.

## 4. 타겟 변수 분포
- `no`: 36548개 (88.73%)
- `yes`: 4640개 (11.27%)

## 5. 수치형 변수 통계

| 변수 | 평균 | 표준편차 | 최소값 | 25% | 50% | 75% | 최대값 |
|------|------|----------|--------|-----|-----|-----|--------|
| age | 40.02 | 10.42 | 17.00 | 32.00 | 38.00 | 47.00 | 98.00 |
| duration | 258.29 | 259.28 | 0.00 | 102.00 | 180.00 | 319.00 | 4918.00 |
| campaign | 2.57 | 2.77 | 1.00 | 1.00 | 2.00 | 3.00 | 56.00 |
| pdays | 962.48 | 186.91 | 0.00 | 999.00 | 999.00 | 999.00 | 999.00 |
| previous | 0.17 | 0.49 | 0.00 | 0.00 | 0.00 | 0.00 | 7.00 |
| emp.var.rate | 0.08 | 1.57 | -3.40 | -1.80 | 1.10 | 1.40 | 1.40 |
| cons.price.idx | 93.58 | 0.58 | 92.20 | 93.08 | 93.75 | 93.99 | 94.77 |
| cons.conf.idx | -40.50 | 4.63 | -50.80 | -42.70 | -41.80 | -36.40 | -26.90 |
| euribor3m | 3.62 | 1.73 | 0.63 | 1.34 | 4.86 | 4.96 | 5.04 |
| nr.employed | 5167.04 | 72.25 | 4963.60 | 5099.10 | 5191.00 | 5228.10 | 5228.10 |

## 6. 생성된 시각화
- `plots/target_distribution.png`: 타겟 변수 분포
- `plots/numeric_distributions.png`: 수치형 변수 분포
- `plots/categorical_distributions.png`: 범주형 변수 분포
- `plots/correlation_heatmap.png`: 상관관계 히트맵


# 모델링 결과

## 1. 모델 성능 비교

| 모델 | AUC-ROC |

|------|----------|

| Random Forest | 0.9484 |

| Gradient Boosting | 0.9536 |


## 2. 특성 중요도 (상위 10개)


### Random Forest

| 순위 | 특성명 | 중요도 |

|------|--------|--------|

| 1 | duration | 0.3193 |

| 2 | euribor3m | 0.1043 |

| 3 | age | 0.0931 |

| 4 | nr.employed | 0.0619 |

| 5 | job | 0.0484 |

| 6 | education | 0.0440 |

| 7 | campaign | 0.0425 |

| 8 | day_of_week | 0.0404 |

| 9 | pdays | 0.0333 |

| 10 | poutcome | 0.0297 |


### Gradient Boosting

| 순위 | 특성명 | 중요도 |

|------|--------|--------|

| 1 | duration | 0.4775 |

| 2 | nr.employed | 0.2606 |

| 3 | euribor3m | 0.0880 |

| 4 | pdays | 0.0533 |

| 5 | cons.conf.idx | 0.0388 |

| 6 | poutcome | 0.0191 |

| 7 | month | 0.0146 |

| 8 | cons.price.idx | 0.0126 |

| 9 | age | 0.0087 |

| 10 | contact | 0.0072 |


## 3. 시각화

### ROC 곡선 비교

![ROC 곡선 비교](plots/roc_curves.png)

### 특성 중요도 비교

![특성 중요도 비교](plots/feature_importance.png)
