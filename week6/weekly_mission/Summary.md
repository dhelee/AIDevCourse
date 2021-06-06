# 6주차 과제 요약문

### 문제 정의
- "음식배달에 걸리는 시간(Delivery Time)"을 예측하는 회귀 문제

### 성능측정지표
- MAE(Mean Absolute Error)
- Under-prediction의 비율(under-prediction 개수 / 테스트 데이터 샘플 수)

## 1. 탐색적 데이터 분석
- 변수 파악하기
  - Restaurant: 음식점 고유의 ID
  - Location: 음식점의 위치
  - Cuisines: 음식점에서 취급하는 메뉴
  - Average_Cost: (고객당) 평균 주문가격
  - Minimum_Order: 최소 주문량(또는 금액)
  - Rating: 음식점의 평점
  - Votes: 평점을 남긴 고객 수
  - Reviews: 리뷰 수
  - Delivery_Time: 배달시간(예측하고자 하는 값)

- 변수 간 상관관계 확인


## 2. 머신러닝을 위한 데이터 준비
- 결측치 처리

- 새로운 변수 정의

## 3. 모델 학습
- 변수 조합 실험

- 모델 세부 튜닝

## 4. 모델 평가
- 테스트셋으로 최종 모델의 성능 평가
