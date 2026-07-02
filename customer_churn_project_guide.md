# 8단계: 실제 프로젝트 (고객 이탈 예측 / 사기 탐지)
> **비즈니스 문제를 해결하는 엔드투엔드(End-to-End) 데이터 사이언스 파이프라인**

---

## 0. 전체 흐름 (실무 표준)

실무에서 데이터 분석 및 머신러닝 프로젝트는 모델링 자체보다 문제를 정의하고 데이터를 비즈니스 가치로 연결하는 흐름이 훨씬 중요합니다.

[문제 정의] ──> [데이터 이해] ──> [전처리] ──> [모델링] ──> [평가] ──> [개선] ──> [배포]

---

## 1. 문제 정의 (🔥 가장 중요)

### 🎯 비즈니스 문제
* **핵심 질문:** “어떤 고객이 우리 서비스를 떠날 것인가(Churn)?”

### 🎯 목표
* **이탈 가능성이 높은 고객을 사전에 예측**하여 고객 유지(Retention) 비용을 최적화합니다.

### 🎯 활용 방안
* **마케팅 타겟팅:** 이탈 징후 고객 대상 맞춤형 프로모션 진행
* **혜택 제공:** 할인 쿠폰 발행 및 요금제 개편 제안
* **VIP 관리:** 고가치 고객(High-Value Customer) 집중 이탈 방지 케어

### 🎯 문제 유형
* **분류 (Classification):** 고객의 상태를 두 가지 클래스로 나누는 이진 분류 문제입니다.
  * **1:** 이탈 (Churn)
  * **0:** 유지 (Retention)

---

## 2. 데이터 구조 (실무 형태)

### 📊 데이터 프레임 예시 컬럼
실무에서는 고객의 기본 정보, 행동 로그, 계약 정보가 결합된 형태의 데이터를 사용합니다.

| 고객ID (ID) | 나이 (Age) | 사용기간 (Tenure) | 월요금 (MonthlyCharges) | 고객센터문의 (CustomerServiceCalls) | 계약유형 (ContractType) | 이탈여부 (Churn) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| C001 | 34 | 12 (개월) | 45,000 (원) | 3 (회) | 1년 계약 | **1 (이탈)** |
| C002 | 28 | 45 (개월) | 22,000 (원) | 0 (회) | 약정 없음 | **0 (유지)** |

### 💻 X / y 데이터 분리
예측에 사용할 특징 변수(X)와 맞추고자 하는 정답 타겟 변수(y)를 분리합니다.

    import pandas as pd
    
    # df.drop()을 통해 정답 컬럼인 'churn'만 제거하여 입력 데이터(X)를 만듭니다.
    # axis=1은 세로 열(Column) 방향으로 연산하라는 의미입니다.
    X = df.drop("churn", axis=1)
    
    # 대괄호 규칙을 사용해 정답 컬럼인 'churn'만 쏙 빼내어 타겟 변수(y)로 지정합니다.
    y = df["churn"]

---

## 3. 데이터 전처리 (🔥 성능의 80% 결정)

> *"Garbage In, Garbage Out." 모델 알고리즘보다 데이터의 정제 상태가 예측 성능을 좌우합니다.*

### ① 결측치 처리 (Missing Values)
데이터에 빈 곳이 있으면 머신러닝 알고리즘이 연산을 수행하지 못합니다. 상황에 따라 0, 평균값, 중위수 등으로 채웁니다.

    # df.fillna(0)은 데이터프레임 내의 모든 빈 칸(NaN)을 숫자 0으로 채워주는 함수입니다.
    # inplace=True는 변경 사항을 다른 변수에 담지 않고 원본 df에 바로 적용하라는 뜻입니다.
    df.fillna(0, inplace=True)

### ② 범주형 데이터 처리 (Categorical Encoding)
텍스트 형태의 데이터(예: 계약유형 - '1년 계약', '약정 없음')를 컴퓨터가 이해할 수 있도록 숫자로 변환합니다.

    # pd.get_dummies()는 텍스트(문자형) 데이터를 0과 1로 이루어진 여러 열로 쪼개는 원-핫 인코딩 함수입니다.
    # drop_first=True는 다중공선성 문제를 방지하기 위해 첫 번째 더미 열을 자동으로 삭제하는 실무 필수 옵션입니다.
    df = pd.get_dummies(df, drop_first=True)

### ③ 데이터 스케일링 (Feature Scaling)
로지스틱 회귀와 같은 선형 모델은 변수 간의 단위 크기(예: 나이는 20~60, 월요금은 수만 원 단위)에 큰 영향을 받으므로 평균 0, 표준편차 1로 규격화해야 합니다.

    from sklearn.preprocessing import StandardScaler
    
    # StandardScaler 클래스의 인스턴스(객체)를 생성합니다.
    scaler = StandardScaler()
    
    # fit_transform()으로 데이터의 평균과 표준편차를 학습(fit)함과 동시에 변환(transform) 작업을 한 번에 처리합니다.
    X_scaled = scaler.fit_transform(X)

### ❗ 실무 핵심
* **전처리가 모델보다 중요합니다.** 아무리 화려하고 복잡한 최신 모델을 가져와도 데이터 전처리가 엉망이면 좋은 성능을 절대 낼 수 없습니다.

---

## 4. 데이터 분리 (Train / Test Split)

모델이 학습 데이터만 암기(과적합)하는 것을 방지하기 위해 데이터를 학습용과 검증용으로 분리합니다.

    from sklearn.model_selection import train_test_split
    
    # 전체 데이터를 학습용(80%)과 검증 평가용(20%)으로 나누는 함수입니다.
    # test_size=0.2는 평가 데이터의 비율을 20%로 설정하겠다는 의미입니다.
    # random_state=42는 코드를 다시 실행해도 매번 똑같은 형태로 쪼개지도록 고정하는 난수 시드 값입니다.
    # stratify=y는 데이터가 불균형할 때 학습셋과 평가셋의 이탈 고객 비율을 동일하게 유지해 주는 실무 필수 옵션입니다.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

* **실무 팁:** 이탈 데이터처럼 정답의 불균형이 심할 때는 `stratify=y` 옵션을 주어 Train과 Test 세트의 이탈 비율을 동일하게 맞춰야 안정적인 평가가 가능합니다.

---

## 5. 모델 학습 (Baseline ──────> 성능 개선 구조)

### 1단계: Logistic Regression (Baseline 모델)
가장 빠르고 단순하며 해석이 쉬운 선형 모델로 기준점 점수를 먼저 확보합니다.

    from sklearn.linear_model import LogisticRegression
    
    # 로지스틱 회귀 모델 객체를 생성합니다.
    lr_model = LogisticRegression()
    
    # model.fit() 함수로 학습 데이터(X_train)와 정답(y_train)을 주입해 데이터 간의 선형 관계 규칙을 학습시킵니다.
    lr_model.fit(X_train, y_train)

### 2단계: Random Forest (성능 고도화)
선형 모델보다 복잡한 규칙과 비선형 관계를 잡을 수 있는 트리 기반 앙상블 모델로 성능을 끌어올립니다.

    from sklearn.ensemble import RandomForestClassifier
    
    # 랜덤 포레스트 분류기 객체를 생성합니다 (결과 재현을 위해 random_state 고정).
    rf_model = RandomForestClassifier(random_state=42)
    
    # 여러 개의 의사결정나무(Decision Tree)를 동시에 만들어 집단지성 형태로 복잡한 데이터 패턴을 학습시킵니다.
    rf_model.fit(X_train, y_train)

---

## 6. 예측 및 평가 (🔥 실무 핵심)

    from sklearn.metrics import classification_report
    
    # model.predict() 함수로 한 번도 본 적 없는 시험 데이터(X_test)에 대한 최종 클래스(0 또는 1)를 예측합니다.
    pred = rf_model.predict(X_test)
    
    # classification_report()는 실제 정답(y_test)과 모델의 예측값(pred)을 비교해 정밀도, 재현율, F1 스코어를 종합 리포트로 출력합니다.
    print(classification_report(y_test, pred))

### 📊 이탈 예측에서 진짜 확인해야 할 지표
* **Precision (정밀도):** 모델이 이탈할 것이라고 예측한 사람들 중 **실제 이탈한 사람의 비율** (예측의 정확성)
* **Recall (재현율, 🔥 가장 중요):** 실제 이탈한 전체 고객 중 **모델이 올바르게 찾아낸 이탈 고객의 비율**
* **F1-Score:** 정밀도와 재현율의 균형을 나타내는 조화평균 지표
* **💡 이유:** 이탈 예측 프로모션에서는 이탈할 고객을 단 한 명이라도 놓치지 않고 다 찾아내는 것(**Recall 고도화**)이 비즈니스 손실을 방지하는 지름길입니다.

---

## 7. 실무 핵심 전략 (Advanced 테크닉)

### ① 임계값(Threshold) 조정
기본 `predict` 함수는 확률이 0.5(50%)를 넘어야만 이탈(1)로 분류합니다. 하지만 이를 **0.3(30%)**으로 낮추면 이탈 징후가 조금만 보여도 이탈 위험군으로 분류하게 되어 **Recall(재현율)을 획기적으로 올릴 수 있습니다.**

    # predict_proba()는 최종 결론(0, 1) 대신에 각 클래스별 구체적인 확률값(예: 0일 확률 30%, 1일 확률 70%)을 반환합니다.
    # [:, 1]은 반환된 2차원 배열에서 '이탈(1)할 확률'에 해당하는 두 번째 열만 모두 슬라이싱하겠다는 뜻입니다.
    proba = rf_model.predict_proba(X_test)[:, 1]
    
    # 추출한 이탈 확률이 0.3(30%)보다 크면 True(1), 그렇지 않으면 False(0)로 변환하여 새로운 판정 기준으로 삼습니다.
    # .astype(int)를 통해 True/False 불리언 값을 최종적으로 정수형 1과 0으로 규격화합니다.
    custom_pred = (proba > 0.3).astype(int)

### ② 피처 중요도 (Feature Importance) 확인
블랙박스 형태의 복잡한 앙상블 모델이더라도 어떤 피처를 중요하게 보고 판단했인지 추출하여 비즈니스 부서에 설명할 수 있습니다.

    import numpy as np
    
    # rf_model.feature_importances_ 속성은 모델이 수많은 나무를 만드는 과정에서 어떤 변수를 가장 유용하게 썼는지 수치화한 배열입니다.
    importances = rf_model.feature_importances_
    
    # 삼항 연산자를 활용해 입력 데이터(X)에 컬럼 이름 정보가 있다면 가져오고, 없다면 순서대로 'feature_0, feature_1' 형태로 임시 가공합니다.
    feature_names = X.columns if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
    
    # 판다스 데이터프레임을 생성해 변수명과 중요도 점수를 나란히 매칭합니다.
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    
    # sort_values(by='Importance', ascending=False)를 통해 가장 영향력이 큰 변수부터 내림차순으로 정렬하여 출력합니다.
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    print(feature_imp_df)

---

## 8. 결과 해석 및 비즈니스 가치 연결

머신러닝 결과를 바탕으로 비즈니스 의사결정권자에게 제안할 수 있는 **인사이트(Insight)** 예시입니다.

* **인사이트 1:** "가입 후 **사용기간(Tenure)**이 3개월 미만인 고객의 이탈률이 가장 높습니다."
  * ➡️ *액션 플랜:* 신규 가입자 대상 웰컴 키트 및 첫 3개월 특별 집중 케어 프로모션 진행
* **인사이트 2:** "**고객센터 문의 횟수(CustomerServiceCalls)**가 3회 이상인 순간 이탈 확률이 70%를 돌파합니다."
  * ➡️ *액션 플랜:* 고객센터 3회 이상 문의 기록 발생 시, 즉각 요금 할인 혜택 부서나 해지 방지 전담반으로 콜을 이관 및 선제적 대응
* **인사이트 3:** "**월요금(MonthlyCharges)**이 상위 10%인 VIP 고객들의 이탈률이 증가하고 있습니다."
  * ➡️ *액션 플랜:* 고가치 고객 전용 결합 할인 혜택 및 멤버십 서비스 강화

---

## 9. 실무에서 가장 많이 범하는 3대 실수

* ❌ **실수 1: 오직 정확도(Accuracy)만 맹신함**
  * *현실:* 전체 고객의 95%가 유지되고 5%만 이탈하는 불균형 데이터셋에서는, 모델이 아무 일도 안 하고 무조건 '유지(0)'라고만 답해도 정확도가 95%가 나옵니다. 정작 이탈자 5%는 한 명도 찾지 못했으므로 쓸모없는 모델입니다. **반드시 Recall과 F1-Score를 보아야 합니다.**
* ❌ **실수 2: 임계값(Threshold)을 무조건 0.5로 고정함**
  * *현실:* 비즈니스 상황(이탈 예방 혜택 비용 vs 이탈 시 손실 비용)의 손익분기점을 따져 임계값을 최적화해야 비즈니스 이익이 극대화됩니다.
* ❌ **실수 3: 데이터 전처리 없이 복잡한 최신 모델만 돌림**
  * *현실:* 데이터 클리닝과 정제, 도메인 지식을 활용한 파생 변수(Feature Engineering) 생성 없이 최신 모델(XGBoost, LightGBM 등)만 적용하면 절대 실무 수준의 높은 성능을 얻을 수 없습니다.

  

  # 8단계: 실제 프로젝트 (고객 이탈 예측 / 사기 탐지)
> **비즈니스 문제를 해결하는 엔드투엔드(End-to-End) 데이터 사이언스 파이프라인**

---

## 0. 전체 흐름 (실무 표준)

실무에서 데이터 분석 및 머신러닝 프로젝트는 모델링 자체보다 문제를 정의하고 데이터를 비즈니스 가치로 연결하는 흐름이 훨씬 중요합니다.

[문제 정의] ──> [데이터 이해] ──> [전처리] ──> [모델링] ──> [평가] ──> [개선] ──> [배포]

---

## 1. 문제 정의 (🔥 가장 중요)

### 🎯 비즈니스 문제
* **핵심 질문:** “어떤 고객이 우리 서비스를 떠날 것인가(Churn)?”

### 🎯 목표
* **이탈 가능성이 높은 고객을 사전에 예측**하여 고객 유지(Retention) 비용을 최적화합니다.

### 🎯 활용 방안
* **마케팅 타겟팅:** 이탈 징후 고객 대상 맞춤형 프로모션 진행
* **혜택 제공:** 할인 쿠폰 발행 및 요금제 개편 제안
* **VIP 관리:** 고가치 고객(High-Value Customer) 집중 이탈 방지 케어

### 🎯 문제 유형
* **분류 (Classification):** 고객의 상태를 두 가지 클래스로 나누는 이진 분류 문제입니다.
  * **1:** 이탈 (Churn)
  * **0:** 유지 (Retention)

---

## 2. 데이터 구조 (실무 형태)

### 📊 데이터 프레임 예시 컬럼
실무에서는 고객의 기본 정보, 행동 로그, 계약 정보가 결합된 형태의 데이터를 사용합니다.

| 고객ID (ID) | 나이 (Age) | 사용기간 (Tenure) | 월요금 (MonthlyCharges) | 고객센터문의 (CustomerServiceCalls) | 계약유형 (ContractType) | 이탈여부 (Churn) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| C001 | 34 | 12 (개월) | 45,000 (원) | 3 (회) | 1년 계약 | **1 (이탈)** |
| C002 | 28 | 45 (개월) | 22,000 (원) | 0 (회) | 약정 없음 | **0 (유지)** |

### 💻 X / y 데이터 분리
예측에 사용할 특징 변수(X)와 맞추고자 하는 정답 타겟 변수(y)를 분리합니다.

    import pandas as pd
    
    # df.drop()을 통해 정답 컬럼인 'churn'만 제거하여 입력 데이터(X)를 만듭니다.
    # axis=1은 세로 열(Column) 방향으로 연산하라는 의미입니다.
    X = df.drop("churn", axis=1)
    
    # 대괄호 규칙을 사용해 정답 컬럼인 'churn'만 쏙 빼내어 타겟 변수(y)로 지정합니다.
    y = df["churn"]

---

## 3. 데이터 전처리 (🔥 성능의 80% 결정)

> *"Garbage In, Garbage Out." 모델 알고리즘보다 데이터의 정제 상태가 예측 성능을 좌우합니다.*

### ① 결측치 처리 (Missing Values)
데이터에 빈 곳이 있으면 머신러닝 알고리즘이 연산을 수행하지 못합니다. 상황에 따라 0, 평균값, 중위수 등으로 채웁니다.

    # df.fillna(0)은 데이터프레임 내의 모든 빈 칸(NaN)을 숫자 0으로 채워주는 함수입니다.
    # inplace=True는 변경 사항을 다른 변수에 담지 않고 원본 df에 바로 적용하라는 뜻입니다.
    df.fillna(0, inplace=True)

### ② 범주형 데이터 처리 (Categorical Encoding)
텍스트 형태의 데이터(예: 계약유형 - '1년 계약', '약정 없음')를 컴퓨터가 이해할 수 있도록 숫자로 변환합니다.

    # pd.get_dummies()는 텍스트(문자형) 데이터를 0과 1로 이루어진 여러 열로 쪼개는 원-핫 인코딩 함수입니다.
    # drop_first=True는 다중공선성 문제를 방지하기 위해 첫 번째 더미 열을 자동으로 삭제하는 실무 필수 옵션입니다.
    df = pd.get_dummies(df, drop_first=True)

### ③ 데이터 스케일링 (Feature Scaling)
로지스틱 회귀와 같은 선형 모델은 변수 간의 단위 크기(예: 나이는 20~60, 월요금은 수만 원 단위)에 큰 영향을 받으므로 평균 0, 표준편차 1로 규격화해야 합니다.

    from sklearn.preprocessing import StandardScaler
    
    # StandardScaler 클래스의 인스턴스(객체)를 생성합니다.
    scaler = StandardScaler()
    
    # fit_transform()으로 데이터의 평균과 표준편차를 학습(fit)함과 동시에 변환(transform) 작업을 한 번에 처리합니다.
    X_scaled = scaler.fit_transform(X)

### ❗ 실무 핵심
* **전처리가 모델보다 중요합니다.** 아무리 화려하고 복잡한 최신 모델을 가져와도 데이터 전처리가 엉망이면 좋은 성능을 절대 낼 수 없습니다.

---

## 4. 데이터 분리 (Train / Test Split)

모델이 학습 데이터만 암기(과적합)하는 것을 방지하기 위해 데이터를 학습용과 검증용으로 분리합니다.

    from sklearn.model_selection import train_test_split
    
    # 전체 데이터를 학습용(80%)과 검증 평가용(20%)으로 나누는 함수입니다.
    # test_size=0.2는 평가 데이터의 비율을 20%로 설정하겠다는 의미입니다.
    # random_state=42는 코드를 다시 실행해도 매번 똑같은 형태로 쪼개지도록 고정하는 난수 시드 값입니다.
    # stratify=y는 데이터가 불균형할 때 학습셋과 평가셋의 이탈 고객 비율을 동일하게 유지해 주는 실무 필수 옵션입니다.
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

* **실무 팁:** 이탈 데이터처럼 정답의 불균형이 심할 때는 `stratify=y` 옵션을 주어 Train과 Test 세트의 이탈 비율을 동일하게 맞춰야 안정적인 평가가 가능합니다.

---

## 5. 모델 학습 (Baseline ──────> 성능 개선 구조)

### 1단계: Logistic Regression (Baseline 모델)
가장 빠르고 단순하며 해석이 쉬운 선형 모델로 기준점 점수를 먼저 확보합니다.

    from sklearn.linear_model import LogisticRegression
    
    # 로지스틱 회귀 모델 객체를 생성합니다.
    lr_model = LogisticRegression()
    
    # model.fit() 함수로 학습 데이터(X_train)와 정답(y_train)을 주입해 데이터 간의 선형 관계 규칙을 학습시킵니다.
    lr_model.fit(X_train, y_train)

### 2단계: Random Forest (성능 고도화)
선형 모델보다 복잡한 규칙과 비선형 관계를 잡을 수 있는 트리 기반 앙상블 모델로 성능을 끌어올립니다.

    from sklearn.ensemble import RandomForestClassifier
    
    # 랜덤 포레스트 분류기 객체를 생성합니다 (결과 재현을 위해 random_state 고정).
    rf_model = RandomForestClassifier(random_state=42)
    
    # 여러 개의 의사결정나무(Decision Tree)를 동시에 만들어 집단지성 형태로 복잡한 데이터 패턴을 학습시킵니다.
    rf_model.fit(X_train, y_train)

---

## 6. 예측 및 평가 (🔥 실무 핵심)

    from sklearn.metrics import classification_report
    
    # model.predict() 함수로 한 번도 본 적 없는 시험 데이터(X_test)에 대한 최종 클래스(0 또는 1)를 예측합니다.
    pred = rf_model.predict(X_test)
    
    # classification_report()는 실제 정답(y_test)과 모델의 예측값(pred)을 비교해 정밀도, 재현율, F1 스코어를 종합 리포트로 출력합니다.
    print(classification_report(y_test, pred))

### 📊 이탈 예측에서 진짜 확인해야 할 지표
* **Precision (정밀도):** 모델이 이탈할 것이라고 예측한 사람들 중 **실제 이탈한 사람의 비율** (예측의 정확성)
* **Recall (재현율, 🔥 가장 중요):** 실제 이탈한 전체 고객 중 **모델이 올바르게 찾아낸 이탈 고객의 비율**
* **F1-Score:** 정밀도와 재현율의 균형을 나타내는 조화평균 지표
* **💡 이유:** 이탈 예측 프로모션에서는 이탈할 고객을 단 한 명이라도 놓치지 않고 다 찾아내는 것(**Recall 고도화**)이 비즈니스 손실을 방지하는 지름길입니다.

---

## 7. 실무 핵심 전략 (Advanced 테크닉)

### ① 임계값(Threshold) 조정
기본 `predict` 함수는 확률이 0.5(50%)를 넘어야만 이탈(1)로 분류합니다. 하지만 이를 **0.3(30%)**으로 낮추면 이탈 징후가 조금만 보여도 이탈 위험군으로 분류하게 되어 **Recall(재현율)을 획기적으로 올릴 수 있습니다.**

    # predict_proba()는 최종 결론(0, 1) 대신에 각 클래스별 구체적인 확률값(예: 0일 확률 30%, 1일 확률 70%)을 반환합니다.
    # [:, 1]은 반환된 2차원 배열에서 '이탈(1)할 확률'에 해당하는 두 번째 열만 모두 슬라이싱하겠다는 뜻입니다.
    proba = rf_model.predict_proba(X_test)[:, 1]
    
    # 추출한 이탈 확률이 0.3(30%)보다 크면 True(1), 그렇지 않으면 False(0)로 변환하여 새로운 판정 기준으로 삼습니다.
    # .astype(int)를 통해 True/False 불리언 값을 최종적으로 정수형 1과 0으로 규격화합니다.
    custom_pred = (proba > 0.3).astype(int)

### ② 피처 중요도 (Feature Importance) 확인
블랙박스 형태의 복잡한 앙상블 모델이더라도 어떤 피처를 중요하게 보고 판단했인지 추출하여 비즈니스 부서에 설명할 수 있습니다.

    import numpy as np
    
    # rf_model.feature_importances_ 속성은 모델이 수많은 나무를 만드는 과정에서 어떤 변수를 가장 유용하게 썼는지 수치화한 배열입니다.
    importances = rf_model.feature_importances_
    
    # 삼항 연산자를 활용해 입력 데이터(X)에 컬럼 이름 정보가 있다면 가져오고, 없다면 순서대로 'feature_0, feature_1' 형태로 임시 가공합니다.
    feature_names = X.columns if hasattr(X, 'columns') else [f"feature_{i}" for i in range(X.shape[1])]
    
    # 판다스 데이터프레임을 생성해 변수명과 중요도 점수를 나란히 매칭합니다.
    feature_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    
    # sort_values(by='Importance', ascending=False)를 통해 가장 영향력이 큰 변수부터 내림차순으로 정렬하여 출력합니다.
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    print(feature_imp_df)

---

## 8. 결과 해석 및 비즈니스 가치 연결

머신러닝 결과를 바탕으로 비즈니스 의사결정권자에게 제안할 수 있는 **인사이트(Insight)** 예시입니다.

* **인사이트 1:** "가입 후 **사용기간(Tenure)**이 3개월 미만인 고객의 이탈률이 가장 높습니다."
  * ➡️ *액션 플랜:* 신규 가입자 대상 웰컴 키트 및 첫 3개월 특별 집중 케어 프로모션 진행
* **인사이트 2:** "**고객센터 문의 횟수(CustomerServiceCalls)**가 3회 이상인 순간 이탈 확률이 70%를 돌파합니다."
  * ➡️ *액션 플랜:* 고객센터 3회 이상 문의 기록 발생 시, 즉각 요금 할인 혜택 부서나 해지 방지 전담반으로 콜을 이관 및 선제적 대응
* **인사이트 3:** "**월요금(MonthlyCharges)**이 상위 10%인 VIP 고객들의 이탈률이 증가하고 있습니다."
  * ➡️ *액션 플랜:* 고가치 고객 전용 결합 할인 혜택 및 멤버십 서비스 강화

---

## 9. 실무에서 가장 많이 범하는 3대 실수

* ❌ **실수 1: 오직 정확도(Accuracy)만 맹신함**
  * *현실:* 전체 고객의 95%가 유지되고 5%만 이탈하는 불균형 데이터셋에서는, 모델이 아무 일도 안 하고 무조건 '유지(0)'라고만 답해도 정확도가 95%가 나옵니다. 정작 이탈자 5%는 한 명도 찾지 못했으므로 쓸모없는 모델입니다. **반드시 Recall과 F1-Score를 보아야 합니다.**
* ❌ **실수 2: 임계값(Threshold)을 무조건 0.5로 고정함**
  * *현실:* 비즈니스 상황(이탈 예방 혜택 비용 vs 이탈 시 손실 비용)의 손익분기점을 따져 임계값을 최적화해야 비즈니스 이익이 극대화됩니다.
* ❌ **실수 3: 데이터 전처리 없이 복잡한 최신 모델만 돌림**
  * *현실:* 데이터 클리닝과 정제, 도메인 지식을 활용한 파생 변수(Feature Engineering) 생성 없이 최신 모델(XGBoost, LightGBM 등)만 적용하면 절대 실무 수준의 높은 성능을 얻을 수 없습니다.