# 강화학습 기초 개념

### **"보상의 누적치를 최대화하는 방향으로 행동을 학습하는 방법"**
### **"수학적 이론이 아니라 범용적 이론"**

## 1. 강화학습의 4가지 핵심 요소
| 요소 | 의미 |
| --- | --- |
| Agent | 행동하는 주체 (AI, 로봇, 게임 캐릭터) |
| Environment | 환경 (게임 맵, 공장, 현실 세계) |
| Action | 행동 (이동, 선택, 조작) |
| Reward | 보상 (점수, 성공/실패) |

## 2. 머신러닝과 강화학습의 차이

| 구분 | 머신러닝 (기존) | 강화학습 (RL) |
| :--- | :--- | :--- |
| 데이터 | 정제된 **정답 데이터** 필요 | 정답 데이터 **없음** (경험으로 학습) |
| 학습 방식 | "이게 정답이야" (지도학습) | "이렇게 해보니 점수가 올랐네?" (시행착오) |
| 예시 | 이미지 분류 | 게임 플레이, 로봇 제어 |

강화학습은 **"정답없이 스스로 배우는 방식" -> 강인공지능**

## 3. 강화학습의 학습 과정

1. 현재 상태(State) 확인
2. 행동(Action) 선택
3. 환경 반응 → 새로운 상태 + 보상 
4. 더 좋은 행동으로 업데이트

## 한 줄 정리
**“좋은 행동은 더 많이, 나쁜 행동은 덜 하도록 학습하는 구조”**

# 핵심 수학 구조

## 1. 강화학습의 본질

### **“누적 보상(총 보상)을 최대화하는 행동 전략 찾기”**

## 2. 핵심 개념 4개(반드시 암기)
| 개념 | 의미 |
| --- | --- |
| State (s) | 현재 상황 |
| Action (a) | 행동 |
| Reward (r) | 즉시 보상 |
| Policy (π) | 행동 전략 |

## 3. Policy (정책) = “행동 기준”
### 정의
상태 -> 행동을 결정하는 함수

example)
- 상태 : "벽 앞"
- 정책 : "오른쪽으로 이동"

## 중요한 포인트

### Policy는 2가지 유형
| 유형 | 설명 |
| --- | --- |
| Deterministic | 항상 같은 행동 |
| Stochastic | 확률적으로 행동 |
**실무에서는 대부분 확률 정책 사용**

## Value Function (가치 함수)
### 정의
**"이 상태가 얼마나 좋은 상태인가?"**

### 4. 상태 가치 함수

V(s)=E[총 보상∣s]

### 의미
- 현재 상태에서 시작했을 때
- 앞으로 얻을 수 있는 총 기대 보상

### 직관
- 좋은 상태 -> 값 큼
- 나쁜 상태 -> 값 작음

## 5. Q-Value (가장 중요)
### 정의
**“이 상태에서 이 행동을 하면 얼마나 좋은가?”**
### Q 함수

Q(s,a)=E[총 보상∣s,a]

## 핵심 관계

### Value vs Q
```
V(s) = max Q(s,a)
```

- 즉 상태의 가치는 가능한 행동 중 최고값

## 6. 강화학습 전체 구조 (완성 형태)

1. 상태 s 관찰
2. Policy로 행동 a 선택
3. 보상 r 받음
4. 다음 상태 s' 이동
5. Value / Q 업데이트

### 반복 → 최적 정책 학습

## 7. 한방 정리 (시험/면접 핵심)

- Policy → 어떻게 행동할지
- Value → 상태의 좋고 나쁨
- Q-value → 행동의 좋고 나쁨

### 최종 목표
## **최적 Policy 찾기**

# Bellman + Q-Learning + 탐험 전략

## 1. 핵심
### 강화학습의 본질적인 질문
**"지금 행동이 좋은 선택인지 어떻게 판단하지?"**

### 답 -> 미래까지 고려해서 계산해야 한다.

## 2. Bellman Equation (강화학습의 심장)
### 핵심 아이디어

**“현재 가치 = 지금 보상 + 미래 가치”**

### Bellman Equation (Q 버전)
```
Q(s,a) = r + γ * max_a' Q(s',a')
```
### 각 항목 의미
| 요소 | 의미 |
| --- | --- |
| r | 현재 보상 |
| γ (감마) | 미래 할인율 (0~1) |
| s' | 다음 상태 |
| max Q | 다음 상태에서 최고 행동 |

## 핵심 한 줄
### "지금 행동은 미래까지 포함해서 평가한다."

## 3. Q-learning (실무 핵심 알고리즘)
### 정의
**“경험(경로)을 통해 Q 값을 직접 학습하는 알고리즘”**

### Q-learning 공식 (업데이트 규칙)
```
Q(s,a) = Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
```

### 각 항목 의미
| 요소 | 의미 | 역할 |
| --- | --- | --- |
| Q(s,a) | 현재 저장된 Q 값 | 이전에 학습된 가치 |
| α (알파) | **학습률** (0~1) | 새로운 정보를 얼마나 반영할지 |
| [ ] 안의 전체 | **오차(Error)** | 현재 추정값과 실제 경험의 차이 |
| r + γ * max Q | Bellman 기반 목표값 | 이론상 최적 가치 |
| -Q(s,a) | 현재 추정값 | 현재 믿고 있는 가치 |

## [핵심] 오차(Error)의 의미
```
오차 = (실제 경험) - (기존 믿음)
```

- **경험이 더 좋으면** → 오차 > 0 → Q 값 증가
- **경험이 더 나쁘면** → 오차 < 0 → Q 값 감소

## 4. 탐험(Exploration) vs 활용(Exploitation)
### 강화학습의 딜레마
- **활용:** 지금까지 좋았던 행동만 반복 (안전, 성장 정체)
- **탐험:** 새로운 행동 시도 (위험, 새로운 최적의 발견 가능)

## 5. 대표적인 해결책: ε-greedy(epsilon-greedy) 전략 (가장 많이 사용)
### 방식
1. ε 확률 (입실론) : 랜덤 행동 (탐험)
2. 1-ε 확률 : 최적 Q 행동 (활용)

## 핵심
- 초반 Exploration 많이
- 후반 Exploitation 많이

## 6. 전체 구조 연결 (완성)

1. 상태 s 관찰
2. ε-greedy로 행동 선택
3. 보상 r 받음
4. 다음 상태 s'
5. Bellman 기반 Q 업데이트
6. 반복

## 7. 한 줄 정리
### "탐험 하면서 Q값을 업데이트하여 최적 정책을 찾는다."

# DQN (Deep Q-Network)

## 1. DQN이 필요한 이유
### 기존 Q-learning의 문제
Q-table 방식
```
Q(s, a) 값을 표(Table)로 저장
```

문제
- 상태가 많으면 -> 테이블 터짐
- 이미지, 센서 데이터 -> 불가능

## 해결
### "Q값을 테이블 대신 신경망으로 계산하자"

## 2. DQN 핵심 구조
### 개념
**Q(s,a)를 Neural Network가 예측**

### 구조
```
입력 : 상태(s)
출력 : 각 행동의 Q값
```

## 3. DQN 학습 방식
### 기본 흐름
1. 상태 s 입력
2. 신경망 -> Q값 출력
3. 행동 선택
4. 보상 받음
5. Bellman 기반 업데이트

### 핵심 공식(DQN 타겟)
```
y = r + γ * max_a' Q(s',a';θ⁻)
```

## 4. Experience Replay (필수 개념)
### 기존 DQN의 문제
- 연속적인 데이터 -> 불안정한 학습
- 이전 데이터 사용 불가

### 해결
**"경험을 버리지 않고 저장했다가 무작위로 꺼내 쓰자"**

### 구조
```
[s, a, r, s' (경험 저장)]
  ↓ 랜덤 샘플링
[Main DQN (학습)]
  ↓ 타겟 생성
[Target DQN (참조)]
```

### 효과
- 데이터 중복 사용 가능
- 안정적인 학습
- 과적합 가능성 하락

## 5. Target Network (핵심)
### 문제
**동일한 신경망을 계속 업데이트 -> 값이 계속 흔들림 (발산)**

### 해결
**"2개의 네트워크 사용, 타겟용 신경망을 따로 만들자"**

### 방식
1. 메인 신경망 업데이트
2. 일정 주기마다 타겟 신경망 업데이트
```
Target ← Main (일정 주기마다 복사)
```

### 효과
- 안정적인 학습
- 발산 방지

## 6. DQN 전체 구조

```
1. 상태 s 입력
2. ε-greedy로 행동 선택
3. 환경 실행 → (r, s')
4. (s, a, r, s') 저장
5. Replay Buffer에서 랜덤 샘플
6. Target 값 계산
7. Loss 계산
8. 신경망 업데이트
```

## 7. 실무 코드
```py
# 핵심 구조만 요약

q_values = model(state)

# 행동 선택
if random() < epsilon:
    action = random_action()
else:
    action = argmax(q_values)

# 환경 실행
next_state, reward = env.step(action)

# 저장
replay_buffer.append((state, action, reward, next_state))

# 학습
batch = sample(replay_buffer)

target = reward + gamma * max(target_model(next_state))

loss = (q_value - target)^2
```

## 8. 핵심 포인트(중요)
### 반드시 기억
1. Q-table → 작은 문제
2. DQN → 큰 문제 (이미지, 센서)
3. Replay → 안정성
4. Target Network → 발산 방지

## 한 줄 정리
**"신경망으로 Q값을 근사하고, 경험 재사용 + 타겟 분리로 안정화"**

# 고급 알고리즘 (Double / Dueling / Actor-Critic / PPO)
## DQN의 한계
### 핵심 문제
1. Q값 과대평가
    - max 연산 때문에 Q값이 실제보다 높게 추정됨
    ```
    max Q(s,a)
    ```
    - 실제보다 더 크게 평가됨
2. 상태 가치 vs 행동 가치 구분 못함
    - “이 상태가 좋은 건지”
    - “이 행동이 좋은 건지”
    - 분리 안됨

## 2. Double DQN (과대평가 해결)
### 아이디어
**"행동 선택과 평가를 분리"**

### 핵심
| 단계 | 역할 |
| --- | --- |
| Main Network | 행동 선택 |
| Target Network | 행동 평가 |

결과
- 과대평가 감소 → 성능 안정

## 3. Dueling DQN (가치 분리)
### 아이디어
**“상태 가치(Value) + 행동 이점(Advantage) 로 분리”**

### 구조
```
State 입력
↓
State-Value Stream (V)
↓
Action-Advantage Stream (A)

Q(s,a) = V(s) + A(s,a)
```
### 의미
| 요소 | 설명 |
| --- | --- |
| V(s) | 상태 자체의 가치 |
| A(s,a) | 행동의 상대적 가치 |
### 효과
- 학습 속도 증가
- 안정성 상승

## 4. Policy Gradient (완전히 다른 접근)
### 기존 방식

- Value 기반
  - Q-learning
  - DQN

### Policy Gradient

**Policy 자체를 직접 학습**

### 목표

max⁡  E[R]

### 핵심 차이
| 구분 | Value 기반 | Policy 기반 |
| --- | --- | --- |
| 출력 | Q값 | 행동 확률 |
| 방식 | 간접 | 직접 |

## 5. Actor-Critic(현업 핵심 구조)
### 아이디어
**Policy + Value 같이 사용**

### 구조
| 역할 | 설명 |
| --- | --- |
| Actor | 행동 결정 |
| Critic | 평가 |

## 흐름
1. Actor → 행동 선택
2. Critic → 평가 (좋았는지)
3. Actor 업데이트

## 한 줄 정리
**"행동은 Actor, 판단은 Critic"**

## 6. PPO (현재 표준 알고리즘)
### 등장 배경
- Policy Gradient 문제
- 기존 Actor-Critic은 업데이트 폭이 커서 불안정했다.

### 해결
- PPO는 업데이트 폭을 제한하여 안정성을 높였다.

### 의미
- 정책 변화 폭 제한
- 안정적인 학습

### 특징
| 특징 | 설명 |
| --- | --- |
| 안정성 | 매우 높음 |
| 구현 | 비교적 쉬움 |
| 활용 | 가장 널리 사용 |

## 7. 전체 알고리즘 흐름 비교 (핵심 정리)
| 단계 | 알고리즘 | 특징 |
| --- | --- | --- |
| 1 | Q-learning | 기본 |
| 2 | DQN | 딥러닝 |
| 3 | Double DQN | 과대평가 해결 |
| 4 | Dueling DQN | 구조 개선 |
| 5 | Policy Gradient | 정책 직접 학습 |
| 6 | Actor-Critic | 하이브리드 |
| 7 | PPO | 실무 표준 |

## 실무 선택 기준(중요)
### 상황별 추천
| 상황 | 추천 |
| --- | --- |
| 간단 문제 | Q-learning |
| 이미지/게임 | DQN |
| 안정성 필요 | Double/Dueling |
| 연속 행동 (로봇) | Actor-Critic |
| 최신 실무 | PPO |

## 한 줄 정리
**"Q-learning → DQN → Actor-Critic → PPO로 발전하며 안정성과 성능을 개선한 것"**

# PyTorch 기반 DQN 실습 구조
## 1. 전체 구조(매우 중요)
### DQN 시스템 구성
```
1. Environment (Gym)
2. Q-Network (Neural Net)
3. Replay Buffer
4. Target Network
5. 학습 루프
```

## 2. 환경 준비(Gym)
### 설치
```bash
pip install gymnasium torch numpy
```

### 예제 환경
- CartPole (가장 기본 RL 문제)
- 막대 안 쓰러지게 유지

### 코드
```py
import gymnasium as gym

env = gym.make("CartPole-v1")

state, _ = env.reset()

print(state)  # 상태값 (4개)
```

## 3. Q-Network (핵심 모델)
### 역할
- 상태 -> Q값 출력

### 코드
```py
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)
```
### Output
```
[왼쪽: 1.2, 오른쪽: 3.8]
```
오른쪽 선택

## 4. Replay Buffer (경험 저장소)
### 역할
경험 저장 + 랜덤 샘플링

### 코드
```py
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
```

## 5. 학습 핵심 함수
### 코드
```py
def train_step(model, target_model, optimizer, batch, gamma=0.99):
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones)

    # 현재 Q값
    q_values = model(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze()

    # 다음 Q값 (Target Network)
    next_q_values = target_model(next_states).max(1)[0]
    target = rewards + gamma * next_q_values * (1 - dones)

    # Loss
    loss = nn.MSELoss()(q_value, target.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 6. 전체 학습 루프
### 코드 (핵심만)
```py
env = gym.make("CartPole-v1")

state_dim = 4
action_dim = 2

model = QNetwork(state_dim, action_dim)
target_model = QNetwork(state_dim, action_dim)
target_model.load_state_dict(model.state_dict())

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

buffer = ReplayBuffer()

epsilon = 1.0

for episode in range(500):

    state, _ = env.reset()
    done = False

    while not done:

        # ε-greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            q_values = model(torch.tensor(state, dtype=torch.float32))
            action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.push((state, action, reward, next_state, done))
        state = next_state

        # 학습
        if len(buffer) > 64:
            batch = buffer.sample(64)
            train_step(model, target_model, optimizer, batch)

    # 타겟 네트워크 업데이트
    if episode % 10 == 0:
        target_model.load_state_dict(model.state_dict())

    epsilon *= 0.995  # 탐험 감소
```

## 7. 실무에서 반드시 주의할 점
### 1. Reward 설계가 전부다
- RL 성능 = Reward 설계

### 2. Exploration 부족
- 너무 빨리 greedy → 망함

### 3. Target Network 없으면
- 발산 (학습 실패)

### 4. Replay Buffer 필수
- 없으면 overfitting

## 8. 한 줄 정리
**"DQN은 상태를 입력받아 Q값을 예측하고, 경험을 재사용하면서 Bellman 기반으로 업데이트하는 구조"**