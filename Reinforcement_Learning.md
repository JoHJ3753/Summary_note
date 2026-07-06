# Stable-Baselines3를 배우기 전에 꼭 알아야 하는 구조

## 1. ROS2 전에 SB3를 배우는 이유
**"강화 학습의 입출력 구조를 이해해야 하기 때문, SB3는 ROS2 강화학습으로 가기 전의 훈련용 안전지대"**

## 2. Stable-Baselines3
**강화학습 알고리즘을 직접 수식부터 구현하지 않고도, 검증된 형태로 바로 사용할 수 있게 해주는 라이브러리**

## 3. 강화학습 최소 핵심 개념
## 3가지에 집중
- 환경(Environment)
- 보상(Reward)
- 알고리즘 선택(Algorithm Selection)

### 환경(Environment)
#### 에이전트가 상호작용하는 세계

### 관측(Observation)
#### 에이전트가 보는 입력값
### 많이 하는 실수
#### "내가 알고 싶은 값"과 "에이전트가 실제로 받는 값"을 혼동하는 것

### 행동(Action)
#### 에이전트가 내리는 결정
### SB3에서 중요한 기준
#### 행동 공간이 이산형인지 연속형인지

### 보상(Reward)
#### "잘했는가, 못했는가"를 숫자로 알려주는 신호
#### 강화학습은 "사람이 원하는 행동"이 아니라 **보상이 큰 행동**을 학습함

### 에피소드(Episode)
#### 한 번의 시도
#### Gymnasium에서는 종료가 `terminated`와 `truncated`로 나뉩니다.

- `terminated`: 목표 달성, 실패 등 **문제 자체가 끝난 상태**
- `truncated`: 시간 제한 등 **외부 조건으로 잘린 상태**
#### 이 차이는 부트스트래핑과 학습 해석에서 중요합니다.

## 4. SB3에서 가장 먼저 이해해야 하는 실행 흐름
### **전체흐름**
1. 환경을 만든다.
2. 환경을 `reset()` 해서 초기 관측을 받는다.
3. 정책이 관측을 보고 행동을 고른다.
4. 환경이 행동을 받아 다음 상태와 보상을 돌려준다.
5. 이 데이터를 모아서 알고리즘이 정책을 업데이트한다.
6. 반복한다.

즉, 핵심은 **“관측 → 행동 → 보상 → 업데이트” 반복 구조**입니다.

## 5. 알고리즘을 처음 배울 때 반드시 구분해야 할 3가지
### DQN
**"대표적인 이산 행동 공간용 알고리즘"**
#### 적합한 예:

- 좌/우 이동
- 정지/전진/후진
- 버튼형 제어

부적합한 예:

- 조향각 0.13
- 관절 토크 0.42
- 연속 속도 제어
### A2C
**"여러 워커를 사용하며 replay buffer 없이 학습하는 정책 기반 계열 알고리즘"**
#### 장점:

- 구조가 비교적 직관적
- PPO보다 개념 설명용으로 좋음

단점:

- PPO보다 실전 기본값으로는 덜 선택되는 경우가 많음
### PPO
**"A2C와 TRPO의 아이디어를 결합하고, 업데이트가 너무 크게 변하지 않도록 clipping을 사용하는 알고리즘"**
#### 초보자 첫 실습 추천 이유:

- 비교적 안정적
- 자료가 많음
- CartPole, LunarLander, 간단한 로봇 시뮬레이션 예제에 많이 사용
- ROS2/Gazebo/MuJoCo 확장 시 개념 연결이 쉬움
### 첫 메인 알고리즘은 PPO로 진행하는 것이 좋다.

## 6. 실무에서 자주 하는 오해
### 오해 1.SB3를 배우면 강화학습을 다 아는 것이다. -> X
### 핵심
- 상태 설계
- 행동 정의
- 보상 설계
- 종료 조건 설계
- 평가 기준

### 오해 2. timesteps만 크게 늘리면 성능이 오른다. -> X
#### 환경 설계와 보상 설계가 잘못되면 1만 step이든 100만 step이든 이상한 행동만 더 강화됩니다.

### 오해 3. reward가 올라가면 무조건 성공이다 -> X
#### 보상 해킹이 일어나면 숫자는 오르는데 실제 행동은 실패할 수 있습니다.

### 오해 4. ROS2와 붙이면 더 진짜 같으니 더 빨리 배운다 -> 오히려 반대
#### 처음엔 **작고 단순한 Gymnasium 환경**에서 확실히 익히는 것이 훨씬 빠릅니다.

## 학습 목표
- 강화학습의 기본 용어를 설명할 수 있다.
- Gymnasium 환경이 어떤 형태로 동작하는지 안다.
- SB3가 어떤 역할을 하는지 안다.
- PPO, A2C, DQN의 차이를 아주 큰 그림에서 구분할 수 있다.
- 왜 ROS2 전에 SB3를 배우는지 설명할 수 있다.

## 미니 실습
```py
import gymnasium as gym

env = gym.make("CartPole-v1")

obs, info = env.reset(seed=42)
print("초기 관측값:", obs)

for step in range(5):
    action = env.action_space.sample()  # 랜덤 행동
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\n[step {step+1}]")
    print("action:", action)
    print("obs:", obs)
    print("reward:", reward)
    print("terminated:", terminated)
    print("truncated:", truncated)

    if terminated or truncated:
        obs, info = env.reset()
        print("환경이 종료되어 reset() 수행")

env.close()
```
- `obs`는 상태 입력
- `action`은 에이전트가 한 행동
- `reward`는 즉시 보상
- `terminated`, `truncated`는 종료 여부
- 종료되면 반드시 `reset()` 필요

# Stable-Baselines3 설치 + 첫 PPO 실습
## 1. 실습 목표
- SB3 설치 및 실행 환경 구축
- PPO로 CartPole 학습 실행
- 학습 모델 저장 / 불러오기
- 학습된 정책으로 실제 플레이 확인
- “왜 학습이 되는지” 감각 이해

## 2. 개발 환경
- Python 3.9 ~ 3.11
- Anaconda 또는 venv
- VSCode
- CPU만으로도 충분 (GPU 필요 없음)

## 3. 설치
### 가상 환경 생성
```bash
conda create -n sb3_env python=3.10 -y
conda activate sb3_env
```

### 필수 패키지 설치
```bash
pip install stable-baselines3[extra]
pip install gymnasium
pip install shimmy
```

## 4. PPO 실습 코드
```py
import gymnasium as gym
from stable_baselines3 import PPO

# 1. 환경 생성
env = gym.make("CartPole-v1")

# 2. 모델 생성 (가장 핵심)
model = PPO("MlpPolicy", env, verbose=1)

# 3. 학습 수행
model.learn(total_timesteps=10000)

# 4. 모델 저장
model.save("ppo_cartpole")

env.close()
```

## 5. 코드 해설(실무 관점)
### 환경 생성
```py
env = gym.make("CartPole-v1")
```
### 의미
- RL 세계 생성
- 상태, 행동, 보상 규칙 포함

### 모델 생성
```py
model = PPO("MlpPolicy", env, verbose=1)
```
### 핵심 구조
- `"MlpPolicy"`
→ 입력: 상태
→ 출력: 행동
→ 내부: 신경망
- `env`
→ 어떤 문제를 풀 것인가
- `verbose=1`
→ 학습 로그 출력

### 학습
```py
model.learn(total_timesteps=10000)
```
### 내부에서 일어나는 일(반복):
1. 상태 관측
2. 행동 선택
3. 보상 획득
4. 데이터 저장
5. 정책 업데이트

### 모델 저장
```py
model.save("ppo_cartpole")
```
### 실제 파일
```
ppo_cartpole.zip
```

## 6. 학습된 모델 실행 (가장 중요한 부분)
```py
import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode="human")

model = PPO.load("ppo_cartpole")

obs, _ = env.reset()

for _ in range(500):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
```

## 7. 여기서 꼭 봐야 하는 포인트
### 학습 전
- 랜덤 → 막대 바로 떨어짐
### 학습 후
- 균형 유지
- 오래 버팀
**"이 변화가 강화학습의 핵심"**

## 8. PPO가 실제로 잘 되는 이유 (직관 설명)
PPO는 다음을 합니다:
- 잘한 행동 → 더 많이 하도록 강화
- 나쁜 행동 → 덜 하도록 감소
- 업데이트 너무 크게 하지 않음 (clipping)

**"그래서 안정적이다"**

## 9. 실무에서 가장 많이 하는 실수 TOP 5
### 1. timesteps 너무 작음
```python
model.learn(1000)  # 너무 적음
```
최소:
```
10000 ~ 100000
```

### 2. render_mode 없이 실행
결과 안 보임

### 3. 학습 없이 predict
아무것도 못함

### 4. env.close() 안 함
메모리 누수 가능

### 5. 학습 결과 해석 안 함
그냥 돌아가는 것만 보고 끝냄

# DQN vs A2C vs PPO 비교 실습
## 1. 목표
- DQN / A2C / PPO 차이를 직관적으로 이해
- 행동 공간에 따라 알고리즘 선택 가능
- 실무에서 잘못된 선택을 피할 수 있음
- ROS2/로봇 환경에서 어떤 알고리즘이 맞는지 판단 가능

## 2. 세 알고리즘 한 줄 정의
| 알고리즘 | 핵심 개념 |
| --- | --- |
| **DQN** | Q값 테이블을 신경망으로 근사 |
| **A2C** | 정책 + 가치 함수 동시에 학습 |
| **PPO** | 정책을 안정적으로 업데이트 |

## 3. 가장 중요한 기준 (이거 하나로 80% 결정됨)
### ✔ 행동 공간 기준
| 행동 공간 | 추천 알고리즘 | 이유 |
| --- | --- | --- |
| **이산**<br>(좌/우 버튼) | DQN, A2C, PPO | | 
| **연속**<br>(속도, 토크) | A2C, PPO | DQN은 연속 공간을 다루기 어려움 |
### 핵심 결론
**"DQN은 연속 제어에 거의 사용 불가"**

## 4. 실습 1-DQN(이산 행동)
```py
import gymnasium as gym
from stable_baselines3 import DQN

# 1. 환경 생성 (이산 행동)
env = gym.make("CartPole-v1")

# 2. 모델 생성
model = DQN("MlpPolicy", env, verbose=1)

# 3. 학습
model.learn(total_timesteps=10000)

# 4. 저장
model.save("dqn_cartpole")

env.close()
```

### 특징
- action: 0 or 1 (좌/우)
- Q값 기반 학습

### 장점
- 구조 단순
- 이산 문제에서 강력

### 단점
- 연속 제어
- 불안정할 수 있음

## 5. 실습 2-A2C
```py
import gymnasium as gym
from stable_baselines3 import A2C

# 1. 환경 생성 (이산 행동)
env = gym.make("CartPole-v1")

# 2. 모델 생성
model = A2C("MlpPolicy", env, verbose=1)

# 3. 학습
model.learn(total_timesteps=10000)

# 4. 저장
model.save("a2c_cartpole")

env.close()
```

### 특징
- Actor (행동 결정)
- Critic (평가)

### 장점
- 구조 이해 쉬움
- PPO보다 가벼움

### 단점
- PPO보다 성능 낮은 경우 많음

## 6. 실습 3-PPO
```py
import gymnasium as gym
from stable_baselines3 import PPO

# 1. 환경 생성 (이산 행동)
env = gym.make("CartPole-v1")

# 2. 모델 생성
model = PPO("MlpPolicy", env, verbose=1)

# 3. 학습
model.learn(total_timesteps=10000)

# 4. 저장
model.save("ppo_cartpole")

env.close()
```

## 7. 결과 비교
각 알고리즘 실행 후 아래를 비교하세요:

- 평균 reward
- 학습 속도
- 안정성 (진동 여부)
- episode 길이

## 8. 핵심 차이 (직관 정리)
### DQN
- Q값 기반
- 이산 전용
- 단순하지만 제한적

### A2C
- 정책 기반
- actor + critic 구조

### PPO
- 안정성 최강
- 실무 기본 값

## 9. 실무 선택 기준 (중요)
### 상황별 추천
| 상황 | 추천 |
| --- | --- |
| 게임 (버튼 입력) | DQN |
| 로봇 제어 | PPO |
| 연속 제어 | PPO |
| 빠른 테스트 | A2C |
| 안정성 중요 | PPO |

### ROS2 기준
| 환경 | 추천 |
| --- | --- |
| Turtlebot 이동 | PPO |
| 로봇팔 제어 | PPO |
| 간단 버튼형 시뮬레이션 | DQN |

### 결론
**"ROS2 + 강화학습 = 거의 무조건 PPO 부터 시작"**

## 10. 실무에서 가장 큰 실수
### 실수 1
DQN으로 로봇 제어하려고 함
#### 결과:
- 로봇이 이산적으로 움직임
- 실제 로봇과 괴리감
- 제어가 불안정

### 실수 2
알고리즘 바꿔도 환경 그대로
- 문제는 알고리즘이 아니라
- 보상 설계인 경우 많음

### 실수 3
성능 비교 안 함
- 항상 최소 2개 이상 비교해야 함

# 학습이 안 되는 진짜 이유 (하이퍼파라미터 + 보상 설계)
## 1. 목표
- 왜 학습이 안 되는지 스스로 진단 가능
- 하이퍼파라미터를 “숫자”가 아니라 “의미”로 이해
- 보상 설계를 실전 수준으로 설계 가능
- ROS2 환경에서도 문제 원인 구분 가능

## 2. 강화학습이 실패하는 원인
### 보상 설계 실패 (80% 원인) (가장 중요)
- 에이전트가 "원하는 행동"이 아니라 "보상이 큰 행동"을 학습

### 하이퍼파라미터 설정 오류
- 학습 자체가 불안해짐

### 상태/행동 정의 문제
- 아예 학습 불가능

## 3. 하이퍼파라미터 핵심
### 감가율(gamma)
$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$
### 의미
- 미래 보상을 얼마나 중요하게 볼 것인가

### 직관
| gamma | 의미 |
| --- | --- |
| 0.5 | 현재만 중요 |
| 0.99 | 미래도 매우 중요 |

### 실무 기준
- CartPole: 0.99
- 로봇 제어: 0.98 ~ 0.999

### 실수
- gamma 너무 작음 → 단기 행동만 학습
- gamma 너무 큼 → 학습 느림

### 학습률(learning_rate)
### 의미
- 정책을 얼마나 빠르게 바꿀 것인가

### 직관
| 값 | 결과 |
| --- | --- |
| 너무 큼 | 발산 |
| 너무 작음 | 학습 안됨 |
```py
learning_rate = 3e-4  # PPO 기본
```

### n_steps(PPO 핵심)
### 의미
- 몇 step 모아서 학습할 것인가

### 직관
| 값 | 특징 |
| --- | --- |
| 작음 | 빠르지만 불안정 |
| 큼 | 안정적이지만 느림 |
```py
# PPO 기본값
n_steps = 2048
```

### batch_size
### 의미
- 한 번에 업데이트 할 데이터 크기

### 직관
| 값 | 특징 |
| --- | --- |
| 작음 | noisy |
| 큼 | 안정적 |

### 실무 추천
```py
# PPO 기본값
batch_size = 64
```

## 4. 하이퍼파라미터 변경 실험
```py
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1")

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    gamma=0.99,
    n_steps=2048,
    batch_size=64,
    verbose=1
)

model.learn(total_timesteps=10000)
```

### 실험 1

```python
gamma=0.5
```
결과:
- 미래 고려 안함
- 금방 실패

## 실험 2
```python
learning_rate=0.01
```
결과:
- 발산
- 행동 이상해짐

## 실험 3
```python
n_steps=32
```
결과:
- 불안정
- reward 들쭉날쭉

## 5. 보상 설계(중요)
### 좋은 보상
```py
reward = -distance_to_goal
```
특징:
- 점진적 학습 가능

### 나쁜 보상
```python
if goal:
    reward = 100
else:
    reward = 0
```
문제:
- 학습 거의 안됨

### 보상 설계 3대 원칙
1. Dense Reward (촘촘한 보상)
    - 매 step마다 피드백
2. 방향성 있는 보상
    - 목표 방향으로 유도
3. 악용 방지
    - 편법 행동 차단

## 실무에서 자주 발생하는 보상 버그

### 문제 1
벽에 계속 부딪히는 행동 학습
- 이유:
    - 충돌 penalty 없음
### 문제 2
제자리에서 흔들기
- 이유:
    - 작은 보상 반복 exploit

### 문제 3
목표 근처에서 멈춤
- 이유:
    - 종료 조건 없음

## 6. 실전 디버깅 방법(진짜 중요)
## reward 그래프 확인
올라가는가?

## 행동 직접 보기
이상한 행동인가?

## 랜덤 정책 비교
학습이 의미 있는가?

## 환경 단순화
너무 복잡하면 실패

## 7. ROS2 연결 시 핵심 포인트
| 요소 | 중요 포인트 |
| --- | --- |
| 상태 | 센서 값 (노이즈 포함) |
| 행동 | 속도 / 토크 |
| 보상 | 목표 + 안정성 |
| 종료 | 충돌 / 목표 도달 |

### 대부분의 실패 원인
**보상 설계 실패 + 센서 정의 오류**

# 커스텀 Gymnasium 환경 만들기
## 1. 목표
- `reset()` / `step()` 구조 완전 이해
- 상태 / 행동 / 보상 직접 설계
- SB3에 커스텀 환경 연결 가능
- ROS2 환경으로 확장할 수 있는 기반 확보

## 2.Gymnasium 환경의 본질 (핵심구조)
강화 학습 환경은 아래 함수 2개입니다.
### reset()
```py
obs, info = env.reset()
```
### 역할:
- 환경 초기화
- 시작 상태 반환

### step()
```py
obs, reward, done, truncated, info = env.step(action)
```
### 역할:
- 행동 적용
- 보상 계산
- 다음 상태 반환
- 종료 여부 반환

## 3. 가장 중요한 구조 (무조건 암기)
```py
while True:
    action = agent(obs)
    obs, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        obs = env.reset()
```
#### 모든 강화학습의 핵심 루프
#### 한 줄 씩 뜯어보기
```py
while True:
```
- 무한 루프: 게임이나 시뮬레이션을 멈추지 않고 계속해서 실행합니다.
```py
action = agent(obs)
```
- 행동 선택: 에이전트(AI)가 현재 관측 상태(obs,Observation)를 보고, 자신이 가진 정책에 따라 어떤 행동(action)을 할지 결정합니다
```py
obs, reward, terminated, truncated, _ = env.step(action)
```
- 환경의 변화: 에이전트가 결정한 action을 환경(env)에 전달하여 한 스텝 실행합니다(env.step). 그 결과 환경은 에이전트에게 5가지 결과물을 돌려줍니다.
- obs (새로운 상태): 행동을 취한 후 바뀐 다음 화면이나 로봇의 상태입니다.
- reward (보상): 방금 한 행동이 잘한 건지 못한 건지 점수(숫자)로 알려줍니다.
- terminated (종료 여부): 게임 오버(예: 로봇이 넘어짐) 또는 미션 성공(예: 목적지 도달)으로 인해 에피소드가 정상적으로 끝났는지 여부(True/False)입니다.
- truncated (중단 여부): 정상 종료는 아니지만, 시간 제한(예: 1,000 스텝 초과) 같은 외부 요인으로 인해 강제로 중단되었는지 여부(True/False)입니다.
- _ (부가 정보): 디버깅용 데이터 등이 들어있는데, 지금은 안 쓰겠다는 의미로 언더바(_) 처리해 둔 것입니다.
```py
if terminated or truncated:
    obs = env.reset()
```
- 에피소드 초기화: 게임 오버가 되었거나(terminated), 시간 제한이 다 되었으면(truncated) 게임을 처음부터 다시 시작해야 합니다.env.reset()을 호출하여 환경을 처음 상태로 되돌리고, 새로운 첫 번째 관측값(obs)을 받아와 다음 에피소드를 준비합니다.

## 4. 커스텀 환경 만들기
### 목표
"목표 지점으로 이동하는 1차원 환경"

### 코드
```py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleEnv(gym.Env):

    def __init__(self):
        super(SimpleEnv, self).__init__()

        # 상태: 위치 (연속값)
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(1,), dtype=np.float32
        )

        # 행동: -1 or +1 (좌우 이동)
        self.action_space = spaces.Discrete(2)

        self.state = None
        self.goal = 5.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.state = np.array([0.0], dtype=np.float32)
        return self.state, {}

    def step(self, action):

        # 행동 적용
        if action == 0:
            self.state[0] -= 1
        else:
            self.state[0] += 1

        # 보상 설계
        distance = abs(self.goal - self.state[0])
        reward = -distance

        # 종료 조건
        terminated = distance < 0.5
        truncated = False

        return self.state, reward, terminated, truncated, {}
```
## 5. 코드 해설
### Observation_space
```py
spaces.Box(...)
```
### 의미
- 상태 범위 인정

### action_space
```py
spaces.Discrete(2)
```
### 의미
- 행동 개수

### state
- 현재 상태(위치)

### reward
```py
reward = -distance
```
### 핵심
- 목표에 가까울수록 reward 증가

### 종료 조건
```py
terminated = distance < 0.5
```
- 목표 도달 시 종료

## 6. SB3 연결 (핵심)
```py
from stable_baselines3 import PPO

env = SimpleEnv()

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)
```

## 7. 학습 결과 테스트
```py
obs, _ = env.reset()

for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)

    print("state:", obs, "reward:", reward)

    if terminated:
        print("목표 도달!")
        break
```
---

```py
obs, _ = env.reset()
```
- 환경 초기화: 게임을 시작하기 위해 판을 새로 짭니다. 환경(env.reset())을 초기 상태(위치 [0.0])로 되돌리고, 에이전트에게 첫 번째 눈앞의 화면(obs)을 건네줍니다. 뒤의 언더바(_)는 지금은 쓰지 않는 부가 정보(info)를 받아두는 빈 상자입니다.

```py
for _ in range(50):
```
- 제한 시간 설정: 에이전트에게 딱 50번의 기회(스텝)만 주겠다는 반복문입니다. 50번 움직이는 동안 목표에 도달해야 합니다.

```py
    action, _ = model.predict(obs)
```
- AI의 판단: 학습된 신경망 모델(model)에게 "지금 상태가 obs인데 어떤 행동을 해야 하니?"라고 물어봅니다. 모델은 자신의 뇌(정책)를 굴려 가장 좋다고 생각하는 행동(action, 왼쪽 0 또는 오른쪽 1)을 결정해 알려줍니다.

```py
    obs, reward, terminated, truncated, _ = env.step(action)
```
- 주사위는 던져졌다: 에이전트가 고른 action을 환경에 실제로 입력(env.step)합니다. 그 결과로 발걸음을 한 칸 옮긴 새로운 위치(obs), 그 행동으로 인해 얻은 점수(reward), 목표에 도달했는지 여부(terminated), 시간 제한에 걸렸는지 여부(truncated)를 한꺼번에 리턴받습니다.

```py
    print("state:", obs, "reward:", reward)
```
- 상황 중계: 매 스텝마다 에이전트가 현재 어디에 위치해 있고(state), 방금 한 행동으로 점수(reward)를 얼마나 까먹거나 얻었는지 화면에 출력하여 모니터링합니다.

```py
    if terminated:
        print("목표 도달!")
        break
```
조기 퇴근: 만약 에이전트가 움직이다가 목표 지점(5.0 근처)에 안전하게 도달해서 terminated가 True가 되었다면, "목표 도달!"을 시원하게 출력하고 50번을 다 채우지 않더라도 반복문(for 루프)을 탈출(break)하여 게임을 종료합니다.

## 8. 실무에서 가장 많이 하는 실수
1. observation_space 잘못 정의
- 실제 값 범위와 다름
- 학습 이상
2. reward 방향 반대로 설정
```py
reward = distance
```
#### 결과:
- 목표에서 멀어짐
3. 종료 조건(terminated/truncated) 설정 오류
- 무한 루프 -> 학습 망함
4. 상태 업데이트 안함
- 항상 같은 상태 → 학습 불가능
5. action 적용 안됨
- agent가 아무 영향 없음

## 9. ROS2 연결 관점 (매우 중요)
구조가 그대로 확장됨
### 현재 구조
```
SB3 -> Gym Env -> state/reward
```
### ROS2 확장
```
SB3 -> ROS2 Env -> 센서 -> 상태
                    ↓    
                액추에이터 -> 행동
                    ↓
                보상 계산
```
### 핵심
- `state` → 센서값 (LiDAR, camera, joint)
- `action` → velocity / torque
- `reward` → 목표 + 안정성

## 10. 실습 과제 (필수)
## 과제 1
목표 위치 바꿔보기
```python
self.goal = -3.0
```
- 에이전트가 방향 바꾸는지 확인

## 과제 2
행동을 3개로 변경
```python
spaces.Discrete(3)
```
- 왼쪽
- 정지
- 오른쪽

## 과제 3
보상 변경
```python
reward = -distance - 0.1
```
- 이동 비용 추가

# 평가 / 로그 / 영상 기록 / 학습 분석
## 1. 목표
- 학습 결과를 **정량적으로 평가**할 수 있음
- reward 그래프를 보고 상태 판단 가능
- 영상으로 행동 검증 가능
- TensorBoard 로그 해석 가능
- “학습 실패 vs 성공” 구분 가능

## 2. 평가가 왜 중요한가?
초보자는 보통 이렇게 합니다:
```python
model.learn(10000)
```
-> 끝

### 문제:
- 잘 된 건지 모름
- 더 좋아질 수 있는지 모름
- 버그인지 모름

### 그래서 반드시 필요:
- 평균 reward
- episode 길이
- 행동 패턴
- 학습 곡선

## 3. 공식 평가 방법 (가장 중요)
### evaluate_policy 사용
```py
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=10
)

print("평균 reward:", mean_reward)
print("표준편차:", std_reward)
```

### 코드 한 줄씩 뜯어보기
```py
from stable_baselines3.common.evaluation import evaluate_policy
```
- 도구 가져오기: Stable Baselines3에서 제공하는 정책 평가 전용 함수인 evaluate_policy를 불러옵니다. 일일이 반복문을 짜지 않아도 알아서 여러 판을 돌려보고 평균값을 계산해 주는 똑똑한 도구입니다.
```py
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=10
)
```
- **성적 측정하기**: 에이전트(model)를 테스트 환경(env)에 던져주고 실력을 평가합니다.
- model: 우리가 학습시킨 AI 에이전트입니다.
- env: 에이전트가 시험을 치를 시뮬레이션 환경(여기서는 SimpleEnv)입니다.
- n_eval_episodes=10: 총 10판(에피소드)의 시험을 보게 하겠다는 뜻입니다. 강화학습 환경은 무작위성이 존재할 수 있으므로, 딱 한 판만 보고 판단하는 게 아니라 10판을 수행하게 합니다.
- mean_reward (결과 1): 10판 동안 얻은 총 보상들의 평균값을 돌려받습니다. (에이전트의 평균 실력)
- std_reward (결과 2): 10판 동안 얻은 총 보상들의 표준편차를 돌려받습니다. (에이전트가 얼마나 기복 없이 안정적으로 잘하는지 나타내는 지표)
```py
print("평균 reward:", mean_reward)
print("표준편차:", std_reward)
```
- 결과 출력: 계산된 평균 점수와 점수의 널뛰기 정도(표준편차)를 콘솔 창에 출력합니다.

### 해석 방법
| 결과 | 의미 |
| --- | --- |
| reward ↑ | 성능 좋음 |
| std ↓ | 안정적 |
| std ↑ | 불안정 |

#### 실무 기준:
- mean_reward만 보면 안 됨
- 반드시 std도 같이 봐야 함

## 4. 로그 기록 (TensorBoard)
### 로그 활성화
```py
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs"
)
```

### 실행
```py
tensorboard --logdir ./logs
```

### 주요 그래프
### rollout/ep_rew_mean
- 평균 보상
### rollout/ep_len_mean
- 에피소드 길이
### train/loss
- 학습 안정성

### 그래프 해석
#### 1. 학습 잘 되는 경우
```
Reward:📈 (점점 오름)
Length:📉 (점점 짧아짐)
Loss: ⚖️ (안정적)
```
#### 2. 학습 이상
```
Reward 평평: 📉 (떨어짐)
Length: 📈 (길어짐)
Loss: 💥 (폭주)
```

#### 3. 학습 정체
```
Reward: ↔️ (변화 없음)
Length: ↔️ (변화 없음)
Loss: ⚖️ (안정적)
```

## 5. 영상 기록 (행동 검증)
### RecordVideo 사용
```py
from gymnasium.wrappers import RecordVideo

env = gym.make("CartPole-v1", render_mode="rgb_array")

env = RecordVideo(
    env,
    video_folder="./videos",
    episode_trigger=lambda x: True
)
```
### 실행
```py
obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()
```
### 결과 확인
- ./videos 폴더에 영상 생성
```
./videos
├── video_0.mp4
├── video_1.mp4
└── ...
```
- 행동 직접 확인 가능

### 왜 중요한가?
- reward만으로는 모르는 걸 알 수 있음
- "어떻게" 잘하는지 확인 가능
- 영상은 최고의 디버깅 툴

## 6. 학습 성공 vs 실패 판단 기준
### 성공
- reward 지속적 증가
- 행동 자연스러움
- 목표 달성

### 실패
- reward 정체 또는 감소
- 이상행동 (관절이 꼬임 등)
- 랜덤과 차이 없음

## 7. 실무 디버깅 체크리스트
## reward 증가하는가?
- NO → 보상 문제
## 행동이 정상인가?
- NO → 상태/행동 문제
## variance 큰가?
- YES → 불안정
## random과 비교
- 차이 없으면 실패
## episode 길이 증가?
- YES → 학습 성공

## 8. 전체 통합 코드
```py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import RecordVideo

# 환경
env = gym.make("CartPole-v1")

# 모델
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs"
)

# 학습
model.learn(total_timesteps=10000)

# 평가
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print("평균:", mean_reward, "표준편차:", std_reward)

# 영상 저장
video_env = gym.make("CartPole-v1", render_mode="rgb_array")
video_env = RecordVideo(video_env, "./videos", episode_trigger=lambda x: True)

obs, _ = video_env.reset()
for _ in range(500):
    action, _ = model.predict(obs)
    obs, _, terminated, truncated, _ = video_env.step(action)

    if terminated or truncated:
        obs, _ = video_env.reset()

video_env.close()
```
### 핵심 블록별로 요약
### 📦 1. 도구 준비 (Imports)

* **`import gymnasium as gym`**: 강화학습 시뮬레이션용 가상 세계 가져오기
* **`from stable_baselines3 import PPO`**: 똑똑한 AI 알고리즘(PPO) 가져오기
* **`evaluate_policy`**: AI의 평균 실력을 측정해 줄 시험관 가져오기
* **`from gymnasium.wrappers import RecordVideo`**: 플레이 화면을 영상으로 찍어줄 카메라 맨 가져오기

### 🏗️ 2. 환경 및 모델 설정 (Setup)

* **`env = gym.make("CartPole-v1")`**: '카트 위 막대기 세우기' 게임판 개설
* **`model = PPO("MlpPolicy", env, verbose=1, ...)`**: 수치 데이터를 학습할 PPO 인공지능 생성 (과정 출력 켬)

### 🏋️‍♂️ 3. 훈련 및 평가 (Train & Test)

* **`model.learn(total_timesteps=10000)`**: 에이전트에게 10,000번 움직이며 독학시킴 (학습)
* **`evaluate_policy(model, env, n_eval_episodes=10)`**: 학습된 AI에게 총 10판의 시험을 보게 함
* **`print("평균:", mean_reward, ...)`**: 10판 점수의 평균과 기복(표준편차) 출력

### 📹 4. 영상 녹화 및 저장 (Video)

* **`gym.make(..., render_mode="rgb_array")`**: 영상 촬영이 가능하도록 배경 이미지 픽셀 추출 모드 활성화
* **`RecordVideo(video_env, "./videos", ...)`**: 해당 환경에 카메라를 장착하고 `./videos` 폴더에 저장 설정
* **`obs, _ = video_env.reset()`**: 카메라를 켜고 첫 화면 준비
* **`action, _ = model.predict(obs)`**: AI가 화면을 보고 가장 최선의 행동을 결정
* **`video_env.step(action)`**: 결정한 행동을 게임판에 입력 (화면 1프레임 캡처)
* **`if terminated or truncated: reset()`**: 막대기가 쓰러지거나 기한이 다 되면 게임판을 리셋하고 촬영 지속
* **`video_env.close()`**: 카메라를 끄고 녹화된 프레임들을 모아 최종 `.mp4` 영상 파일로 변환

## 9. 실무에서 가장 많이 하는 실수
### TensorBoard 안 씀
- 학습 상태 모름

### 평가 없이 모델 사용
- 실제 성능 모름

### 영상 확인 안 함
- 이상 행동 못 잡음

### reward만 보고 판단
- 잘못된 결론

# ROS2 + 강화학습 연결 구조 설계 (실전 아키텍처)
## 1. 목표
- SB3와 ROS2를 **어떻게 연결하는지 구조 이해**
- 토픽 / 서비스 / 액션을 RL 관점에서 해석 가능
- 실제 로봇 제어용 강화학습 아키텍처 설계 가능
- 기업 프로젝트 수준 설계 가능

## 2. 전체 구조 (핵심)
```
[ Stable-Baselines3 ]
        ↓
    (action)
        ↓
[ ROS2 Node ]
        ↓
[ Robot / Simulator ]
        ↓
    (sensor data)
        ↓
[ State Processing ]
        ↓
    (reward)
        ↓
[ RL Environment ]
        ↓
[ SB3 학습 업데이트 ]
```
### 핵심 개념:
- ROS2를 "Gym Environment로 감싸는 것"이 핵심

## 3. ROS2 요소를 RL로 해석
### ROS2 -> RL 매핑
| ROS2 | RL |
| --- | --- |
| Topic (sensor) | Observation |
| Topic (cmd_vel) | Action |
| Node | Environment |
| Service | Reset |
| Action | Episode control |
### 이 매핑이 이해되면 끝

## 4. 코드 구조 (핵심 설계)
### RL 환경 클래스
```py
class ROS2Env(gym.Env):

    def __init__(self):
        super().__init__()

        self.observation_space = ...
        self.action_space = ...

    def reset(self):
        # ROS2 서비스 호출 (환경 초기화)
        return state, {}

    def step(self, action):

        # 1. ROS2 토픽으로 행동 전송
        # publish(cmd_vel)

        # 2. 센서 데이터 수신
        # subscribe(sensor)

        # 3. 상태 구성
        state = ...

        # 4. 보상 계산
        reward = ...

        # 5. 종료 조건
        terminated = ...
        truncated = False

        return state, reward, terminated, truncated, {}
```
### 핵심
- ROS2 전체를 step()안에 넣는다

### 핵심 블록별로 요약
### 🏗️ 1. 환경 정의 및 공간 설정 (__init__)
- `class ROS2Env(gym.Env)`: ROS2 로봇 시뮬레이션(또는 실물 로봇)을 제어하기 위한 커스텀 Gym 환경 선언
- `self.observation_space = ...`: 로봇이 받아올 센서 데이터(라이다, 카메라, 관절 각도 등)의 형태와 범위 정의
- `self.action_space = ...`: 로봇에게 내릴 명령(바퀴 속도, 로봇 팔 제어 토크 등)의 가짓수나 범위 정의

### 🔄 2. 로봇 위치 초기화 (reset)
- `# ROS2 서비스 호출 (환경 초기화)`: 시뮬레이션 가상 세계(Gazebo/MuJoCo)나 로봇을 시작 위치로 강제 순간이동(Service Call) 시킴
- `return state, {}`: 초기화된 로봇의 첫 상태(센서 값)를 AI 에이전트에게 전달하며 시작

### 🏃♂️ 3. 로봇 제어 및 데이터 수집 (step)
- `# 1. ROS2 토픽으로 행동 전송`: AI가 결정한 행동을 로봇이 알아들을 수 있는 속도 명령(cmd_vel 등) 토픽으로 발행(Publish)
- `# 2. 센서 데이터 수신`: 행동을 취한 후 로봇이 움직였을 때의 레이저나 모터 센서 값을 구독(Subscribe)하여 수집
- `state = ...`: 받아온 센서 데이터들을 AI가 학습하기 좋은 형태(NumPy 배열 등)로 가공하여 상태로 지정
- `reward = ...`: 로봇이 목표에 가까워졌으면 플러스 점수, 벽에 박았으면 마이너스 점수를 계산
- `terminated = ...`: 로봇이 목적지에 도달했거나 벽에 부딪혀서 판이 완전히 끝났는지 체크
- `truncated = False`: 제한 시간 초과 등으로 인해 강제로 중단되었는지 여부 설정
- `return state, reward, ...`: 가공된 상태, 보상, 종료 여부를 AI 알고리즘에게 넘겨주어 다음 행동을 유도

## 5. ROS2 통신 흐름
### Action (행동)
```
SB3 -> action -> ROS2 Publisher -> cmd_vel
```
#### example)
- 선속도
- 각속도
### Observation (상태)
```
LiDAR / Camera / Joint -> ROS2 Subscriber -> state vector
```
### Reward (보상)
```
목표 거리 / 충돌 / 안정성 -> reward 계산
```
### Reset
```
ROS2 Service -> 위치 초기화
```

## 6. TurtleBot 예시 (실무 기준)
### 상태 (Observation)
```py
state = [
    distance_to_goal,
    angle_to_goal,
    lidar_min_distance
]
```
### 행동 (Action)
```py
action = [
    linear_velocity,
    angular_velocity
]
```
### 보상 (Reward)
```py
reward = -distance_to_goal
if collision:
    reward -= 100

if goal_reached:
    reward += 100
```
### 종료 (Done)
```py
terminated = collision or goal_reached
```

## 7. 전체 코드 흐름
```
1. SB3가 action 생성
2. ROS2에 publish
3. 로봇 움직임
4. 센서 데이터 수신
5. state 구성
6. reward 계산
7. SB3 업데이트
```
### 위의 루프가 계속 반복

## 8. 실무 아키텍처
```
[ SB3 Trainer ]
        ↓
[ Gym Wrapper (ROS2Env) ]
        ↓
[ ROS2 Bridge Node ]
        ↓
[ Simulator (Gazebo / MuJoCo) ]
        ↓
[ Sensor Streams ]
        ↓
[ Reward Engine ]
        ↓
[ Logging / Monitoring ]
```

## 9. 실무에서 반드시 고려해야 할 요소
### 1. 동기화 문제
- step() 타이밍 중요
- ROS2는 비동기 구조

### 2. 센서 노이즈
- 실제 환경과 다름

### 3. 지연(latency)
- action → 반응 지연 존재

### 4. reward 설계
- 현실적이어야 함

### 5. 안전성
- 로봇 폭주 방지 필수

## 10. 실무에서 가장 많이 하는 실수
### 1. step() 동기화 안 함
- 데이터 꼬임

### 2. 센서 raw 값 그대로 사용
- 학습 불안정

### 3. reward 너무 단순
- 학습 실패

### 4. reset 제대로 안 됨
- episode 깨짐

### 5. 실제 로봇 바로 적용
- 반드시 시뮬레이터 먼저

## 11. 확장 흐름 (중요)
### 1단계
CartPole (완료)

### 2단계
Custom Gym Env (완료)

### 3단계
ROS2 + Gazebo

### 4단계
MuJoCo / Isaac Gym

### 5단계
실제 로봇 적용

## 12. 전체 과정 요약
### 핵심 흐름
```
SB3 → ROS2 → Sim → Reward → SB3
```
### 핵심 역량
- 환경 설계
- 보상 설계
- 알고리즘 선택
- 학습 분석
- 시스템 통합

# APS + 강화학습(RL) 실무설계
## 1. APS 문제를 RL로 바꾸기 (핵심)
### APS(생산계획) 문제
기존 방식
- 룰 기반 (FIFO, SPT, EDD 등)
- 휴리스틱

### 문제
- 변화 대응 어려움
- 실시간 최적화 불가능
- 복잡한 제약 반영 어려움

### RL로 전환
**"스케줄링을 '의사결정 문제'로 본다"**

### 스케줄링 문제 → RL 문제 매핑
```
이전                     | RL 방식
---------------------------|---------------------------
작업 목록                | 상태(State)             
작업 선택 규칙             | 행동(Action)            
완료/미완료                | 보상(Reward)            
```

## 2. RL로 모델링 (가장 중요)

### 1. 상태(State)
시스템의 현재 상황
```
- 설비 상태 (가동/고장)
- 작업 대기 큐
- 작업별 납기
- 재고 상태
- 공정 진행률
```
### 2. 행동(Action)
무엇을 결정할 것인가
```
- 다음 작업 선택
- 설비 할당
- 작업 순서 변경
```
### 3. 보상(Reward) (핵심)
성능의 기준
```
+ 납기 준수 → +100
+ 설비 가동률 ↑ → +50
- 지연 → -100
- 재고 과잉 → -30
```

### 핵심
Reward 설계 = RL 성능 90%

## 3. 전체 아키텍처 (실무 구조)
### 구조
```
Kafka -> Flink -> RL Engine -> API -> UI
```

### 데이터 흐름
1. Kafka : 실시간 이벤트 수집
2. Flink : 상태 계산 / 특징 생성
3. RL : 의사결정
4. 결과 : DB / UI 반영

## 4. 구성 요소 상세
### 1. KafKa (이벤트 스트림)
- 생산 이벤트 수집
```
- 작업 시작/종료
- 설비 상태
- 재고 변화
```

### 2. Flink (실시간 처리)
- 상태 생성
```
- 현재 WIP
- 병목 구간
- 지연 예측
```
### 3. RL Engine (핵심)
- PPO / DQN 알고리즘
```
- 입력 : 상태
- 출력 : 작업 선택
```
### 4. API 서버
- RL 결과 전달
```
POST /schedule/decision
```
### 5. UI
- Gantt Chart / Dashboard

## 5. RL 모델 선택 기준
### 추천
| 상황 | 알고리즘 |
| --- | --- |
| 단순 이산 선택 | DQN |
| 복잡/연속 | PPO ⭐ |
- APS 대부분 PPO 추천 (현업 표준)

## 6. 실시간 의사결정 흐름
### 전체 흐름
```
1. 이벤트 발생 (KafKa)
2. 상태 계산 (Flink)
3. RL 입력 생성
4. RL -> action 결정
5. 결과 적용 (스케줄 변경)
```

## 7. RL API 예시
```py
# [실무 포인트] FastAPI의 데코레이터를 사용하여 /decision 경로로 들어오는 POST 요청을 처리하는 API 엔드포인트를 정의합니다.
@app.post("/decision")
def get_action(state):

    # [실무 포인트] 모델 연산을 위해 입력받은 환경의 상태(state) 데이터를 PyTorch의 기본 데이터 구조인 32비트 부동소수점 텐서(Tensor)로 변환합니다.
    state_tensor = torch.tensor(state, dtype=torch.float32)

    # [실무 포인트] 신경망 모델에 상태를 입력하여 각 행동(Action)에 대한 확률 분포(probs)를 받아옵니다. (가치 평가는 사용하지 않으므로 _로 처리)
    probs, _ = model(state_tensor)

    # [실무 포인트] 확률이 가장 높은 index(가장 유망한 행동)를 뽑아낸 뒤, 텐서 내의 순수 값만 파이썬 기본 데이터 타입(.item())으로 추출합니다.
    action = torch.argmax(probs).item()

    # [실무 포인트] 클라이언트(로봇 또는 시뮬레이터)가 쉽게 파싱할 수 있도록 JSON 형태로 최적의 행동 체증 결과를 반환합니다.
    return {"action": action}
```
### 구성 요소 요약
- @app.post(...): FastAPI 문법으로, 외부 시스템(예: 로봇 시뮬레이터, AGV 제어기 등)이 현재 상태 데이터를 던지며 요청을 보낼 수 있는 통신 창구를 엽니다.

- torch.~: PyTorch 라이브러리 문법으로, 학습된 인공지능 신경망에 데이터를 넣어 추론(Inference)하고 최적의 행동 결정을 내리는 강화학습 에이전트의 핵심 로직입니다.

## 8. 실무 핵심 포인트 (진짜 중요)
###  1. Reward 설계 실패 = 프로젝트 실패
- KPI 기반 설계 필수

###  2. Offline → Online 단계 필요
```
1. 시뮬레이션 학습
2. Shadow 테스트
3. 실제 적용
```

### 3. 안정성 확보
- Rule + RL hybrid 구조 추천

### 4. Explainability 필요
- 왜 이 선택 했는지 설명

## 9. 최종 구조 (완성 아키텍처)
```
[Kafka]
   ↓
[Flink Feature Engineering]
   ↓
[RL Engine (PPO)]
   ↓
[Decision API]
   ↓
[APS System]
   ↓
[Dashboard]
```

## 한 줄 정리
###  “APS를 상태-행동-보상 구조로 바꾸고, PPO 기반 RL로 실시간 스케줄링 최적화한다”