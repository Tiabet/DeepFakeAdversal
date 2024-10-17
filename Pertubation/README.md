# 다양한 Perturbation 기법 정리

---

### 1. **FGSM (Fast Gradient Sign Method) - 2014**
- **논문**: [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)  
- **핵심 아이디어**: 손실 함수의 그래디언트 방향으로 교란을 추가해 모델을 공격함.  
- **기법 설명**:  
  - $\eta = \epsilon \cdot \text{sign}\left( \nabla_x J(\theta, x, y) \right)$
- **의의**: 빠르고 계산 효율적이지만 단순해 쉽게 방어될 수 있음.

---

### 2. **CW (Carlini & Wagner Attack) - 2016**
- **논문**: [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644)  
- **핵심 아이디어**: 최적화 기반으로 교란의 은밀함을 극대화해 강력한 공격을 수행함.  
- **기법 설명**:  
  - $f(x') = \max\left( Z(x')_y - \max_{i \neq y} Z(x')_i, -\kappa \right)$
  - $f(x') = \max\left( Z(x')_y - \max_{i \neq y} Z(x')_i, -\kappa \right)$
  - $여기서 \(Z(x')\)는 모델의 로짓 값, \(y\)는 원래 클래스임.$  
- **의의**: 최적화된 교란으로 기존 방어 메커니즘을 쉽게 우회할 수 있음.

---

### 3. **PGD (Projected Gradient Descent) - 2017**
- **논문**: [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)  
- **핵심 아이디어**: 여러 단계로 나눠 반복적인 그래디언트 업데이트를 통해 강력한 교란을 생성함.  
- **기법 설명**:  
  - $x^{(t+1)} = \Pi_{B(x, \epsilon)}\left( x^{(t)} + \alpha \cdot \text{sign}\left( \nabla_x J(\theta, x^{(t)}, y) \right) \right)$
  - $\(B(x, \epsilon)\)은 허용 범위, \(\alpha\)는 학습률임.$
- **의의**: FGSM보다 강력하며 다양한 연구에서 방어 기준으로 자주 사용됨.

---

### 4. **APAA (Adaptive Perturbation Attack Algorithm) - 2021**
- **논문**: [APAA: Adaptive Perturbation Attack Algorithm](https://arxiv.org/abs/2111.13841)  
- **핵심 아이디어**: 모델에 적응적으로 교란을 생성해 공격 효과를 극대화함.  
- **기법 설명**:  
  - 교란 크기와 방향을 모델에 맞게 동적으로 조정해 공격함.  
- **의의**: 정적인 교란 대신 적응형 교란으로 다양한 환경에서 효과적으로 작동함.

---

### 5. **ANDA (Adversarial Noise Disentanglement Attack) - 2022**
- **논문**: [ANDA: Adversarial Noise Disentanglement Attack](https://arxiv.org/abs/2209.11964)  
- **핵심 아이디어**: 교란을 구조적 노이즈와 비구조적 노이즈로 분리해 효율적인 공격을 수행함.  
- **기법 설명**:  
  - 구조적 노이즈는 주요 특징에 영향을 주는 요소, 비구조적 노이즈는 배경 같은 요소로 구분됨.  
- **의의**: 중요한 정보에 집중해 더 효과적인 공격 수행 가능.

---

### 6. **CPO (Critical Part Optimization) - 2023**
- **논문**: [Quantization Matrix Optimization for Adversarial Attacks](https://arxiv.org/abs/2312.06199)  
- **핵심 아이디어**: CNN이 가장 효과적으로 분석할 부분을 찾아 최소한의 교란으로 공격함.  
- **기법 설명**:  
  - 양자화 행렬(Quantization Matrix)을 최적화해 이미지의 주파수(frequency)를 분석함.  
  - 주파수는 인접 픽셀 간 RGB 값 변화 정도를 의미함. 예: 구름 없는 하늘은 낮은 주파수, 사람 얼굴은 높은 주파수임.  
  - 고주파수 부분에 더 높은 가중치를 부여해 강력한 공격 수행.  
- **의의**: CNN 특성을 활용해 효율적이고 효과적인 공격 가능.
