# DeepFake Adversarial Attack 연구

## 관련 논문

1. **한국인 논문:**
   - 제목: [Exploiting Style Latent Flows for Generalizing Deepfake Video Detection](https://arxiv.org/pdf/2403.06592)

2. **핵심 논문:**
   - 제목: [Face Poison: Obstructing DeepFakes by Disrupting Face Detection](https://ieeexplore.ieee.org/document/10220056)

3. **관련 논문:**
   - 제목: [Landmark Breaker Obstructing DeepFake](https://arxiv.org/abs/2102.00798)

---

## 연구 핵심 요약

**모든 딥페이크 모델을 대상으로 강건한 방어 전략:**

현재 모든 방법론은 특정 모델의 학습 데이터를 망가뜨리는 것이 아닌, **어떠한 방식을 통해 학습 과정 자체를 방해하는** 방법론입니다. 해당 논문의 방법론은 GitHub 코드가 제공되지 않으므로, 논문을 기반으로 직접 구현이 필요합니다.

---

## 논문 상세 요약: Face Poison

### 주요 내용 정리

1. **특정 모델을 타겟으로 한 공격:**
   - DeepFake 모델이 아닌 **Face-Detector 모델을 공격**합니다. 모든 딥페이크 모델은 얼굴 인식이 필수이기 때문에, 이 방식을 사용하면 모든 딥페이크 모델에 강건하게 대응할 수 있습니다.

2. **Multi-scale feature-level adversarial attack:**
   - 특정 모델의 **여러 레이어를 타겟으로 공격**하여 이미지를 변형합니다.

3. **훈련 데이터의 이미지 교란:**
   - 변형된 이미지는 육안으로는 멀쩡해 보이지만, **딥페이크 모델이 훈련을 통해 생성하는 결과를 무너뜨립니다.**
   - 특히 **얼굴 검출 과정에서** 라벨링된 얼굴 이미지가 공격을 받습니다.

---

### 추가 내용 정리

1. **Multi-scale feature-level adversarial attack (MSFLA)**:
   - SOTA(Sota) Face-Detector 모델들은 모두 DNN 기반입니다.
   - 인코더는 ReLU, CNN, FCNN 등 여러 레이어로 이루어진 딥러닝 모델입니다.
   - MSFLA는 **특정 레이어를 타겟으로 삼아 공격**을 수행합니다.
   - 원본 이미지를 통과시킬 때, **특정 레이어에서 약간의 교란(Poison)을 주어 이미지가 변형**됩니다.
   - 변형된 이미지는 육안으로는 멀쩡해 보이지만, **Face-Detector는 얼굴을 인식하지 못하게 됩니다.**
   - **이미지 자체를 공격하는 것이 아닙니다.** 얼굴 인식 시스템에만 영향을 미칩니다.

---

## 참고 링크
- [Exploiting Style Latent Flows 논문 PDF](https://arxiv.org/pdf/2403.06592)
- [Face Poison 논문 PDF](https://ieeexplore.ieee.org/document/10220056)
- [Landmark Breaker 논문 PDF](https://arxiv.org/abs/2102.00798)

