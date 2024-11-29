# DeepFake Adversarial Attack 연구

## 관련 논문

1. **한국인 논문:**
   - 제목: [Exploiting Style Latent Flows for Generalizing Deepfake Video Detection](https://arxiv.org/pdf/2403.06592)

2. **핵심 논문:**
   - 제목: [Nightshade: Prompt-Specific Poisoning Attacks on Text-to-Image Generative Models](https://arxiv.org/abs/2310.13828)
   - 제목: [Face Poison: Obstructing DeepFakes by Disrupting Face Detection](https://ieeexplore.ieee.org/document/10220056)

4. **관련 논문:**
   - 제목: [Landmark Breaker Obstructing DeepFake](https://arxiv.org/abs/2102.00798)

---
## 사용 모델
1. **Deepfake**
2. **FaceDetector**
3. **StableDiffusion**
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

## 논문 상세 요약 : Nightshade
중요 포인트 : 이 방법론은 **SDXL 을 공격하는 방법**에 대해 초점이 맞춰져 있음. (SD, Stable Diffusion 모델들은 전부 text-to-image 모델들임)

하지만 **FaceSwap 같은 유명 딥페이크 모델들은 모두 인코더-디코더 레이어로 이루어진 오토인코더 모델**이어서, 정확히 똑같은 방법으로 딥페이크를 예방하는 것은 힘들 수도 있음. → 이점은 더 찾아봐야함. (SD 모델도 오토인코더 형식이긴 한데 중간에 latent diffusion이라는 알고리즘이 사용되고 있어서 딥페이크하고는 다른 구조라고 봐야함)

**개요 (Part1)**

일반적으로 SD 모델들은 공격하기가 쉽지 않은 것으로 알려져있었음. (워낙 많은 양으로 사전학습 되어 있기 때문)

하지만 저자들은 모델의 약점을 찾아냈음.

1. 특정(Sparse한) Concept나 프롬프트와 연관된 학습 데이터는 그렇게 많지 않았다. 

    (Ex : LAION-Aesthetic) 

1. Poison된 샘플 몇 개만으로 결과를 크게 손상시킬 수 있다.

이 점에 착안해서 나온 것이 NightShade고, 이런 식으로 Online artist들은 자신의 저작물을 지킬 수 있을 것이라고 언급함.

**Part2 ~ 3**

중간에 SD에 대해 자세히 설명하는 파트가 나와서 이 부분도 공부해야 할 것 같음. 

**(Part4)**

저자는 Nightshade의 방법이 유의미하다, 즉 SD 모델들이 소수의 Posion image만으로 공격이 가능하다는 것을 증명하기 위해 선실험을 진행했음. 이 때 사용한 공격 방법은 꽤 단순한데 그냥 개 사진을 줘놓고 텍스트는 고양이라고 주는 방식임.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/988bdfb8-d626-48e3-bb1c-5c3086f7c69c/cfd5cae7-0ffc-4e67-9252-c783a0321559/image.png)

500장으로는 명백한 공격효과가 나오지 않았지만 1000장부터는 유의미한 공격결과가 나왔다고 말함. (Posioned Concept C는 텍스트가 C라는 것이고, Destination Concept A는 이미지가 A라는 것임. 보면 개라는 텍스트를 줬는데 1000장째부턴 고양이를 생성하는 걸 볼 수 있음.)

**Part5 : Nightshade**

Part4처럼 단순하게 텍스트랑 이미지를 잘못 매칭해주는 경우는 문제가 두 개 있음.

1. **사람한테든 모델한테든 Train 과정에서 걸러지기가 쉽다.**
2. **인터넷상에는 Posion된 이미지보다 그렇지 않은 이미지가 압도적으로 많기 때문에 실효성이 떨어진다.**

Nightshade는 이 두 문제를 다음과 같이 극복함.

1. **poison image가 자연스러워 보이기**
2. **적은 수로도 효과적인 Posion을 넣기 (posion potent)**

근데 이 해결법이 상당히 단순함. 우선 별개의 모델이 필요 없이 타겟 모델 하나로 다 해결하는데, 이 모델을 여기선 편의상 SD라고 정리.

1. 공격하고자 하는 **특정 컨셉 (C)의 프롬프트들을 검사**한다. 여기서 검사란, SD의 텍스트 인코딩 파트에 프롬프트들을 다 넣어보고, C라는 단어만을 인코딩했을 때의 결과와 비교한다. 코사인 유사도를 계산한뒤 상위 5000개 프롬프트들 중 랜덤하게 샘플링하여 다음 단계에 활용한다.
2. 위에서 샘플링된 프롬프트들에는 C라는 단어가 무조건 들어있을 것이다. **C를 A라는 단어로 바꿔서 프롬프트를 재생산**한다. **이 프롬프트를 다시 SD에 넣어서 이미지를 만든다.** 그럼 A 이미지들이 잘 생산된다.
3. 이 파트가 핵심. 
    
    F는 이미지 SD의 image extractor, D는 거리 함수, 엡실론은 poison으로 해석하면 됨 (perbutation이라는 용어가 계속 나오는데 이건 SD 모델에서 사용하는 개념인듯)
    

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/988bdfb8-d626-48e3-bb1c-5c3086f7c69c/d9a09fc4-8c61-48a1-887d-f415e03b4cca/image.png)

**위 수식으로 최적의 엡실론을 찾는다.** (xt는 C의 이미지, xa는 A의 이미지) 이 수식을 끝까지 돌려서 최적의 엡실론을 찾으면, 원본 이미지에 더해서 새로운 이미지 xt’를 얻는다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/988bdfb8-d626-48e3-bb1c-5c3086f7c69c/322a8b50-4603-4392-bade-22b901ca868f/image.png)

 

LPIPS라는 방법으로 최적해를 찾을 때는 위의 수식을 쓴다고 하는데 뭔지는 잘 모르겠음. 별도 논문 존재.

**Part 6. 실험결과**

이렇게 새롭게 나온 생성된 이미지 xt’를 원본 텍스트 C와 매칭시켜 새로운 데이터셋을 만든다. **이 xt’ 이미지는  사람이 눈으로 보거나, 모델이 분석했을 때는 아무런 문제가 없는 데이터** (C가 들어간 이미지) 이다. (텍스트는 프롬프트)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/988bdfb8-d626-48e3-bb1c-5c3086f7c69c/8ea78a55-b233-4226-878a-78c3e08e50db/image.png)

하지만 이 데이터셋으로 SD를 학습시키면 다음과 같은 결과가 나온다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/988bdfb8-d626-48e3-bb1c-5c3086f7c69c/495bf3ea-5326-4cad-9971-45ca613dd268/image.png)

그림이 완전히 망가지는 모습이다.

평가 방법 : CLIP이라는 image랑 text 매치해주는 모델 결과랑 Human Evaluation

실험 결과를 요약하자면 50장부터 유의미하게 공격이 성공하기 시작한다. 실험결과도 엄청 자세하기 분석해놨는데 너무 길어서 우선 보류. Posion Image인지 판별하는 모델들한테 잘 안걸러질 뿐더러 소수의 이미지만으로도 효과적이라는 내용임.

## 참고 링크
- [Exploiting Style Latent Flows 논문 PDF](https://arxiv.org/pdf/2403.06592)
- [Face Poison 논문 PDF](https://ieeexplore.ieee.org/document/10220056)
- [Landmark Breaker 논문 PDF](https://arxiv.org/abs/2102.00798)

