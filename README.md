# DeepFakeAdversal

- 한국인 논문
Exploiting Style Latent Flows for Generalizing Deepfake Video Detection
https://arxiv.org/pdf/2403.06592

- 핵심 논문
Face_Poison_Obstructing_DeepFakes_by_Disrupting_Face_Detection
https://ieeexplore.ieee.org/document/10220056

- 관련 논문
Landmark Breaker Obstructing DeepFake By
https://arxiv.org/abs/2102.00798

**핵심 : 현재 모든 방법론들은 특정 모델로 Train Data를 망가트리는 것이 아닌 어떠한 Method를 사용하는 것임 -> 깃헙 코드가 없고, 방법론을 보고 구현해야함**

**Face_Poison_Obstructing_DeepFakes_by_Disrupting_Face_Detection**

큰 내용 정리
1. 우선 특정 모델을 타겟으로 하는 것이 맞음. 하지만 Deepfake 모델이 아닌 Face-Detector 모델을 공격함. (모든 DeepFake 모델은 Face-Detecting이 필수적임) 이 방법의 장점은 모든 딥페이크 모델에 강건하게 대응할 수 있다는 것. 그래서 Deepfake 모델들이 이런 데이터들로 훈련을 하면 결과가 멍청해지는 것임.
2. 특정 모델의 "레이어"를 타겟으로 공격 (Multi-scale feature-level adversarial attack)
3. 그렇게 생선된 이미지를 딥페이크 모델 훈련 과에 넣으면 육안으로 볼 땐 똑같지만 모델 생성 결과가 박살남. (얼굴을 학습시킬 때 Train Image에 어디가 얼굴이라고 라벨링이 되어있다는 점을 생각하면 쉬움)

작은 내용 정리
1. Multi-scale feature-level adversarial attack (MSFLA)
   1-1. 현재 DTCNN 등 모든 SOTA Face-Detector 들은 DNN 모델들임.
   1-2. 인코더는 ReLU, CNN, FCNN 등이 아주 여러개가 쌓인 우리가 일반적으로 사용하는 딥러닝 모델임.
   1-3. MSFLA는 이 중 특정 몇개의 레이어를 타겟으로 삼음.
   1-4. 오리지널이미지가 특정 레이어를 통과할때의 특징을 약간씩 Posion을 넣어서 이미지를 망가트리는 느낌.
   1-5. 이렇게 Face-Detecting의 결과가 나오면 당연히 이미지는 눈으로 볼때 멀쩡함. 하지만 Face-Detector들은 얼굴 인식을 못함. **이미지 자체를 공격하는 것이 아님!**


