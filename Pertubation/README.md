다양한 Pertubation 기법 시간순으로 정리

1. FSGM
   https://arxiv.org/abs/1412.6572
   
2. PGD
   https://arxiv.org/abs/1706.06083
3. CW
   https://arxiv.org/abs/1608.04644
4. APAA
   https://arxiv.org/abs/2111.13841
5. ANDA
   https://arxiv.org/abs/2209.11964
6. CPO
   https://arxiv.org/abs/2312.06199
   전체 이미지 중 CNN이 가장 효과적으로 분석할 곳을 찾아냄. (아주 영향이 큰 부분은 살짝만 건드려도 효과가 크기 때문)
   CPO는 Quantization Matrix Optimization(양자화 행렬 최적화)이라는 기법을 사용함.
   이 Matrix는 주어진 이미지의 Frequency를 탐색하는 데에 사용됨.
   Frequency란, 인접한 픽셀과 RGB값이 얼마나 바뀌는지를 의미함.
   예를 들면 구름 없는 하늘 사진은 Frequency가 매우 낮음. 하지만 사람의 얼굴 같은 경우, Frequency가 높음.
   Quantization Matrix는 각각의 Frequency마다 얼마나 가중치를 주어야 하는지를 저장하는 행렬. (High-Frequency에 높은 가중치를 주게 역전파로 업데이트됨)
   이를 통해 더욱 효과적이고 효율적인 공격을 할 수 있게 됨.
   
