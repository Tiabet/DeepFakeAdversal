FaceSwap의 Extraction 과정

1. Detection : 사진 안에서 얼굴을 찾음.
2. Alignment : 얼굴 안의 다양한 landmark를 찾고, 얼굴을 가운데로 위치시킴. 얼굴 안의 눈, 코, 입 등을 찾음.
3. 얼굴 빼고 나머지는 삭제시킴.

따라서 공격은 Detection과정에서나 Alignment 과정에서 일어날 수 있을 듯.

![image](https://github.com/user-attachments/assets/6cd7bdb5-0a51-45fd-95de-5a07f6dd6df5)
Alignment 예시

Faceswap은 3단계의 과정에서 각각 다른 3개의 모델을 사용하게 됨 (Detector, Aligner, Masker)

![image](https://github.com/user-attachments/assets/78b46c82-b84c-4899-b366-af6850ba1dce)

여기서 External은 만약 다른 모델로 처리한 정보가 있으면 갖고오라는 의미.
Detector에선 S3Fd가, Aligner에선 Fan이 최고 성능이라고 말하고 있음. (추가확인필요)

모델 설명

Detector

CV2-DNN
OPENCV 라이브러리의 FaceDetector
https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
구조를 확인해보면 CNN을 겹겹이 쌓은 것을 알 수 있음 (ResNet기반으로 보임)

MTCNN
P-net, R-net, O-net 으로 이루어진 Multi-Task Cascaded CNN
https://github.com/ipazc/mtcnn
얼굴로 확인되는 곳에 박스를 치고, landmark까지 찾아줌

S3FD
https://github.com/sfzhang15/SFD
얼굴 크기가 작든 크든 다 얼굴로 인식할 수 있는 모델
VGG16같은 CNN기반 모델로 한 번 feature extraction을 진행하고, multi-scale feature layers를 통과하면서 다양한 크기의 얼굴 탐지.
다양한 크기의 얼굴 중 진짜 얼굴이 어디인지 최종결정.

Alginer

CN2-DNN
Detector의 것과 동일

Fan
https://github.com/1adrianb/face-alignment
다양한 표정, 각도에서도 얼굴의 특징을 정확히 align할 수 있는 모델 (SOTA인 이유인듯)
Hourglass Network 사용 (feature maps에 대해 downsampling과 upsampling을 같이 진행하는 symmetric network)
매 Network마다 align의 가능성을 계산해서 heatmap을 그림


