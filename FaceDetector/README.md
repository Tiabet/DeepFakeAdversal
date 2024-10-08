# FaceSwap Extraction 과정

FaceSwap은 얼굴을 인식하고 교체하는 과정을 다음과 같은 3단계로 나누어 진행합니다:

1. **Detection (탐지)**: 이미지 안에서 얼굴을 찾습니다.
2. **Alignment (정렬)**: 얼굴의 다양한 랜드마크(눈, 코, 입 등)를 찾고, 얼굴을 가운데로 위치시킵니다.
3. **Masking (마스킹)**: 얼굴을 제외한 나머지 영역을 삭제합니다.

즉, 공격은 Detection(탐지) 단계나 Alignment(정렬) 단계에서 이루어질 가능성이 큽니다.

![Alignment 과정 예시](https://github.com/user-attachments/assets/6cd7bdb5-0a51-45fd-95de-5a07f6dd6df5)

FaceSwap은 위 세 단계에서 각각 다른 세 개의 모델을 사용합니다: **Detector(탐지기)**, **Aligner(정렬기)**, **Masker(마스커)**.

![FaceSwap 모델](https://github.com/user-attachments/assets/78b46c82-b84c-4899-b366-af6850ba1dce)

여기서 'External'은 외부에서 처리된 정보를 가져오라는 의미입니다.  
**Detector** 단계에서는 **S3FD**가, **Aligner** 단계에서는 **FAN**이 최고 성능을 보인다고 합니다. 
하지만 paperswithcode 확인 결과 Face Detection에서 S3FD는 21위, FAN은 8위 수준으로 보입니다.

---

## 각 모델 설명

### Detector (탐지기)

1. **CV2-DNN**  
   OpenCV 라이브러리의 FaceDetector입니다.  
   - [CV2-DNN FaceDetector GitHub](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)  
   구조를 살펴보면 CNN이 여러 겹 쌓인 것을 확인할 수 있습니다. ResNet 기반으로 보입니다.

2. **MTCNN**  
   Multi-Task Cascaded CNN으로, **P-net**, **R-net**, **O-net**으로 이루어져 있습니다.  
   - [MTCNN GitHub](https://github.com/ipazc/mtcnn)  
   얼굴이 인식된 곳에 박스를 치고, 랜드마크까지 찾아줍니다.

3. **S3FD**  
   얼굴 크기와 관계없이 얼굴을 정확하게 인식할 수 있는 모델입니다.  
   - [S3FD GitHub](https://github.com/sfzhang15/SFD)  
   VGG16 기반의 CNN 모델로 한 번 feature extraction을 진행하고, multi-scale feature layers를 통과하여 다양한 크기의 얼굴을 탐지합니다. 다양한 크기의 얼굴 중 실제 얼굴을 최종 결정합니다.

---

### Aligner (정렬기)

1. **CN2-DNN**  
   Detector에서 사용되는 것과 동일한 구조입니다.

2. **FAN**  
   다양한 표정과 각도에서도 얼굴의 특징을 정확히 정렬할 수 있는 모델입니다. 현재 SOTA(State of the Art) 모델로, Hourglass Network를 사용합니다.  
   - [FAN GitHub](https://github.com/1adrianb/face-alignment)  
   Hourglass Network는 feature maps에 대해 downsampling과 upsampling을 동시에 수행하는 대칭적인 네트워크입니다. 각 Network에서 align 가능성을 계산하고, 이를 바탕으로 heatmap을 그립니다.
