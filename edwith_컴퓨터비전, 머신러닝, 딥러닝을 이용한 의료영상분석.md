# 컴퓨터비전, 머신러닝, 딥러닝을 이용한 의료영상분석

키워드: Medical Image Analysis, Computer Vision, Machine Learning, Deep Learning

교수자: DGIST 박상현 교수 (로봇공학 전공)

연구실: [MI&SP Lab](https://sites.google.com/view/mispl/home)

강의 링크: [edwith](https://www.edwith.org/medical-20200327)



## Week 1 (Chapter 1 & 2) :date:Aug 24-28, 2020

### Introduction to medical image analysis (1:38:17)

#### 1. Overview (7:25)

* 의료영상 분석을 위해 꼭 필요한 ==배경==(background)
* 병원에서 의료 영상들이 어떤식으로 ==저장==되고 의사들에게 ==전송==되어 원하는 영상을 ==확인==할 수 있는지
* X-ray, MRI, CT 등 병원에서 평소에 접하는 영상들의 ==취득 원리==와 ==특징==

Medical Image Analysis는 주로 3D 영상이며, Computer Vision (2D 등)과 Machine Learning 기술과 교집합을 공유한다. Artificial Intenligence의 한 분야로서의 Machine Learning

**강의 Contents**

* Classification: 정상 vs. 환자 영상 분류
* Segmentation: 특정 부위 추출, 이상 위치 감지
* Enhancement: 노이즈 제거, 해상도 향상
* Registration: 영상 비교 전 맞춰주는 방법 (크기, normalization 등)

#### 2. Introduction to medical image analysis 1 (9:21)

* Applications
  * Assist reading of scans, slides, ...
  * Prevent blindness
  * Classify cancer, identify, mutations
* Study topics
  * Image reconstruction methods
    * 2019 GE Healthcare: deep learning-based recon engine, FDA clearance
    * 2017 Philips: IntelliSite Pathology Solution, FDA clearance
  * Image labeling methods
  * Machine learning methods
  * Machine learning explanation methods
    * CheXNet: Chest X-ray Img to Pneumonia Possibility and Coloring

#### 3. Introduction to medical image analysis 2 (18:27)

* AI -> ML -> DL
  * Problem에 대한 문제 해결 방법을  Export에 의한 Rule-based 로 하다가,
  * Machine Learning 을 통해 Data로부터 Rule을 만들어내는 Framework
  * Classification, Segmentation, Enhancement, Registration 등에 Framework 적용 가능
* ImageNet (ML+ComputerVision)
  * DeepLearning 배경: Data의 양이 많아지고, GPU의 성능 향상으로 Parallel computing 가능
  * 2012년 DeepLearning Network 제안 후부터, Error율이 급격히 낮아짐 (Human error rate: 5% 내외)
  * Medical Image Analysis로 연관됨
* 강의 Contents
  * Classification
    * Conventional Methods: Logistic Regression, Neural Network, SVM, Random Forest
    * Deep Learning Methods: Deep neural network, CNN
  * Segmentation
    * Conventional: Thresholding, Region growing, Graph cut, Active contour model, Active shape model(learning-based)
    * DL: FCN, U-Net, DeepLab
  * Enhancement
    * Conventional: Normalization, Histogram equalization, Filtering, Dictionary learning
    * DL: SRCNN, GAN, SRGAN(super-resolution)
  * Registration
    * Conventional: Transformation matrix, ICP(Iterative closest point), Non-rigid ICP, Deformable models
    * DL: FlowNet, CNN for Registration

#### 4. PACS/DICOM/Visualization (13:52)

* PACS: Picture Archiving and Communication System
  * Acquisition Modality로부터 획득된 영상들을 DICOM image format으로 PACS에 저장
  * 의사 등 End-user가 자신의 PC를 통해 PACS server에 접근해 DICOM 영상 열람
* DICOM: Digital Imaging and COmmunications in Medicine
  * RSNA 북미방사선학회에서 정한 표준
  * 영상, 통신 표준 + 간단한 임상 정보(Clinical Information) 포함
* Medical Image Format => 영상 정합 및 Registration시 고려해야함
  * Dimensions (x, y, z): 축별 픽셀 개수
  * Voxel Spacing (x, y, z): 픽셀에 대응되는 물리적 점의 크기
  * Origin (x, y, z), ...
* Medical Image Visualization
  * itk-SNAP -> 3D annotation
  * ImageJ

#### 5. Image acquisition (12:12)

#### 6. X-ray/CT/PET (17:52)

#### 7. Magnetic Resonance Imaging (MRI) (19:08)

#### 8. Quiz 1



### Medical image classification (1)

#### 1. Introduction to medical image classification

#### 2. Linear Regression

#### 3. Logistic Regression

#### 4. Neural Network

#### 5. Image Classification

#### 6. Medical image classification

#### 7. Classification with demographic scores

#### 8. Quiz 2



## Week 2 (Chapter 3 & 4)

### Medical image classification (2)

### Medical image classification (3)



## Week 3 (Chapter 5 & 6)

### Medical image classification (4)

### Medical image segmentation (1)



## Week 4 (Chapter 7 & 8)

### Medical image segmentation (2)

### Medical image segmentation (3)



## Week 5 (Chapter 9 & 10)

### Medical image enhancement (1)

### Medical image enhancement (2)



## Week 6 (Chapter 11 & 12)

### Medical image enhancement (3)

### Medical image registration (1)



## Week 7 (Chapter 13 & 14)

### Medical image registration (2)

### Medical image registration (3)