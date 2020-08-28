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

* 취득 장비
  * Endoscopy, Microscopy
  * X-ray
  * Computed Tomography (CT), Positron Emission Tomography (PET), Magnetic Resonance Imaging (MRI)
  * Ultrasound 초음파, Optical Coherence Tomography (OCT)
* 원리
  * Light Source로부터 빛이 물체에 반사되어 CCD 또는 CMOS Sensor를 통해 촬영
  * Wavelength에 따라 Radio-, Micro-wave, Infrared/Visible/Ultraviolet, X-ray, gamma-ray
  * Wavelength가 짧을 수록 에너지가 크고 유해할 수 있음
* Visible Light 예시
  * Endoscopy (내시경) - 끝단에 LED Light와 Sensor를 부착하여 반사된 빛 촬영
  * Microscopy (현미경) - 조직검사 등에 사용

#### 6. X-ray/CT/PET (17:52)

* X-ray: Tissue(0), Air(+), Bone(-)에 따라 통과하는 X-ray의 양이 달라짐
* CT: X-ray를 여러 단면으로 찍은 후 Reconstruction을 통해 3D화, 단면이 많을 수록 더 clear하지만, 그만큼 방사선을 많이 맞아야 하므로 수를 줄이면서 clear하게 recon 하는게 연구
* PET
  * CT와 반대로 방사선 물질을 몸 안에 주입하고, 안에서부터 나오는 방사선(gamma-ray)을 detect
  * 360도로 positron(양전자)가 emission되므로 detection도 360도로 이루어져야 함
  * 방사선 물질을 포함한 포도당 대사(신진대사 - metabolic)는 주로 암세포에서 비상적으로 많이 일어나기 때문에, 해당 부위에서 밝게 나타남
  * 단, 물질로부터 나타나는 방사선을 촬영하므로, CT와 달리 Structure를 보기는 어려움 => PET-CT, PET-MRI 를 통시에 촬영하는 장비도 나오고 있음

#### 7. Magnetic Resonance Imaging (MRI) (19:08)

* 강한 자기장(1.5Tesla, 3T, 7T 등)을 걸면 수소원자핵은 방향을 정렬하여 세차 운동하게 됨
* 이때, RF(Radiofrequency) pulse를 쏘면 공명을 일으켜 z축에서 넘어진 후, RF Pulse를 끊으면 다시 일어서는 과정에서 에너지 신호를 발산하게 됨
* x, y축의 T1-weighted image와 T2-weighted image를 통해 Tissue 별로(Fat, Muscle, ligament, brain verves 등) 서로 다른 패턴이 나타남 이를 Fourier Transform 하여 구분함 => Soft Tissue를 구분하는데 유용함
* 방사선이 아니기 때문에 몸에 유해하지 않음, 하지만 소음이 있고 취득에 시간에 오래 걸림
* 비교
  * CT: 폐, 뼈를 이미징하기 쉽고, 출혈 등이 있을 때도 가능. 저렴하고 빠르다. 하지만, 방사선 노출, 혈관 조영제의 위험성
  * MRI: 방사선 없고, Soft Tissue를 구분해서 볼 수 있다. 하지만, 비싸고 소음, 느리다.
  * PET: 신진대사(Metabolic function)을 확인할 수 있으므로 조기 진단을 할 수 있다. 하지만, 방사선이 있을 주입해야하는 위험성, 비쌈

#### 8. Quiz 1

* 7/7 100점~



### Medical image classification (1)

#### 1. Introduction to medical image classification (14:00)

* Brain (MRI) image classification
  * White matter, Gray matter, 외곽은 CSF matter
  * 일반 영상과 인지 장애, 알츠하이머(AD)(GM이 작음)의 영상이 다르다. 하지만, age에 따라서도 알츠하이머처럼 GM이 줄어드는 영향도 있다.
  * 영상을 통해 병을 구분하는 것도 중요하지만, screening도 중요하다. screening에서는 false negative를 경계해야한다.
  * 병, 나이, 성별 등에 따라 달라질 수 있으므로 Demographic score를 함께 고려하여 classification 해야함
* Pathology image classification
  * 저해상도로는 세포와 핵 등의 변화를 관찰하기 어렵다. 고배율 영상으로 세포핵의 색깔이 불균일하는 등의 상태로 암세포인지를 구분할 수 있다.
  * 저배율 영상 대비 암세포의 크기는 매우 작기 때문에, 고배율로 모두 일일히(manual로) 다 보기는 힘들다. 따라서, 이런 부분을 인공지능을 통해 자동화하는 연구
* Challenges in Medical Image Classification
  * Limited data: 병원끼리 데이터 공유가 쉽지 않고, 환자 정보이기 때문에 개인정보 보호 때문에 어렵다. ADNI, TCGA(병리) 등의 사이트에서 데이터를 모아서 연구 진행
  * Large image size: 3D 등 데이터는 매우 큰데
  * Small changes: 큰 이미지 사이즈에 비해 이상이 있는 부분은 매우 작다
  * Demographic scores: 단순히 이미지만 아니라, 환자 정보와 상태를 고려해서 분류해야함
* Classification 강의 Contents
  * Conventional Methods: Logistic Regression, Neural Network, (SVM, Random Forest)
  * Deep Learning Methods: Deep neural network, CNN(VGG, ResNet, DensNet 등)

#### 2. Linear Regression (23:21)

* x가 있을 때 y(continuous, not discrete)를 찾는 문제: 데이터로부터 Regressor 라는 모델을 Training 해두면, 추후 임의의 Input x에 대해 알맞는 y를 출력함
* 표현 방법
  * w 등의 weight parameter를 사용한다.
  * 1차 함수, 2차 함수 등으로도 표현할 수 있다.
  * 여러 Input feature(x1, x2, ...)에 대해서도 정의할 수 있다.
* Cost function
  * 현재 상태의 regression model을 평가하는 방법으로서, 수식 표현
  * Cost를 minimize하는 형태로 Training을 수행한다.
  * wx+b 조합에서 cost가 최소가 되는 값을 찾기 위해, cost-w,b space를 그려볼 수 있다.
* Gradient Descent
  * Cost function을 w에 대해 미분하여 w를 계속 수정하는 방식으로, optimal을 찾아감. 미분을 한다는 것은 해당 그래프에서 기울기를 구한다는 것이고, 단면으로 봤을 때 기울기(gradient)를 통해 최저점으로 하강(descent)한다.
  * 각 b, w1, w2, w3 등에 대해서 GD를 모두 수행한다.
  * b는 w0로 표현할 수 있다. 일반화를 위해 w1x1과 같이 bx0을 wx0으로 하고, x0 = 1로 고정할 수 있다.
  * alpha는 기울기에에 따라 얼만큼씩 이동하느냐를 결정하는 "Learning Rate"
* Activation function
  * 결과로 나온 y에 대해서, Logistic function 등을 이용해 값에 변화를 줄 수 있다.

#### 3. Logistic Regression (15:03)

* y가 continuous한 무한의 값을 갖는 Linear Regression과 달리, Logistic Regression에서는 y가 0 또는 1의 값을 갖는다.
* 계단형의 Logtistic Function으로 fitting 해야한다. $g(z) = {1 \over {1+e^{-z}}}, z = wx+b > 0$
* Cost function
  * Logistic은 y값이 0 또는 1이기 때문에, convex가 아니어서 미분되지 않는다.
  * Cross Entropy를 사용해서 0과 1에 대해서 따로 계산이 되게끔 한다.
  * 결국 미분하면 linear regression과 같은 식이 나오게 되고, 이는 즉 gradient descent가 가능하다..
* Activation function을 통해 값을 출력하고 training 하면 model이 된다.

#### 4. Neural Network (15:12)

* Regression과 같지만, Layer가 2개 이상인 구조
* weight의 갯수가 크게 늘어남
* 최종 output에 가까운 Layer의 노드에 대해서 GD를 수행하고, 미분의 Chain rule에 의해서 앞 Layer의 weight에 대해서도 GD를 수행할 수 있게 된다.
* Input Layer로부터 weight를 통해 Ouput을 계산하는 과정이 Forward propagation, 반대로 미분을 통해 derivative를 구하고 weight를 업데이트 하는 과정이 Backward propagation
* Layer의 층을 많이 늘리면 Deep Neural Network: Logistic, NeuralNet보다 분류 성능이 더 좋다.
* Activation function
  * Sigmoid: 중간 Layer에서 sigmoid를 사용하면, 미분했을 때 0이 되는 경우가 많아서 back propagation이 잘 안되는 현상이 나옴
  * ReLU, Leaky ReLU: 많은 부분에 대해서 미분값을 잘 나오게 하기 위한 방법
  * tanh

#### 5. Image Classification (4:13)

* 각 픽셀의 값들이 x1, x2, x3... 로 연결됨
* Input Layer (x)
  * 100x100 영상의 경우 10,000개의 x가 들어가고, 거기에 bias term input도 x0로 추가함
  * 일반 RGB 영상의 경우 30,000개의 feature로 늘어나게 됨
* 이를 Logistic Regression with sigmoid를 통해 0 Dog 또는 1 Cat으로 매핑할 수 있음
* Neural Network/ DNN의 경우 Input Layer는 똑같이 해주면 되고, 대신 Paramter 수가 급증함

#### 6. Medical image classification (7:21)

* 3D 의료영상의 경우는 일반 영상보다 feature 수가 너무 많아짐
  * 100x100x100의 Voxel에 대해서 Grayscale이라 해도 Input Feature가 100만 개가 됨
  * Parameter는 이것보다 훨씬 더 크게 증가함
* Overfitting 문제
  * Training해야할 Parameter 수에 비해 데이터가 적은 경우에는 Overfitting 문제가 생김
  * 학습된 Model을 Training 데이터에 대해 돌리면 accuracy가 100% 나오지만, Test 데이터에 대해 돌리면 확 떨어진다.
  * 의료 영상의 외곽 영역 검은 부분은 Classification에 거의 영향을 주지 않는 의미없는 부분이므로, 이를 제거하여 Feature를 줄일 수 있다.
* Feature Extraction (Preprocessing for Brain)
  * FSL, Freesurfer 등의 SW를 이용해서, 의료영상으로부터 Feature를 추출할 수 있음
  * 외곽을 제거하고, 뇌 영역별로 (white, gray, 등) Segmentation을 수행한다.
  * 이를 통해 얻어진 Volume, Thickness, Intensity 등으로 얻어진 값들을 사용함으로써 400~500여 개로 Input Feature 확 줄일 수 있다.
* Feature Extraction (Preprocessing for Cells)
  * Q-pass 등을 통해 세포 수, 세포 크기, 모양 등을 추출하여 Input Feature로 사용하므로써, Pixel값을 그대로 사용하지 않으므로서 Parameter 수를 크게 줄일 수 있음

#### 7. Classification with demographic scores (15:50)

* Brain의 경우 Age, Gender, Feature1(volume 등), Featrue2, Feature3, ...
  * Normal subject에 대해서, "Age, Gender => Feature1, Feature2, Feature3"으로 Linear Regression
  * 모델 생성 후, 실제 y(feature)값과 비교하여 residual(error?)을 구해서 그 값으로 모델을 Training하는게 일반적. 하지만 요즘은 Feature Extraction이나 Linear Regression 보상없이, Image+Demographic Score를 DNN으로 바로 Classification 하기도 한다.
  * Residual로 Model에 보상하는 과정이 Feature Normalization
* Overall Procedure
  * Training
    Images 획득, Demographic Research => Feature Extraction => Feature, D.Score => Linear Regression => Residuals, Labels => Classifier (Logistic Regression, NN, DNN)
  * Testing
    Image => Feature Extraction => Feature, DS => Linear Regression => Residual => Classifier => Label
* DS나 Feature가 없이 Image의 Raw data를 그대로 사용하면 효율과 정확성이 낮(았)다.

#### 8. Quiz 2

* 7/7 100점~



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