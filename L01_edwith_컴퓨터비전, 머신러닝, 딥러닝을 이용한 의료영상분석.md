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

#### 1. Property of Deep Neural Network (20:15)

* DNN에서 Hidden Layer의 역할
  * Weight를 조정하여 특정 부분의 Local 특징을 강조하여 보는 노드를 만든다.
  * 각 노드는 특정 부분에 대한 일종의 '해석'을 하고 그 해석을 모아서 새로운 '해석' 노드를 만든다.
  * 즉, Low level feature를 모아서 High level feature를 만들고 그걸 바탕으로 classification 수행
* DNN의 Feature Extraction
  * Logistic/Linear Regression이나 Neural Net에서는 Feature Extraction을 별도로 수행하고 학습을 진행했지만, DNN에서는 Hidden Layers에 의해 Feature Extraction이 이뤄진다고 할 수 있다. 즉, Feature Extraction 과정이 통합되어 있다고 말할 수 있다.
  * 기존에는 Feature Extraction을 사람의 판단에 의해 직접해주는 경우가 많았다. DNN에서는 End-to-End로 (많은) 데이터와 라벨의 관계를 스스로 학습하며 Feature Extraction이 내부에서 이뤄진다.
  * 데이터가 적을 때는 기존 Convolution 기반 Classifier나 Deep Learning에서나 성능의 차이가 크지 않다. 왜냐하면 데이터로부터 표현할 수 있는 feature가 제한적이기 때문이다. 오히려 DNN의 성능이 좀 더 낮을 수 있다. 하지만, 데이터가 많아질 수록 DNN은 많은 Parameter를 통해 다양한 variation을 커버할 수 있고, 이를 통해 계속해서 성능이 올라간다.
* DNN의 한계점
  * Layer와 노드를 Deep 하게 쌓으면 쌓을 수록, Parameter의 갯수가 급증한다.

#### 2. Convolution (8:02)

* Image에 Filter를 적용해서 새로운 Image를 생성한다.
  * Average (blurring) Filter
  * Normal distributed filter
* CNN에서는 단순한 parameter가 아니라, 이런 Convolution Filter matrix(Feature Map)의 값을 학습해두고, 추후 이지미를 넣었을 때 학습된 Filter를 통해 Classification을 진행하겠다는 아이디어.

#### 3. Convolutional Neural Network (CNN) (19:49)

* Convolutional layer
  * 각 RGB 채널별로 Filter를 만들고 (3x3 Filter -> 3개 = 9 * 3 = 27개 Parameter 학습)
    6x6 영상의 경우 => 4x4로 output, 4x4 => 2x2 output, 모든 영상은 3 channel
  * Stride: 기본적으로는 1칸씩 이동하면서 수행하지만, Stride를 조정함으로써 Convolution 이동 크기를 바꾸어 output dimension을 바꿀 수 있음
  * Padding: convolution 수행시 size가 작아지는 현상 때문에 output size를 원본과 동일하게 유지하기 위해서, input 영상에 padding pixel을 붙여서 convolution 수행.
    * padding 값은 가장 가까운 픽셀의 값을 넣어줄 수도 있고, 0으로 넣어 줄 수도 있고, 영상을 앞뒤로 붙이기도 하고
  * Hyper-parameter: Filter의 dimension, Stride, Padding 등 parameter를 조정하기 위한 parameter
* Pooling layer
  * 영상의 Size를 줄이기 위한 Layer. 이미지를 Grid로 나누어서, 해당 Grid Cell에서 Max, Average 등을 수행. 보통 Max Pooling 사용
  * Local 특징을 유지하는데 도움이 되도록 하기 위한 개념
* Fully connected layer
  * Convolution과 Pooling을 통해 충분히 Layer가 작아지면, FCL 사용
  * 하나의 노드에 각 채널의 모든 pixel + 1 bias parameter 를 연결하여, 값을 종합하는 과정
  * Fully Connected Node를 여러개 두고, 그 노드를 종합하여 Neural Net 구조로 최종값 도출
* Convolution, Pooling을 통해 Image 사이즈를 줄여가며 Filter를 사용하고, FCL을 통해 Classification을 한다. 결과와 비교하여 error을 확인한 뒤, back propagation을 통해 Filter를 업데이트 한다. 업데이트 된 Filter를 통해 다시 classification을 하고, back propagation 반복.

#### 4. Advanced CNNs (LeNet, AlexNet, VGG) (10:30)

* LeNet-5: Padding을 안 썼기 때문에 Feature Map의 size가 줄어들기 때문에 channel수 (filter 수)를 늘려주는 방법을 사용함
  * Input: 32x32x1
  * 5x5 Filter with No padding => 28x28x  6개 filter
  * 2x2 Average Pooling => 14x14x6
  * 5x5 Filter => 10x10x  16개 filter
  * 2x2 Average Pooling => 5x5x16
  * 120 FC
  * 84 FC => Softmax output
* AlexNet
  * Input: 227x227x3 (Color 영상)
  * 11x11x3 Filter with 4 stride => 55x55x96: input 영상이 크기 때문에 stride를 사용해서 feature map size 줄임
  * 5x5x3 filter(Convolution layer)에서 Padding 사용하여 feature map 크기 유지
  * Max Pooling 사용
  * 각 Layer의 activation function으로 ReLU 사용하여 성능 향상
* VGG-16 (Layer 갯수가 16개)
  * Input: 224x224x3
  * 일관적이게 3x3(x3) convolution layer 사용 (with padding)
  * Max Pooling 사용
  * ReLU 사용
  * 3x3 convolution을 2번 연속 사용하면, 5x5를 사용한 것과 같은 효과
    * 하지만, 5x5x3 = 75개 parameter임에 비해서, 3x3x3 x2 = 54개로 param 수가 더 작다.
  * 하지만, FC Layer에서 Parameter수가 많기 때문에 overfitting의 문제가 있을 수 있다.

#### 5. Advanced CNNS (ResNet, InceptionNet, DenseNet) (24:00)

* 기본적인 구조: (Convolution + Pooling) x N + FC
* ResNet: Residual Network (Layer 갯수가 150개가 넘어도 성능 유지)
  * Residual Block: Residual 값을 다음 레이어에 연결해주는 Skip Connection을 통해 성능 향상
  * Layer가 많아지면 일반적으로 에러가 줄어들지만, 너무 많아지면 다시 에러가 증가한다. 하지만, Residual Block을 통해 Training에서 Layer가 증가해도 Error가 계속 줄어들 수 있도록 해준다.
* InceptionNet <- GoogleNet
  * Inception Module: 1x1 Convolution을 활용해서 다양한 크기/종류의 Filter를 Concat하여 사용한다.
  * 1x1 Convolution: Feature 수를 줄일 수 있고, ReLU (activation) function을 통해 non-linear decision boundary가 생기게 된다. Feature를 줄이면서도 기존에 있던 Feature의 정보를 조합하기 때문에 정보 손실이 크지 않다.
* DenseNet
  * Convolution 수행 후, Input된 이미지 channel을 가져와 concat 시킨다. (channel depth가 이전의 2배) concat을 계속해서 해서 convolution을 하면 gradient decent가 더 잘 일어난다.
  * Skip Connection을 Dense하게 사용한다는 것이 요지
* Performance Changes
  * 2012 AlexNet
  * 2014 VGG, GoogleNet
  * 2015 ResNet 등장 < human error 5%
  * 2017 DenseNet

#### 6. 3D CNN with demographic scores (11:39)

* 3차원 Convolution
  * 3D volume(image)에 3D Filter를 사용함
  * (x,y,z,c) 6x6x6x1 => 의료영상은 대체로 1channel 이미지, 혹시 다른 측정값을 사용해 여러 채널로 만든다 해도, 방법적인 면에서는 크게 다르지 않음
* Demographic Score 이용
  * Convolution이 끝나고, FC 할 때 Age, Gender 등의 값을 추가 Feature로 이용해서 최종 Decision에 이용

#### 7. Quiz 3

* 5/7 



### Medical image classification (3)

#### 1. Overall procedure (8:34)

* 강의 Contents
  * 의료영상 분류 모델을 학습하는 과정
  * 모델 학습시 모델을 평가하고 성능을 높이는 방법
  * Validation, overfitting을 줄이기 위한 Regularization, Data Augmentation
  * 분류 결과를 얻고나서, 이 결과의 성능을 평가하는 방법
* Classification Procedure
  * 의료 영상 같은 경우에는 영상의 Type이 다르기 때문에 Preprocessing이 추가되는 경우가 많음
    * Voxel spacing이 다른 경우
    * 영상이 정렬되지 않아 registration
    * Intensity의 distribution이 다른 경우 normalization
    * Noise가 많은 경우 de-noising
  * 영상의 크기가 너무 큰 경우에, overfitting 문제가 생길 수 있기 때문에 Feature Extraction을 통해 추출하고 Classifier를 학습하게 됨
  * DNN은 End-to-End로 할 수 있지만, Data가 많아야 하므로, 적을 경우엔 Feature Extraction 등을 통해 LR, NN, SVM, RF(random forest) 등을 통해 Prediction 하는 경우가 많음
    * Feature Extraction을 하게 되면, 이 Feature를 DS(Demographic Score)를 이용해 Normalization 수행하고, 많을 경우엔 Feature Selection도 수행
    * Feature Extraction 방법으로는 Image Intensity, Texture (Haar feature), Segmentation에 따른 shape나 intensity 등 정보

#### 2. Validation (13:10)

* Hidden Layer, Node 를 몇개로 할 것인지 등을 정하기 위해 검증해야 함
* Train, Validation, Test Sets
  * Label을 가지고 있는 Training 데이터로 Model Training 수행 후, Test 데이터로 확인
  * Model을 Training 데이터에 최적화 되기 때문에 Test 데이터에서 잘 동작할지 모름
  * Training 데이터를 Validation set으로 나눠서, Training 데이터로 학습하고 Validation set으로 검증한다.
    * 학습 초기 training error와 validation error가 함께 감소하다가, validation error가 증가하는 시점이 생기는데, 이때부터 overfitting이 발생한다고 봄
  * Cross Validation (데이터가 적은 경우)
    * Training Data를 k개의 set으로 나눠서 k-fold cross validation
    * Validation set을 번갈아 가며 Training 수행

#### 3. Overfitting / Regularization (18:05)
* Overfitting
  * Linear Regression에서, 1차, 2차 등 n차 함수 모델을 만들 수 있음. 단, Training에서 학습한 모델에 Test data를 넣었을 때, 올바른 값을 예측할 수 있어야 함.
  * Overfitting이 되면 Training에서는 거의 에러가 0에 가깝지만, Test에서는 매우 커지는 현상이 발생함
  * Training에러도 큰 경우에는 Underfitting이라고 함
* Regularization
  * Linear regression에서 정의한 Cost function을 통해 Minimization이 이뤄져야 함
  * Weight의 L2-norm(제곱의 합) 등을 cost function에 포함해주고 함께 minimization하게 되면, weight가 무작정 커질 수는 없으므로 decision plane이 조금 단순/smooth 해진다.
  * 비슷하게 Logistic regression이나 NN에서도 가능함
* 순서
  * 일단 단순한 모델부터 시작해서 점점 복잡하게 만들어 가면서, Training 데이터에서의 underfitting(High bias)을 해소
  * 그후 Test 데이터를 확인하면서, Overfitting(high variance)에 의한 Test 데이터의 에러를 줄이기 위해 Regularization이나 data의 양을 늘려가면서 다시 overfitting을 해소
  * 반복 => Bias와 Variance가 모두 낮은 모델을 찾으면 Classifier로 이용

#### 4. Transfer Learning (4:44)

* DNN에서 데이터가 적을 때 보완할 수 있는 방법
* 의료영상이나 Normal image나 Low level feature(엣지, 직선, 텍스쳐 등)는 비슷하게 나타날 것이라는 가정(기대)아래, normal image로 먼저 학습을 한 후, 의료영상으로 transfer 함
* Normal image로 학습 완료 후 Low, Middle level feature까지는 fix시킨 후, 의료영상 데이터로 High level feature만 학습시킴. 데이터의 양에 따라, 어느 level까지 고정시킬지 판단

#### 5. Data Augmentation (2:48)

* 학습 데이터 양이 적을 때 보강해주기 위한 방법 - Training set, Validation set, Test set 중, Training/Val에만 augmentation 적용
* Mirroring: 좌우, 상하 반전
* Rotation
* Shearing: 약간 기울이기
* Local Warping: 영상의 일부분만 변화를 주도록 함
* Intensity change: 밝게 또는 어둡게 바꾸어서 학습

#### 6. Evaluation of classification model (20:46)

* Confusion Matrix
  * True Negative(TN), True Positive(TP) / False Negative(FN), False Positive(FP)
  * Sensitivity: True positive rate, recall = TP/(TP+FN)
  * Positive Predictive value: PPV, Precision = TP/(TP+FP)
  * Specificity: True negative rate = TN/(TN+FP)
  * Accuracy: (TP+TN) / (TP+TN+FP+FN)
* Screening: Normal->Cancer(FP)는 괜찮지만, Cancer->Normal(FN) 은 안됨. Recall이 중요함
* ROC Curve(Receiver Operating Characteristics Curve): x=(1-TNR), y=(TPR)
* AUC(Area Under a ROC Curve)
* F1 score: Harmonic mean of recall and precision (조화평균) Positive와 Negative에 대해 동시에 평가하기 위한 값. 어느 한쪽이 많이 낮으면 낮은 쪽으로 평균지점이 내려감

#### 7. Evaluation of classification model (Multi-label) (9:44)

* Mutli-label의 Confusion Matrix
  * 의료분석에서는 상태의 Grading 문제 등이 Multi-label에 해당함
  * 각 Label을 기준으로, A이냐 아니냐로 구분하여 A에 대한 precision, recall을 구함
  * 각 Label의 precision, recall을 평균하여 mean precision, mean recall을 통해 F1 score를 구함
* Inbalanced data
  * 한 class의 데이터가 유난히 많아서 inbalanced 한 경우에는, Accuracy값이 왜곡될 수 있다.
  * 특히 이럴 경우에 F1 score가 도움이 된다.

#### 8. Quiz 4

* 7/7 100점~



## Week 3 (Chapter 5 & 6)

### Medical image classification (4)

#### 1. Feature selection using L1 regularization (18:17)

* 결과가 왜 나왔는지 알려줄 수 있어야 함
  * 여러 특징들 중 분류를 하는데 중요한 역할을 하는 특징들을 선택
  * 결정에 중요한 역할을 한 부위를 추출해서 보여줄 수 있는 기법
* Regularization
  * L2-norm이 아닌 L1-norm을 사용하면 일부 weight를 0으로 만들어 준다.
  * sgn(w) = w>0 일 때 1, w<0일 때 -1, w=0일 때 0을 이용해서, 미분했을 때 0에 가까워지도록 유도함
  * L0-norm도 있지만 수식으로 표현되지 않기 때문에 사용할 수 없다.
* Feature Selection
  * Regularization term이 추가된 Cost function을 이용해 Back propagation한다.
  * 특정 feature의 weight가 0이되고 나면, 남은 feature들이 선택 됐다고 볼 수 있음

#### 2. Feature selection using Entropy / Mutual information (18:25)

* Entropy
  * 연관성이 높은 어떤 두 벡터를 찾을 수 있으면, Label 분류시 유용한 Feature를 찾을 수 있다는 말이다.
  * Amount of information = -log{p(x)}: 항상 일어나는 일은 뻔하고 정보가 적다. 확률이 낮은 희귀한 일일 수록 정보량이 높다.
  * Entropy = H(x) = -sigma_x{p(x)*log(p(x))}: 정보량의 기댓값 = 각 정보x가 발생할 확률을 곱한 값
  * Joint entropy = H(X, Y): 두 사건이 같이 일어날 확률. 서로 독립 사건일 경우 H(X, Y) = H(X) + H(Y)이지만, 완전히 독립이 아니면 조건부확률을 사용해 계산됨
  * Mutual information: X, Y를 따로 보았을 때보다 같이 보았을 때 얼마나 불확실성이 감소하는가, 즉 의존도가 높고 관련성이 높은가를 나타내는 지표
* Decision Tree
  * 기존 set을 두 그룹으로 나눴을 때, Entropy(불확실성)이 가장 낮아지도록 하는 feature 기준을 선택하는 방법
* mRMR Feature Selection
  * Minimum-redundancy-maximum-relevance
  * D: Class와 비슷한 Feature가 선택 됐다면, R: 그 Feature와 비슷하지 않은 Feature를 선택하도록 함

#### 3. Feature extraction using Deep Learning (14:08)

* Feature Extraction
  * Feature Selection은 Feature 목록에서 중요한 것을 선택하는 것이지만, Feature Extraction은 Feature로부터 조합하여 새로운 중요한 Feature를 만들어 내는 것
  * Filter 등을 이용해서 Shape이나 Haar feature를 추출할 수 있다.
  * Convolution을 통해 새로운 Filter feature를 추출해 feature 수를 줄일 수 있음
  * Dimensionality Reduction: 특별히 Label이 정해져 있는 건 아니고 Unsupervised 방식으로 영상의 특징점을 구분하기 위한 방법 (e.g., PCA, AutoEncoder, ...)
* Auto-Encoder
  * Input과 OutputLayer를 같은 값으로 하고, Input보다 더 적은 수의 node를 가지는 Hidden Layer를 학습시키면, Input의 값을 함축한 Feature를 추출하는 것과 같다.
  * Stacked auto-encoder 구조를 통해, 한 번에 너무 작은 node로 줄여서 발생하는 정보 손실을 방지할 수 있다.
  * Noise를 추가해서 학습시킴으로써 노이즈에 robust하게 만들 수도 있다.

#### 4. Class Activation Map (12:27)

* Deep Learning이 좋은 성능을 내지만, Black box 같은 특성으로 왜 그런 결과가 나왔는지 설명해주진 못한다.
  * 최종 Feature Map에서 마지막에 Fully Connected Layer를 거치면서, 공간적인 정보가 사라지기 때문
  * 의료에서는 왜 그런 결과가 나왔는지를 아는 것이 중요하다. 이를 위한 기법으로 Class Activation Map
* Global Average Pooling: 각 Feature Map의 전체 픽셀 값을 더해서 평균한 값을 Fully Connected Node에 weight를 곱해서 최종 판단을 하게 된다. 이때, 이 Weight가 사실상 Feature Map의 각 픽셀에 곱해지는 것과 같으므로, Feature 이미지에 weight를 곱해 더해줌으로써 최종 판단 전 이미지에서 큰 weight를 갖는 부분을 확인할 수 있다.

#### 5. Weakly supervised learning (03:41)

* 영상 단위의 Class만 가지고 있고, 이상 부분은 Rounding Box나 Segmentation 등으로 표현할 수 있다.
* Image Label 으로 Training 후, Image Label 추론하는 건 Supervised learning이지만, Bounding Box나 Pixel Label을 추론하는건 Weakly supervised learning으로 조금 더 어려운 문제에 속한다.
* 마찬가지로, Bounding Box로 Training 한 후, Bounding Box를 추론하는 건 supervised learning이지만, Pixel Label을 추론하는 건 Weakly supervised learning이다.
* 의료영상의 Labeling은 시간이 오래걸리고, labeling을 할 수 있는 전문가의 수가 적기 때문에 이런 Weakly supervised learning 기법을 사용할 수 있다.

#### 6. Multiple instance learning (9:56)

* 특정 데이터가 아닌, 여러 데이터의 set이 Positive거나 Negative임을 알 때
  Positive training sample, Negative training sample
* Pathology 영상의 경우 전체 영상 크기에 비해 병변의 크기는 매우 작다.
  * 따라서, 작은 패치로 나눠서 작은 패치를 분류할 수 있는 Classifier를 만들고, 그 결과로 전체 영상의 분류를 하겠다는게 Multiple instance learning의 아이디어
  * 각 패치가 Instance이다.
  * Instance Classifier가 잘 작동한다면, 영상에서 어떤 위치에 병변이 있는지도 알 수 있기 때문에 Weakly supervised learning의 한 예시가 된다.

#### 7. Quiz 5

* 7/7 100점~



### Medical image segmentation (1)

#### 1. Introduction to medical image segmentation (6:24)

* Segmentation: 영상을 분할해주는 기법
  * 시간차를 두고 찍은 CT (longitudinal study)를 비교할 때 부위별 크기 비교
  * 구별하고 진단하는데 근거가 됨
* Conventional methods
  * 밝기값 기준: Thresholding, Region growing
  * 영상 정보+Prior 정보: Graph cut, Active contour model
  * Learning 기반: Active shape model
* Deep Learning methods
  * FCN
  * U-Net
  * DeepLab

#### 2. Otsu thresholding (9:11)

* (Threshold: 문턱값) 기준으로 크거나 작은 위치를 추출하는 방법
  * Voxel의 값으로 1 or 0으로 Labeling
  * Threshold를 몇으로 정하느냐가 중요한데, 보통은 Manual로 정하게 됨
  * Otsu Thresholding은 Threshold를 자동으로 지정해주는 알고리즘
* Otsu Thresholding
  * Weight를 주고, Variance를 기준으로 Background(0)와 Foreground(1)를 구분함
  * Within-class Variance = V_w = W_b * V_b + W_f * V_f
  * Between-class Variance = V_b = W_b * W_f * (E(b) - E(f))^2 가 최대가 되는 지점을 구함
    * ==수식 유도 복습 필요==

#### 3. Morphological processing (11:28)

* Thresholding 등으로 생긴 노이즈를 보정할 수 있음
* Convolution과 비슷한 Morphological processing
  * Dilation(팽창): 1, 0으로 둔 Matrix(Structural element)에서 foreground와 겹치는 부분이 있다면 모든 1 위치를 다 foreground로 채워주는 것
  * Erosion(침식): Dilation과 마찬가지이나, background와 겹치는 부분이 있다면 모든 1 위치를 다 background로 채워주는 것
  * Opening: Erosion->Dilation = 원본의 shape은 유지하면서 background의 노이즈를 제거하는 과정
  * Closing: Dilation->Erosion = 원본의 shape를 유지하면서 foreground의 hole을 채우는 과정
  * Opening->Closing(E->D->D->E)을 해서 영상을 다듬을 수 있음

#### 4. Region growing / Watershed algorithm (8:52)

* Region growing
  * 사용자 입력을 받아 영역을 생성하는 방법
  * Mask이미지에서 선택된 점을 기준으로 주위의 255 지점을 포함시킴, 또 다시 새로 추가된 점으로부터 주위의 255 지점을 포함시키는 과정을 반복
  * Color 영상 등에서 growing을 할 때는 정확히 같은 값이 없으므로, 기준값과 intensity 차이가 일정 threshold 이하일 경우 growing에 포함하는 방법 사용
* Watershed algorithm
  * Line profile grpah에서 굴곡에 물을 채우듯이, 봉우리를 기준으로 서로 다른 영역을 설정하는 방법
  * Marker를 설정해서, 영역을 분리할지 합칠지 결정할 수 있음

#### 5. Segmentation using graph model (17:56)

* Observation = 영상의 컬러값 (z1, z2, ..., z9)
* Label = 관찰된 값(Observation)으로부터 구분한 값 (x1~x9)
* Graph Model: Observation (z1~9)로부터 x1~9의 값을 구하는 것
  * Posterior probability: P(x1, x2, .., x9 | z1, z2, ..., z9) = P(z1, z2, ..., z9 | x1, ..., x9) * P(x1, ..., x9) / P(z1, ..., z9) by Bayes rule
  * Posterior ~ Likelihood prob * Prior prob
  * with Naive assumption: 각 샘플(x1, .., x9)는 독립적이다 => P(z1 | x1) * P(z2 | x2) * ...
    * Likelihood prob = 유사도만 보게 되면, 각 pixel의 값을 보는 거게 때문에 Region growing과 크게 달라지지 않음
    * Prior prob를 지정해줌으로써, 다양한 조합에 대해서 확률이 계산될 수 있도록 함 => 확률 최대화
    * 모든 Pixel을 고려할 수는 없기 때문에, Markov Random field를 통해 Label끼리의 연결성을 고려해서 주위의 확률에 의존해 계산하도록 함

#### 6. Graph cut optimization (17:00)

* 노드의 숫자가 많아지게 되면, 예를 들어 100x100 영상의 경우 1만개의 확률이 곱해지면 사실상 0에 가까워지게 됨 => 로그를 이용해서 -log 값의 합을 취하고, Markov Random Field에 의해 주위의 연결된 노드들에 대해서만 계산함
* E(nergy) = -log를 취해 부호가 반대가 되기 때문에, minimization하면 됨. Likelihood와 prior term의 값이 모두 줄어들게 만든는 x(label)값을 찾아야 함
* 3x3 kernel의 경우 2^9 = 512개의 경우의 수 중에서, E가 최소가 되는 조합을 구하는 방법
* Optimization
  * Max flow / Min-cut
  * Sampling

#### 7. Quiz 6

* 5/7



## Week 4 (Chapter 7 & 8)

### Medical image segmentation (2)

#### 1. Active Contour Model (15:03)

* 관심 영역에 바운더리를 그려주고 iteration을 진행하면서 경계를 정교화하는 과정으로 진행
* External Energy
  * E_edge = -|delta(Intensity)|^2 = 경계가 뚜렷한 곳, Gradient가 큰 곳, 즉 값의 차이가 큰 곳에 바운더리가 생길 것이다
  * E_line = Intensity itself => 배경이 흰색(255)이고 선이 검정(0)일 때, 혹은 반대인 경우엔 -Intensity로 정의하면 된다
* Internal Energy
  * Shape의 특성을 식에 반영하는 Term
  * E_elastic = Shape의 vertex 관계를 1차 미분하여, 전체적인 복잡한 형태를 smooth하게 만들어주는 term
  * E_bending = Shape의 vertex를 2차 미분하여, 뾰족한 형태를 뭉툭하게 만들어주는 term
* 전체 Energy_snake
  * External Energy + Internal Energy (+ User input energy)
  * Energy_snake를 최소화하는 v(s) => 즉 점들의 형태를 구하는 것
  * 사전에 알고 있는 형태 정보를 이용해서 구한다는 점에서 prior를 이용하는 것과 비슷함

#### 2. Atlas based method / Label fusion (12:53)

* 사용자 Input이 없이 자동으로 Segmentation하는 기법 => 기존에 생성해둔 영상과 Mask label을 이용(학습시켜서 모델을 만드는 것은 아님)
* Segmentation Map = Atlas
* 원본 이미지(Y1) -> Segmentation Map(X)로 바꿔주는(registration 시켜주는) Transformation Matrix를 찾는다.
* N개의 영상에 대해서, Transformation Matrix를 구한 후, 새로 들어오는 영상에 대해서 각각의 Transformation Matrix를 적용해서, 어떤 픽셀에 대해 Majority Voting을 하거나 새 영상과 기존 영상의 Similarity를 구한 후, 높은 Similarity의 T.M에 대해 weight를 더 높여 주는 식으로 적용.
* Non-rigid registration: Transformation Matrix 하나로 정합하는 것이 아니라, 더 복잡한 방식 사용
* Patch-based Label fusion: Non-rigid가 아닌 Affine registration을 수행하되, 전체 영상이 아니라, 윈도우를 잡아서 이동하면서 Simillarity가 높은 registrator를 사용함. 

#### 3. Segmentation via laearning based method (3:28)

* Image, Label pair를 모아놓은 Training Data로 학습을 함
* Segmentation Model의 방식들
  * Input Image로부터 Label을 직접 도출
  * 각 픽셀에 대해 prediction을 하고, 이를 모아서 Label 도출
  * Shape을 업데이트하고, 모델을 통해 더 나은 Shape로 업데이트하고, 이를 통해 Label 도출

#### 4. Principle Component Analysis (PCA) (14:42)

* Active shape model의 배경지식 : PCA
* Dimensionality Reduction
  * 어떤 축으로 데이터를 투영(projection)했을 때, 분포의 분산이 최대가 되게끔 해준다.
  * Lagrange Multiplier Method를 사용해서 편미분하여, 고유벡터와 고유값 구하기
  * Source 차원 갯수에 맞춰서 pair가 나옴. 그 중에서 축소하고자 하는 갯수만큼 pair를 선택 (큰 값부터)
* PCA Implementation
  * N개의 Training set에 대해서, Preprocessing 수행 (feature scaling, mena normalization 등)
  * Covariance matrix 계산
  * Singular value decomposition을 통해 Convariance Matrix에 대한 eigenvectors = U를 계산
  * 선택한 차원의 Eigenvector를 이용해서, Projection

#### 5. Active shape model (22:41)

* Training 데이터에 대해서, 각 점이 이동할 수 있는 범위를 학습해 놓겠다는 모델
* Active contour model로 가장 경계일 것 같은 곳으로 점을 이동시킴 => 잘못된 경계인 Local minimum으로 점을 이동시키면, 모양이 이상해질 수 있음 => 학습된 Active shape 범위 내에서 점이 이동하도록 범위를 제한해줄 수 있음
* Model construction
  * Data normalization = Shape alignment
  * 한 영상에 대해서, PCA 데이터는 2차원 (x, y)인 점이 10개일 때, 20차원의 데이터로 표현(representation)
  * 각 차원의 평균 => Mean shape(statistical shape model)로 표현
* Active Shape Model 테스트
  * Mean shape을 테스트하고자 하는 영상에서 위치를 조정해서 옮김
  * 학습 모델을 바탕으로 b를 구하고, 그 b를 가지고 다시 복원시킴으로써 모양을 유지한 채 Shape 변화를 시킴. Active contour model에서는 Energy를 이용해 모양을 유지하고자 했지만, Active shape model에서는 eigenvalue를 가지고 모양을 유지시킴
  * 학습시키기 위해 매칭점을 잡아줘야 한다는게 어려운 작업이지만, 학습을 하고 나면 굉장히 잘 동작한다.

#### 6. Segmentation using classifier (13:35)

* 단순히 Scribble에 의존해서 만든 모델보다, Classification의 트레이닝 데이터(Fore/Background 전체가 분류되어 있는 데이터)를 가지고 모델을 만들면, 좀더 정교하게 Segmentation 할 수 있다.
* Haar-like feature: 아래-위의 차를 구하는 Filter, 좌-우의 차를 구라는 필터 등등 여러 형태의 필터를 사용해, (Convolution 하듯이) 얻어진 Response를 모아서 Feature로 이용함
* Feature selection을 통해 선택하고, Logitstic regression 등을 이용해 각 픽셀에 대해 Fore/Background를 분류해서 얻을 수 있음
* Markov Random Field를 통해 조금 더 Smooth하게 결과를 만들 수 있음

#### 7. Quiz 7

* 7/7 100점~



### Medical image segmentation (3)

#### 1. Fully Convolution Network (FCN) (18:09)

* Deep Learning 기반 Segmentation 
  * 줄어든 Feature들로부터 다시 원본영상 크기의 Segmentation 결과를 Prediction 해야함
  
* Fully Convolution Network(FCN)
  * 기존의 CNN을 이용한 Segmentation은 Patch 단위로 Classifier 처럼 동작하여, 영상의 모든 Patch에 대해서 Classifier처럼 판단하고 이를 종합하는 과정을 거쳐야 했음 => Patch에 대해서 반복 수행하지 않고, 한번에 End-to-end로 수행하기 위해서 FCN 사용
  * FCN은 CNN의 마지막 부분의 Fully connected Layer를 사용하지 않고 Fully Convolution Layer로 대체
    * Dense(FConnected) Layer에 해당하는 자리에서, 1x1 convolution layer를 활용하되 채널 수를 조절해가며 수행, 최종 output의 크기와 맞게끔 조절하며 Layer를 concat한다.
    * 최종 Output layer의 size가 N이라면, 각 pixel의 prediction 값은 원본 영상의 1/N 크기의 Patch에 해당
    * Stride를 조절하면 Patch간의 거리도 조절 가능
    * Pixel 단위로 예측을 하려면 Upsampling(deconvolution)을 해야하는데, Skip connection을 이용해서, 이전 단계의 영상과 sum하는 방식으로 디테일을 높인 Segmentation 가능
  * 마지막 Output이 Input의 각 영역을 나타냄. Convolution을 한번만 수행하여도 각 Input 영역의 Prediction을 바로 확인가능함.
  * upsampling을 통해 Prediction의 결과인 Segmentaion map을 만듦. 해당 맵을 Labeling 된 데이터와 비교하여 Back propagation을 수행.
  
#### 2. U-net (17:10)
* 기존의 Upsampling 영상을 크게 만들어줌
   * Nearest neighborhood : 가장 가까운 점의 값을 가져옴.
   * Linear interporation : 양쪽 점의 값을 이용함.  (-> 2차원에서는 Bi-linear interpolation)
   * Unpooling
  * Deconvolution과 비슷한 개념
     * Pooling이 수행될 때, 해당 값이 획득된 위치를 저장해놓고, Unpooling할 때 해당 위치로 값을 가져옴
   
* Transposed convolution
   * 학습을 통해 Upsampling
   * Input matrix를 Vector로 풀면서, Kernel도 변형
   * Pooling 결과에 대해, 변형된 kernel의 Transpose을 곱함으로써, 원본 영상 형태로 복원하는 방법
   * 기존의 Convolution 연산을 거치면 Input 보다 Output feature map 이 작아짐. ex) 4x4 -> 2x2
   * Transposed Convolution 연산을 거치면 Input 보다 Output feature map이 커짐. ex) 2x2 -> 4x4
   * C^T x (4x1) = C^T x C X (16 x 1) 
* U-Net U-Net(MICCAI 2015)
  * Up convolution 을 사용한 방식
  * Convolution 연산을 통해 작아진 Feature map으로 부터 Trasposed convolution을 사용해서 사이즈를 늘려줌. 
  * 늘려줄때 이전의 Feature map을 concat하여 upsample을 수행. 
  * Transposed Convolution과 FCN에서처럼 Skip connection을 이용해 점진적으로 Up-sampling하는 네트워크

#### 3. Dilated Convolution (10:03)

* Filter의 receptive field의 사이사이에 공백을 넣어서 확장된 형태로 convolution
* 일반 Convolution을 해서 prediction할 때, upsampling 과정에서 영역이 뭉개지는 형태가 나타나는데, Dilated(atrous) ocnvolution을 하게 되면 조금 더 clear한 prediction을 얻을 수 있다.
* Downsampling, Upsampling 없이 바로 prediction을 하게 되면서 좀더 선명해지지만, 파라미터 수 변화는 없음
* Atrous Spatial Pyramid Pooling: Rate를 변화시켜가며 convolution한 후 모아서 Sum-fusion

#### 4. DeepLab V3+ (10:06)

* Depth-wise Separable Convolution
  * 3x3x3 => 1x1x1로 convolution하려면, 3x3x 3채널짜리 Filter를 사용학습하면 27개의 파라미터를 학습해야하지만, 3x3x1 filter와 1x1x3 필터를 사용하면 9+3 =12개의 파라미터를 학습하면서, 같은 기능을 할 수 있다.
  * DeepLab V3를 Encoder 파트로 가져오고, Encoder-Decoder 구조로 만들어, Decorder로 upsampling 수행
  * Encoder/Decoder, Dilated Convolution, SPP 를 통해 성능 향상

#### 5. Segmentation using 3D CNN (8:07)

* z축이 생기면서 Filter도 z축을 추가해주면, 파라미터 갯수가 크게 증가하지만, 구조 자체는 쉽게 적용 가능
* 영상이 크기 때문에, Patch로 나누어서 Patch-wise segmentation을 수행하기도 함
* 일반 영상에 비해, MRI 영상 등은 어떤 Patch더라도 밝기 패턴이 비슷하기 때문에 학습이 더 잘 되긴 함. 하지만, 전역적인 정보는 잃게 될 순 있음
* U-Net, FCN, ... 적용 가능

#### 6. Loss Function (10:00)

* 픽셀 단위로 레이블이 매칭이 되는지 확인함
* Classification과는 조금 다른 Loss function이 사용되는 경우가 많음
  * Segmentation에서는 Cross Entropy를 사용하지 않음
    * Gound Truth에서 object의 크기가 굉장히 작은 경우가 많음
    * 이때 소수의 부분만을 제외하고 대부분의 영역이 비슷하기 때문에, 단순히 Similiarity(CE)를 사용하면 전체적인 영역이 비슷하기 때문에 판별하기가 어려움
  * Weighted Cross Entropy에서는 beta값을 사용해서, Foreground나 Background에서 강조하게 됨
    * beta > 1 : false negative 감소. 왠만하면 foreground로 인식하도록
    * beta < 1 : false positive 감소. 왠만하면 background로 인식하도록
    * Balanced Cross Entropy: 두 개 term을 모두 beta로 조절하는 방식
  * Dice Loss <- Dice Coefficient
    * Segmentation ground truth와 Prediction 에서, 교집합의 비율이 얼마냐 되는가하는 Dice coefficient 응용
    * Background를 고려하지 않고, Foreground만 비교함
  * Cross Entropy와 Dice loss를 결합하여 사용하기도 함

#### 7. Segmentation Metric (08:36)

* Pixel Accuracy: Object 크기가 작아서 전체 영상에서 1%만 다르게 되면, True Negative가 들어간 Accuracy 식은 무조건 높기 나오기 때문에 적합하지 않을 수 있음
* Intersection over Union
* Dice Coefficient: True Positive에 좀 더 강조된 식
* Hausdorff distance: Noise에 취약하다
* Average distance: max를 average로 변경해서 노이즈에 좀 더 robust하도록 만든 식

#### 8. Quiz 8

* 7/7 100점~



## Week 5 (Chapter 9 & 10)

### Medical image enhancement (1)

### Medical image enhancement (2)



## Week 6 (Chapter 11 & 12)

### Medical image enhancement (3)

### Medical image registration (1)



## Week 7 (Chapter 13 & 14)

### Medical image registration (2)

### Medical image registration (3)
