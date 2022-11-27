# Dcase-Challenge-2021

dcase 2021 챌린지에서 음성 분류 모델을 참고합니다.  
task1, task4의 baselilne을 실행하고, 각 모델을 튜닝합니다. 

## Task1 - task A : Acoustic Scnene Classification

### 챌린지 개요
This is the baseline system for the Low-Complexity Acoustic Scene Classification with Multiple Devices (Subtask A) in Detection and Classification of Acoustic Scenes and Events 2021 (DCASE2021) challenge.

테스트 녹음을 미리 정의된 10개의 음향 장면 중 하나로 분류하는 것을 목표합니다.  

> **task A: Classification of data from multiple devices (일반화 높음)**  
task B: Classificatoin of audio and video data (복잡함) - 다루지 않음
> 

  ![t1](https://user-images.githubusercontent.com/50253860/204144277-0a1ebd21-d0c0-4491-aa37-074288ec85b2.png)


제공되는 오디오 데이터는 4개의 다른 장치를 사용하여 10개의 다른 음향 씬에 대한 12개의 유럽 도시의 녹음이 포함되어 있습니다.

Acoustic scenes (10) :

> Airport - `airport`/ Indoor shopping mall - `shopping_mall`/ Metro station - `metro_station`
Pedestrian street - `street_pedestrian`/ Public square - `public_square`
Street with medium level of traffic - `street_traffic`/ Travelling by a tram - `tram`
Travelling by a bus - `bus`/ Travelling by an underground metro - `metro`/ Urban park - `park`
> 

**Each acoustic scene has 1440 segments (240 minutes of audio). The dataset contains in total 40 hours of audio.**

### 다운로드 및 환경설정

```
# 작업 환경 
tensorflow-gpu               2.4.0
tensorflow                   2.4.0
cudatoolkit                  11.3
cudnn                        8.1 
Python                       3.6.8

# anaconda 환경 설정하기
conda create -n dcase1 python=3.6 # 가상환경 생성
conda activate dcase1 # 가상환경 활성화
conda deactivate # 가상환경 비활성화

# 라이브러리 다운받기
conda install ipython
conda install numpy
conda install tensorflow-gpu=2.4.0
conda install -c anaconda cudatoolkit
conda install -c anaconda cudnn
pip install librosa
pip install absl-py
pip install sed_eval
pip install pyyaml==5.3.1
pip install pyparsing==2.2.1
```
### baseline 파일 구조
```
# 파일 구조 
  ├── task1a.py                     # Baseline system for subtask A
  ├── task1a.yaml                   # Configuration file for task1a.py(초기 설정?)
  ├── task1a_tflite.py              # Baseline for subtask A with TFLite quantification
  |
  ├── model_size_calculation.py     # Utility function for calculating model size 
  ├── utils.py                      # Common functions shared between tasks
  |
  ├── README.md                     # This file
  └── requirements.txt              # External module dependencies(라이브러리)
```

### 실행문
```
> python task1a.py
```

### 실행 결과 
```
Evaluation
=====================
Logloss       | 1.505
Accuracy      | 45.5%
```



## Task4 : Sound Event Detection

### 챌린지 개요
task4의 목표는 약하게 레이블이 지정되었거나(*음성 데이터에 시간 정보 없이 이벤트 정보만 있는 데이터를 말함)
레이블이 지정되지 않은 실제 데이터(*이벤트 정보 없음)와 
강하게 레이블이 지정된 시뮬레이션 데이터(*이벤트 정보와 시간 정보까지 있음)를 사용하여 소리 이벤트를 감지하는 시스템을 평가하는 것입니다.

![t4](https://user-images.githubusercontent.com/50253860/204144332-e6d43a22-f6c4-4abf-951a-b15dee3cb796.png) 


훈련, 검증, 평가는 DESED dataset을 제공합니다.

> 데이터의 10가지 사운드 이벤트는 다음과 같습니다. 
연설Speech  
개Dog  
고양이Cat  
알람/벨/울림Alarm_bell_ringing  
그릇Dishes  
전유Frying  
블렌더Blender  
흐르는 물Running_water  
진공 청소기Vacuum_cleaner  
전기면도기/칫솔Electric_shaver_toothbrush  

### 다운로드 및 환경설정

```
# 작업 환경
tensorflow-gpu               2.7.0
tensorflow                   2.7.0

cuda toolkit: 11.3 ver
cudnn: 8.2.1 ver (for cuda 11.3)

torchmetrics                 '0.7.3'
pytorch_lightning            '1.4.9'
torch                        '1.10.0+cu113'

# 라이브러리 다운받기
pip install setuptools==59.5.0
pip install h5py
pip install soundfile==0.10.2
pip install codecarbon

# sox 다운로드 필요(https://sourceforge.net/projects/sox/files/latest/download)
# C++ 컴파일러 필요(https://visualstudio.microsoft.com/downloads/)
```

### 실행문
```
> python train_sed.py
```

### 실행 결과 
```
DATALOADER:0 TEST RESULTS # 2021ver. ep200
{'hp_metric': 0.5264670008136499,
 'test/student/event_f1_macro': 0.3943663101819644,
 'test/student/intersection_f1_macro': 0.6349045259670811,
 'test/student/loss_strong': 0.11856575310230255,
 'test/student/psds_score_scenario1': 0.3394053403602582,
 'test/student/psds_score_scenario2': 0.5264670008136499,
 'test/teacher/event_f1_macro': 0.4253272409720047,
 'test/teacher/intersection_f1_macro': 0.6592853435971635,
 'test/teacher/loss_strong': 0.11480239033699036,
 'test/teacher/psds_score_scenario1': 0.35144223211434455,
 'test/teacher/psds_score_scenario2': 0.5428737223292076}
```






