DCASE2021 - Task 1 A - Baseline systems
-------------------------------------

저자:
**Irene Martin**, *Tampere University* 
[Email](mailto:irene.martinmorato@tuni.fi). 
Adaptations from the original code DCASE2020 - Task 1 by
**Toni Heittola**, *Tampere University* 


Getting started
===============

1. Clone repository from [Github](https://github.com/marmoi/dcase2021_task1a_baseline).
2. 라이브러리 설치: `pip install -r requirements.txt`.
3. model quantization에 따라 다음의 2가지로 실행할 수 있음 :  
   - keras model(deault): `python task1a.py` or  `./task1a.py`
   - TFLite: `python task1a_tflite.py` or  `./task1a_tflite.py`

### Anaconda installation

To setup Anaconda environment : 

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
	
**Note**: dcase_util 라이브러리를 사용하기 위해 tensorflow 2 설치가 필수.  
dcase_util toolbox >= ver.0.2.16 [dcase_util](https://github.com/DCASE-REPO/dcase_util) 


Introduction
============

이 베이스라인 시스템은 **Low-Complexity Acoustic Scene Classification with Multiple Devices (Subtask A) in Detection and Classification of Acoustic Scenes and Events 2021 (DCASE2021) challenge** 챌린지의 음향 이벤트 분류 과제A 에서 제공하는 시스템입니다.

Description
========

### Subtask A - Low-Complexity Acoustic Scene Classification with Multiple Devices

+ 사용되는 데이터셋:   
[TAU Urban Acoustic Scenes 2020 Mobile Development dataset](https://zenodo.org/record/3819968) is used.


오디오는 단일 장치 A로 녹음되었습니다. (device A, 48 kHz / 24bit / stereo).  
데이터셋은 10개의 음향 이벤트로 분류되어 있고, 총 40시간의 오디오가 포함되어 있습니다.  
데이터셋의 오디오는 모두 10초 길이입니다.

* 챌린지에 대한 자세한 설명은 [DCASE 챌린지 과제 설명](http://dcase.community/challenge2020/task-acoustic-scene-classification)을 참조. 
* 데이터셋에 대한 자세한 설명은 [DCASE Challenge task description](http://dcase.community/challenge2020/task-acoustic-scene-classification)을 참조.  
* 모델 사이즈 계산에 대한 자세한 설명은 [DCASE Challenge task description](http://dcase.community/challenge2020/task-acoustic-scene-classification)을 참조.  
* Model calculation for Keras models is implemented in `model_size_calculation.py`  


#### System description

시스템에서는 먼저 10초의 오디오에 대해 log mel-band 를 먼저 추출합니다.  
두 개의 CNN layer와 한 개의 FCN layer 로 구성된 네트워크가 오디오 신호에 레이블을 할당하는 컴볼루션 신경망(CNN) 기반 접근을 구현합니다. 
모델 사이즈는 keras quantization : 90.82 KB, TFLite quantization : 89.82 KB 입니다.  


##### Parameters

###### Acoustic features

- Analysis frame 40 ms (50% hop size)
- Log mel-band energies (40 bands)

###### Neural network 구조

- Input shape: 40 * 500 (10 seconds)
- Architecture:
  - CNN layer #1
    - 2D Convolutional layer (filters: 16, kernel size: 7) + Batch normalization + ReLu activation
  - CNN layer #2
    - 2D Convolutional layer (filters: 16, kernel size: 7) + Batch normalization + ReLu activation
    - 2D max pooling (pool size: (5, 5)) + Dropout (rate: 30%)
  - CNN layer #3
    - 2D Convolutional layer (filters: 32, kernel size: 7) + Batch normalization + ReLu activation
    - 2D max pooling (pool size: (4, 100)) + Dropout (rate: 30%)
  - Flatten
  - Dense layer #1
    - Dense layer (units: 100, activation: ReLu )
    - Dropout (rate: 30%)
  - Output layer (activation: softmax/sigmoid)
- Learning (epochs: 200, batch size: 16, data shuffling=True between epochs)
  - Optimizer: Adam (learning rate: 0.001)
- Model selection:
  - train set의 약 30%가 validation set으로 할당됩니다.
  - 매 epoch마다 validation set으로 모델 성능을 평가하고 best 모델을 선택합니다. 
  
**Network summary**

    _________________________________________________________________
	Layer (type)                 Output Shape              Param #   
	=================================================================
	conv2d (Conv2D)              (None, 40, 500, 16)       800       
	_________________________________________________________________
	batch_normalization (BatchNo (None, 40, 500, 16)       64        
	_________________________________________________________________
	activation (Activation)      (None, 40, 500, 16)       0         
	_________________________________________________________________
	conv2d_1 (Conv2D)            (None, 40, 500, 16)       12560     
	_________________________________________________________________
	batch_normalization_1 (Batch (None, 40, 500, 16)       64        
	_________________________________________________________________
	activation_1 (Activation)    (None, 40, 500, 16)       0         
	_________________________________________________________________
	max_pooling2d (MaxPooling2D) (None, 8, 100, 16)        0         
	_________________________________________________________________
	dropout (Dropout)            (None, 8, 100, 16)        0         
	_________________________________________________________________
	conv2d_2 (Conv2D)            (None, 8, 100, 32)        25120     
	_________________________________________________________________
	batch_normalization_2 (Batch (None, 8, 100, 32)        128       
	_________________________________________________________________
	activation_2 (Activation)    (None, 8, 100, 32)        0         
	_________________________________________________________________
	max_pooling2d_1 (MaxPooling2 (None, 2, 1, 32)          0         
	_________________________________________________________________
	dropout_1 (Dropout)          (None, 2, 1, 32)          0         
	_________________________________________________________________
	flatten (Flatten)            (None, 64)                0         
	_________________________________________________________________
	dense (Dense)                (None, 100)               6500      
	_________________________________________________________________
	dropout_2 (Dropout)          (None, 100)               0         
	_________________________________________________________________
	dense_1 (Dense)              (None, 10)                1010      
	=================================================================
	Total params: 46,246
	Trainable params: 46,118
  	Non-trainable params: 128
  	_________________________________________________________________

     Input shape                     : (None, 40, 500, 1)
     Output shape                    : (None, 10)

  
#### Results for development dataset

데이터셋 *TAU Urban Acoustic Scenes 2020 Mobile Development dataset* 이 베이스라인 평가에 사용되었습니다.   
결과는 GPU(using Nvidia Tesla V100 GPU card) 모드로 실행된 결과이며, 10번의 테스트 후 mean값과 standard deviation값의 결과입니다.  
 

| Scene label       | Log Loss |   A   |   B   |   C   |   S1  |   S2  |   S3  |   S4  |   S5  |   S6  | Accuracy|  
| -------------     | -------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ------- | 
| Airport           | 1.429    | 1.156 | 1.196 | 1.457 | 1.450 | 1.187 | 1.446 | 1.953 | 1.505 | 1.502 | 40.5%   | 
| Bus               | 1.317    | 0.796 | 1.488 | 0.908 | 1.569 | 0.997 | 1.277 | 1.939 | 1.377 | 1.503 | 47.1%   |  
| Metro             | 1.318    | 0.761 | 1.030 | 0.963 | 2.002 | 1.522 | 1.173 | 1.200 | 1.437 | 1.770 | 51.9%   |  
| Metro station     | 1.999    | 1.814 | 2.079 | 2.368 | 2.058 | 2.339 | 1.781 | 1.921 | 1.917 | 1.715 | 28.3%   |  
| Park              | 1.166    | 0.458 | 1.022 | 0.381 | 1.130 | 0.845 | 1.206 | 2.342 | 1.298 | 1.814 | 69.0%   |  
| Public square     | 2.139    | 1.542 | 1.708 | 1.804 | 2.254 | 1.866 | 2.146 | 3.012 | 2.716 | 2.202 | 25.3%   |  
| Shopping mall     | 1.091    | 0.946 | 0.830 | 1.091 | 1.302 | 1.293 | 1.196 | 1.140 | 0.976 | 1.042 | 61.3%   |  
| Pedestrian street | 1.827    | 1.178 | 1.310 | 1.454 | 1.789 | 1.656 | 1.883 | 3.146 | 2.068 | 1.956 | 38.7%   |  
| Traffic street    | 1.338    | 0.854 | 1.154 | 1.368 | 1.104 | 1.325 | 1.356 | 1.747 | 0.764 | 2.365 | 62.0%   |  
| Tram              | 1.105    | 0.674 | 1.116 | 1.016 | 0.866 | 1.378 | 0.750 | 0.942 | 1.776 | 1.424 | 53.0%   |  
| -------------     | -------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ------- |  
| Average          | **1.473**<br>(+/-0.051)| 1.018 | 1.294 | 1.282 | 1.552 | 1.441 | 1.421 | 1.934 | 1.583 | 1.729 |  **47.7%**<br>(+/-0.977)|  
                                                                                

**Note:** The reported system performance is not exactly reproducible due to varying setups. However, you should be able obtain very similar results.


### Model size
 Manual quantization acoustic model

    | Name                        | Param     |  NZ Param  | Size                        |  NZ Size                    | 
    | --------------------------- | --------- |  --------- | --------------------------- |  -------------------------- |
    | conv2d_3                    | 800       |  800       | 1.562 KB                    |  1.562 KB                   |
    | batch_normalization_3       | 64        |  64        | 256 bytes                   |  256 bytes                  |
    | activation_3                | 0         |  0         | 0 bytes                     |  0 bytes                    |
    | conv2d_4                    | 12560     |  12560     | 24.53 KB                    |  24.53 KB                   |
    | batch_normalization_4       | 64        |  64        | 256 bytes                   |  256 bytes                  |
    | activation_4                | 0         |  0         | 0 bytes                     |  0 bytes                    |
    | max_pooling2d_2             | 0         |  0         | 0 bytes                     |  0 bytes                    |
    | dropout_3                   | 0         |  0         | 0 bytes                     |  0 bytes                    |
    | conv2d_5                    | 25120     |  25120     | 49.06 KB                    |  49.06 KB                   |
    | batch_normalization_5       | 128       |  128       | 512 bytes                   |  512 bytes                  |
    | activation_5                | 0         |  0         | 0 bytes                     |  0 bytes                    |
    | max_pooling2d_3             | 0         |  0         | 0 bytes                     |  0 bytes                    |
    | dropout_4                   | 0         |  0         | 0 bytes                     |  0 bytes                    |
    | flatten_1                   | 0         |  0         | 0 bytes                     |  0 bytes                    |
    | dense_2                     | 6500      |  6500      | 12.7 KB                     |  12.7 KB                    |
    | dropout_5                   | 0         |  0         | 0 bytes                     |  0 bytes                    |
    | dense_3                     | 1010      |  1010      | 1.973 KB                    |  1.973 KB                   |
    | --------------------------- | --------- |  --------- | --------------------------- |  -------------------------- |
    | Total                       | 46246     |  46246     | 93,004 bytes (90.82 KB)     |  93,004 bytes (90.82 KB)    |

 TFLite acoustic model

    | Name                        | Param     | NZ Param  | Size                        |  NZ Size                   |   
    | --------------------------- | --------- | --------- | --------------------------- |  --------------------------|  
    | ReadVariableOp              | 784       | 784       | 1.531 KB                    |  1.531 KB                  |   
    | Conv2D_bias                 | 16        | 16        | 32 bytes                    |  32 bytes                  |   
    | ReadVariableOp              | 12544     | 12544     | 24.5 KB                     |  24.5 KB                   |   
    | Conv2D_bias                 | 16        | 16        | 32 bytes                    |  32 bytes                  |   
    | ReadVariableOp              | 25088     | 25088     | 49 KB                       |  49 KB                     |   
    | Conv2D_bias                 | 32        | 32        | 64 bytes                    |  64 bytes                  |   
    | transpose                   | 6400      | 6400      | 12.5 KB                     |  12.5 KB                   |   
    | MatMul_bias                 | 100       | 100       | 200 bytes                   |  200 bytes                 |   
    | transpose                   | 1000      | 1000      | 1.953 KB                    |  1.953 KB                  |   
    | MatMul_bias                 | 10        | 10        | 20 bytes                    |  20 bytes                  |   
    | --------------------------- | --------- | --------- | --------------------------- |  --------------------------|  
    | Total                       | 45990     | 45990     | 91,980 bytes (89.82 KB)     |  91,980 bytes (89.82 KB)   |
Usage
=====

For the subtask there are two separate application (.py file):

- `task1a.py`: DCASE2021 baseline for Task 1A, **with Keras model quantization**
- `task1a_tflite.py`: DCASE2021 baseline for Task 1A,  **with TFLite quantization**

### Application arguments

도움말이 필요할 경우:  ``python task1a.py -h``.

| Argument                    |                                   | Description                                                  |
| --------------------------- | --------------------------------- | ------------------------------------------------------------ |
| `-h`                        | `--help`                          | Application help.                                            |
| `-v`                        | `--version`                       | Show application version.                                    |
| `-m {dev,eval}`             | `--mode {dev,eval}`               | Selector for application operation mode                      |
| `-s PARAMETER_SET`          | `--parameter_set PARAMETER_SET`   | Parameter set id. Can be also comma separated list e.g. `-s set1,set2,set3``. In this case, each set is run separately. |
| `-p FILE`                   | `--param_file FILE`               | Parameter file (YAML) to overwrite the default parameters    |
| `-o OUTPUT_FILE`            | `--output OUTPUT_FILE`            | Output file                                                  |
|                             | `--overwrite`                     | Force overwrite mode.                                        |
|                             | `--download_dataset DATASET_PATH` | Download dataset to given path and exit                      |
|                             | `--show_parameters`               | Show active application parameter set                        |
|                             | `--show_sets`                     | List of available parameter sets                             |
|                             | `--show_results`                  | Show results of the evaluated system setups                  |

### Operation modes

**Development mode** - `dev` (default)

Usage example: `python task1a.py` or `python task1a.py -m dev`

**Challenge mode** - `eval` (-eval 모드는 동작하지 않음)  

시스템 결과를 csv로 저장 : `python task1a.py -m eval -o output.csv`

### System parameters

기준 시스템은 서로 다른 시스템 설정 간에 유연한 전환이 가능하도록 다단계 파라미터 덮어쓰기를 지원합니다. 
매개변수 변경사항은 매개변수 섹션에서 계산된 해시를 사용하여 추적됩니다.  
이러한 매개 변수 해시는 데이터(기능/임베딩, 모델 또는 결과)를 저장할 때 스토리지 파일 경로에 사용됩니다.  
이 접근법을 사용하여 시스템은 특정 매개 변수 세트에 대해 형상/임베딩, 모델 및 결과를 한 번만 계산하고, 그 후 이 사전 계산된 데이터를 재사용합니다. 


#### Parameter overwriting

매개변수는 내부적으로 처리되어(`dcase_util.containers.DCASEAppParameterContainer`) YAML-formatted 파일에 저장됩니다.  


#### Parameter file

매개변수 YAML 파일은 3 block으로 구분됩니다.: 

- `active_set`, default parameter set id
- `sets`, list of dictionaries
- `defaults`, dictionary containing default parameters which are overwritten by the `sets[active_set]`  

섹션 이름에 따라 매개 변수 내부의 매개 변수가 다르게 처리되는 경우도 있습니다. 일반적으로 가능한 각 방법에 대한 매개 변수를 포함하는 주 섹션('feature_extractor')과 방법 매개 변수 섹션('feature_extractor_method_parameters')이 있습니다. 매개변수가 처리되면 매개변수 아래의 메소드 매개변수 섹션에서 메인 섹션으로 올바른 메소드 매개변수가 복사됩니다. 이를 통해 많은 방법이 매개 변수화되고 쉽게 접근할 수 있습니다.  

#### Parameter hash

매개변수 해시는 각 매개변수 섹션에 대해 계산된 MD5 해시입니다.  
이러한 해시를 보다 강력하게 만들기 위해 해시 계산 전에 일부 사전 처리가 적용됩니다.


## Extending the baseline

기준 시스템을 확장하는 가장 쉬운 방법은 시스템 매개 변수(YAML file)를 수정하는 것입니다.
3가지의 extra.yaml 확장 모델을 예시로 제공합니다.  

**Example 1**

extra1은 MLP 기반 시스템으로 데이터 처리 체인은 500개 이상의 특징 벡터를 계산하는 체인으로 대체됩니다.  
Parameter file `extra.yaml`: 
        
    active_set: minimal-mlp
    sets:
      - set_id: minimal-mlp
        description: Minimal MLP system
        data_processing_chain:
          method: mean_aggregation_chain
        data_processing_chain_method_parameters:
          mean_aggregation_chain:
            chain:
              - processor_name: dcase_util.processors.FeatureReadingProcessor
              - processor_name: dcase_util.processors.NormalizationProcessor
                init_parameters:
                  enable: true
              - processor_name: dcase_util.processors.AggregationProcessor
                init_parameters:
                  aggregation_recipe:
                    - mean
                  win_length_frames: 500
                  hop_length_frames: 500
              - processor_name: dcase_util.processors.DataShapingProcessor
                init_parameters:
                  axis_list:
                    - time_axis
                    - data_axis
        learner:
          method: mlp_mini
        learner_method_parameters:
          mlp_mini:
            random_seed: 0
            keras_profile: deterministic
            backend: tensorflow
            validation_set:
              validation_amount: 0.20
              balancing_mode: identifier_two_level_hierarchy
              seed: 0
            data:
              data_format: channels_last
              target_format: same
            model:
              config:
                - class_name: Dense
                  config:
                    units: 50
                    kernel_initializer: uniform
                    activation: relu
                    input_shape:
                      - FEATURE_VECTOR_LENGTH
                - class_name: Dropout
                  config:
                    rate: 0.2
                - class_name: Dense
                  config:
                    units: 50
                    kernel_initializer: uniform
                    activation: relu
                - class_name: Dropout
                  config:
                    rate: 0.2
                - class_name: Dense
                  config:
                    units: CLASS_COUNT
                    kernel_initializer: uniform
                    activation: softmax
            compile:
              loss: categorical_crossentropy
              metrics:
                - categorical_accuracy
            optimizer:
              class_name: Adam
            fit:
              epochs: 50
              batch_size: 64
              shuffle: true
            callbacks:
              StasherCallback:
                monitor: val_categorical_accuracy
                initial_delay: 25

Command to run the system:

    python task1a.py -p extra.yaml

**Example 2**

extra2는 네트워크 크기가 작아지도록 베이스라인을 약간 수정합니다.   
Parameter file `extra.yaml`: 
        
    active_set: baseline-minified
    sets:
      - set_id: baseline-minified
        description: Minified DCASE2021 baseline subtask A minified
        learner_method_parameters:
          cnn:
            model:
              constants:
                CONVOLUTION_KERNEL_SIZE: 3            
        
              config:
                - class_name: Conv2D
                  config:
                    input_shape:
                      - FEATURE_VECTOR_LENGTH   # data_axis
                      - INPUT_SEQUENCE_LENGTH   # time_axis
                      - 1                       # sequence_axis
                    filters: 8
                    kernel_size: CONVOLUTION_KERNEL_SIZE
                    padding: CONVOLUTION_BORDER_MODE
                    kernel_initializer: CONVOLUTION_INIT
                    data_format: DATA_FORMAT
                - class_name: Activation
                  config:
                    activation: CONVOLUTION_ACTIVATION
                - class_name: MaxPooling2D
                  config:
                    pool_size:
                      - 5
                      - 5
                    data_format: DATA_FORMAT
                - class_name: Conv2D
                  config:
                    filters: 16
                    kernel_size: CONVOLUTION_KERNEL_SIZE
                    padding: CONVOLUTION_BORDER_MODE
                    kernel_initializer: CONVOLUTION_INIT
                    data_format: DATA_FORMAT
                - class_name: Activation
                  config:
                    activation: CONVOLUTION_ACTIVATION
                - class_name: MaxPooling2D
                  config:
                    pool_size:
                      - 4
                      - 100
                    data_format: DATA_FORMAT
                - class_name: Flatten      
                - class_name: Dense
                  config:
                    units: 100
                    kernel_initializer: uniform
                    activation: relu    
                - class_name: Dense
                  config:
                    units: CLASS_COUNT
                    kernel_initializer: uniform
                    activation: softmax                        
            fit:
                epochs: 100
                                  
Command to run the system:

    python task1a.py -p extra.yaml


**Example 3**

extra3에서는 kernel size=3일 때, kernel size=5일 때의 다른 설정이 순차적으로 진행됩니다.  
Parameter file `extra.yaml`: 
        
    active_set: baseline-kernel3
    sets:
      - set_id: baseline-kernel3
        description: DCASE2021 baseline for subtask A with kernel 3
        learner_method_parameters:
          cnn:
            model:
              constants:
                CONVOLUTION_KERNEL_SIZE: 3
            fit:
              epochs: 100                    
      - set_id: baseline-kernel5
        description: DCASE2021 baseline for subtask A with kernel 5
        learner_method_parameters:
          cnn:
            model:
              constants:
                CONVOLUTION_KERNEL_SIZE: 5
            fit:
              epochs: 100
                
Command to run the system:

    python task1a.py -p extra.yaml -s baseline-kernel3,baseline-kernel5

To see results:
    
    python task1a.py --show_results


Code
====

코드는 python으로 작성되었으며, dcase_util와 tensorflow, pytorch 라이브러리를 사용했습니다.  


### File structure

      .
      ├── task1a.py                     # Baseline system for subtask A
      ├── task1a.yaml                   # Configuration file for task1a.py
      ├── task1a_tflite.py              # Baseline system for subtask A with TFLite quantification
      |
      ├── model_size_calculation.py     # Utility function for calculating model size 
      ├── utils.py                      # Common functions shared between tasks
      |
      ├── README.md                     # This file
      └── requirements.txt              # External module dependencies

Changelog
=========

#### 2.0.0 / 2021-02-19


License
=======

This software is released under the terms of the [MIT License](https://github.com/toni-heittola/dcase2020_task1_baseline/blob/master/LICENSE).
