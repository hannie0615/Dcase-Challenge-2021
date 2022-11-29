
DCASE2021 - Task 4 - Baseline for Sound Event Detection and Separation in Domestic Environments.
-------------------------------------


## Requirements

쉘 파일을 제공합니다. : `conda_create_environment.sh` 
(아래의 코드는 한줄씩 실행하는 것을 권장합니다.)

## Dataset

필요한 데이터셋은 `generate_dcase_task4_2021.py` 스크립트로 다운로드할 수 있습니다. 
또는 사이트 [zenodo_evaluation_dataset][zenodo] 에서 직접 내려받을 수 있습니다. 


## Usage  

- `python generate_dcase_task4_2021.py --basedir="../../data"` (basedir = 데이터 저장 위치)

다음의 데이터셋이 다운로드됩니다.  
[FUSS][fuss_git], [FSD50K][FSD50K], [desed_soundbank][desed] or [desed_real][desed].


## Real data  

**weak, unlabeled, validation data** 

만약 다운로드 후에도 desed_real 데이터가 부족할 경우 여러 번 실행해보는 것을 추천합니다.  
그래도 없는 missing files는 task organisers에게 문의해서 받을 수 있습니다.  
(in priority to Francesca Ronchini and Romain serizel).

audioset 다운로드 : 
```python
import desed
desed.download_audioset_data("PATH_TO_YOUR_DESED_REAL_FOLDER")
```
`PATH_TO_YOUR_DESED_REAL_FOLDER` 는 `desed_real`의 위치입니다.
(ex. `DESED_task/data/raw_datasets/desed_real`)

###  FSD50K, FUSS or DESED already downloaded ?
If you already have "FUSS", "FSD50K", "desed_soundbank" or "desed_real" (audioset data same as previous years),  
Specify their path using the specified arguments (e.g `--fuss "path_to_fuss_basedir"`),  
see `python generate_dcase_task4_2021.py --help`.



## Training
두 개의 베이스라인이 제공됩니다. 
- SED baseline
- joint Separation+SED baseline.(*지금은 사용 불가)  


## SED Baseline

#### Usage
SED baseline을 실행합니다.  
- `python train_sed.py`

(* 사전 훈련된 체크포인트 : [here][zenodo_pretrained_models] )

SED baseline을 테스트합니다. (using validation real data) 
  - `python train_sed.py --test_from_checkpoint /path/to/downloaded.ckpt`

텐서보드 로그를 체크합니다.  
`tensorboard --logdir="path/to/exp_folder"`

#### Results  

Dataset | **PSDS-scenario1** | **PSDS-scenario2** | *Intersection-based F1* | *Collar-based F1*
--------|--------------------|--------------------|-------------------------|-----------------
Dev-test| **0.353**          | **0.553**          | 79.5%                   | 42.1%

#### 평가지표  

Collar-based = event-based. More information about the metrics in the [webpage][dcase21_webpage]  
결과는 'student model'와 'teacher model' 중 'teacher'값을 썼습니다. (dev-test에서 선택됨)  
평가지표는 psds 메트릭을 사용했습니다. 


**Note**) 데이터셋 폴더의 위치는 YAML 파일에서 수정할 수 있습니다.  

`--conf_file="confs/sed_2.yaml` :  다른 configuration YAML 사용하기  
`--log_dir="./exp/2021_baseline` : checkpoints 와 logging 저장하는 위치 바꾸기  
`--resume_from_checkpoint` : 훈련 다시 시작하기


#### Architecture

이 베이스라인은 [2020 DCASE Task 4 baseline][dcase_20_repo] 에서 기반합니다.

DCASE 2020에 비해 달라진 점: 

* Features: hop size를 255->256으로
* **합성 데이터 사용** (*중요)
* 얼리 스탑 없음(200 epochs)
* 인스턴스마다 표준화 (using min-max approach)
* 한 배치 안에서 weak and synthetic 데이터가 믹스됨.
* Batch size=48 (비율은 똑같이 1/4 synthetic, 1/4 weak, 1/2 unlabelled)
* synthetic validation score 에서는 event-based F1 대신 Intersection-based F1 지표가 쓰임.



## SSEP + SED Baseline

#### Usage
사전 훈련된 범용 사운드 분리 모델(YFCC100m)을 다운로드해서 SSEP+SED 베이스라인을 실행할 수 있습니다.  
instructions [here][google_sourcesep_repo] using the Google Cloud SDK ([installation instructions][sdk_installation_instructions]):

- `gsutil -m cp -r gs://gresearch/sound_separation/yfcc100m_mixit_model_checkpoint .`  

이때 SED baseline에서 얻은 사전 훈련된 SED 시스템이 필요합니다.  
데이터셋을 다운받아 직접 훈련하거나 ([here][zenodo_pretrained_models]) 사전 훈련된 시스템을 다운받을 수 있습니다. 

- `wget -O 2021_baseline_sed.tar.gz "https://zenodo.org/record/4639817/files/2021_baseline_sed.tar.gz?download=1"` 
- `tar -xzf 2021_baseline_sed.tar.gz` 

configuration 폴더에서 `./confs/sep+sed.yaml` yaml 파일을 확인합니다. 
SED 체크포인트와 YAML 파일에 대한 경로와 사전 훈련된 소리 분리 모델에 대한 경로가 올바르게 설정되어 있는지 확인해야 합니다. 

첫번째로 소리를 분리합니다.  
- `python run_separation.py` 

그런 다음 SED 모델은 분리된 데이터에서 미세 조정됩니다.  
- `python finetune_on_separated.py` 

[here][zenodo_pretrained_models]에서 사전 훈련된 체크포인트와 텐서 로그를 제공합니다. 

검증 데이터(validation real)로 테스트합니다.    
  - `python train_sed.py --test_from_checkpoint /path/to/downloaded.ckpt`   
  
로그를 확인합니다.  
  -  `tensorboard --logdir="path/to/exp_folder"`   

#### Results  

Dataset | **PSDS-scenario1** | **PSDS-scenario2** | *Intersection-based F1* | *Collar-based F1*
--------|--------------------|--------------------|-------------------------|-----------------
Dev-test| **0.373**          | **0.549**          | 77.2%                   | 44.3%



#### Architecture

SSEP + SED baseline은 사전 훈련된 SED 모델과, 사전 훈련된 소리 분리 모델을 사용합니다.   
SED 모델은 사전 처리를 통해 얻은 분리된 사운드 이벤트에 대해 미세 조정됩니다.  
소리 분리 모델은 TDCN++[4]를 기반으로 하며 YFCC100m 데이터 세트[3]에서 MixIT[5]로 unsupervised 방식으로 훈련됩니다.  

예측값(pred)은 미세 조정된 SED모델을 orginal SED 모델과 앙상블하여 얻습니다.  
앙상블은 예측의 두 모델의 가중 평균에 의해 수행되며, 가중치는 훈련 중에 학습됩니다. 

----------

[dcase21_webpage]: http://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments
[dcase_20_repo]: https://github.com/turpaultn/dcase20_task4/tree/master/baseline
[desed]: https://github.com/turpaultn/DESED
[fuss_git]: https://github.com/google-research/sound-separation/tree/master/datasets/fuss
[fsd50k]: https://zenodo.org/record/4060432
[zenodo_pretrained_models]: https://zenodo.org/record/4639817
[google_sourcesep_repo]: https://github.com/google-research/sound-separation/tree/master/datasets/yfcc100m
[sdk_installation_instructions]: https://cloud.google.com/sdk/docs/install
[zenodo_evaluation_dataset]: https://zenodo.org/record/4892545#.YMHH_DYzadY


#### References
[1] L. Delphin-Poulat & C. Plapous, technical report, dcase 2019.  
[2] Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017).  
[3] Thomee, Bart, et al. "YFCC100M: The new data in multimedia research." Communications of the ACM 59.2 (2016): 64-73.  
[4] Kavalerov, Ilya, et al. "Universal sound separation." 2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE, 2019.  
[5] Wisdom, Scott, et al. "Unsupervised sound separation using mixtures of mixtures." arXiv preprint arXiv:2006.12701 (2020).  
[6] Turpault, Nicolas, et al. "Improving sound event detection in domestic environments using sound separation." arXiv preprint arXiv:2007.03932 (2020).  
