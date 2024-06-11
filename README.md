# 알약 분류 모델
알약 분류하는 모델 제작

# 1. 문제 정의
약 봉지를 잃어버린 경우에는 약에 대한 정보를 알 수 없어서 사용자에게 매우 중요한 문제가 발생할 수 있다. 특히 약물 알레르기나 부작용과 같은 중요한 정보를 파악하지 못하면 심각한 상황에 빠질 수도 있다. 이러한 문제를 해결하기 위해 약물 사진을 분류하고 해당 약물에 대한 정보를 제공하는 모델을 개발하는 것은 매우 중요하다.

우선적으로 이러한 모델을 제작하기 위해서는 컴퓨터 비전과 머신러닝 기술을 활용해야 한다. 머신러닝 알고리즘 중에서는 이미지 분류에 주로 사용되는 딥러닝 모델을 활용할 수 있습니다. 딥러닝은 복잡한 이미지 패턴을 인식하고 분류하는 데 강력한 성능을 보이므로 약물 사진을 분류하는 데 적합하다.

약물 사진을 분류하고 정보를 제공함으로써 사용자들은 보다 안전하고 효과적인 의약품 사용이 가능해질 것이다.



![flowchart](https://github.com/hiinnnii/pill_classification_model/blob/main/AIP_%E1%84%80%E1%85%B5%E1%84%86%E1%85%A1%E1%86%AF_flowchart.png)
------------------

# 2. 필요한 라이브러리
colab 이용

torch = 2.0.1+cu117
torchvision = 0.15.2+cu117

[코랩에서 제공하는 라이브러리 이용]

## torch 관련 라이브러리
* from torchvision import models, transforms
* from torch.utils.data import DataLoader, Dataset
* import torch
* import torch.nn as nn
* import torchaudio
* import torch.nn.functional as F
* from torch.optim.lr_scheduler import MultiStepLR, StepLR, CyclicLR, CosineAnnealingLR, ExponentialLR
* from torchsummary import summary
## 일반 라이브러리
* import argparse
* import numpy as np
* import random
* import os
* from PIL import Image
* import matplotlib.pyplot as plt
* import time
* from tqdm import tqdm
* from sklearn.metrics import f1_score
* from sklearn.model_selection import train_test_split

------------------

# 3. 데이터셋

* train / test dataset 모두 직접 촬영

  
https://drive.google.com/file/d/1vaa-CbI_AYiFQFt3pQOCbsfG-Uo7_1t2/view?usp=drive_link
  
* 0 ~ 9까지 총 10개 class 이용

0 : 덱시부정

1 : 록소닌정

2 : 리노에바티스서방캡슐

3 : 모사핀정

4 : 미프론정

5 : 세파록실캡슐

6 : 소론도정

7 : 엑사드 캡슐

8 : 영진브로멜라인장용정

9 : 코데닝정

## Data agmentation
![aumentation](https://github.com/hiinnnii/pill_classification_model/blob/main/AIP_%E1%84%80%E1%85%B5%E1%84%86%E1%85%A1%E1%86%AF_aumentation.png)
------------------

# 4. 사용모델
![AIP_기말_Ensemble.png](https://github.com/hiinnnii/pill_classification_model/blob/main/AIP_%E1%84%80%E1%85%B5%E1%84%86%E1%85%A1%E1%86%AF_Ensemble.png)

* ResNet 50 + ResNet 18 + VGG 16 더한 Ensemble 모델 활용

* Ensemble 모델 사용한 이유 : 성능향상, 과적합 감소 시키기 위해 Ensemble 모델을 사용

------------------

# 5. 모델 평가
### 모델의 accuracy를 기준으로 평가

![AIP_ablone](https://github.com/hiinnnii/pill_classification_model/blob/main/AIP_%E1%84%80%E1%85%B5%E1%84%86%E1%85%A1%E1%86%AF_abalone.png)

### 각 class별 성능을 확인

![clas 성능확인](https://github.com/hiinnnii/pill_classification_model/blob/main/AIP_%E1%84%80%E1%85%B5%E1%84%86%E1%85%A1%E1%86%AF_%E1%84%80%E1%85%A7%E1%86%AF%E1%84%80%E1%85%AA.png)

class별 성능 확인 결과 class 6의 성능이 안좋은 것을 확인할 수 있다.

[원인 분석] 

test data중 유일하게 후레쉬 이용해 촬영한 데이터가 들어있음을 확인하였다.

성능 저하의 원인이 되었다고 판단하였다.

------------------

# 6. 한계 및 추후 개선 사항

[한계]
* 현재 모델은 10개의 class만 분류할 수 있는 모델
* 알약의 음각으로 구분하는 것이 아닌 모양으로 구분
* 모양이 비슷한 경우 헷갈리는 경우 발생

[개선사항]
* class 늘리기
* 알약의 음각으로 구분할 수 있도록 더 많은 train data set 넣기
* train / test 직접 촬영한 것이라 model의 성능이 좋은 것으로 판단 -> test에 더 다양한 data를 넣거나, test dataset을 구성할 때 타인이 찍어준 것으로 test dataset을 구성해볼 필요가 있음


