# 알약 분류 모델
알약 분류하는 모델 제작

# 1. 문제 정의
약 봉지를 잃어버린 경우 약에 대한 정보를 알지 못하는 문제가 발생한다. 따라서 이를 해결하기 위해 약의 사진을 보고 분류하는 모델을 제작하여 이를 해결하고자 한다.


------------------

# 2. 필요한 라이브러리
colab 이용 가능

colab에서 제공하는 torch


------------------

# 3. 데이터셋

* train / test dataset 모두 직접 촬영
  
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


------------------

# 4. 사용모델
![AIP_기말_Ensemble.png](https://github.com/hiinnnii/pill_classification_model/blob/main/AIP_%E1%84%80%E1%85%B5%E1%84%86%E1%85%A1%E1%86%AF_Ensemble.png)

* ResNet 50 + ResNet 18 + VGG 16 더한 Ensemble 모델 활용

* Ensemble 모델 사용한 이유 : 성능향상, 과적합 감소 하기 위해 Ensemble 모델을 사용




------------------

# 5. 모델 평가
모델의 accuracy를 기준으로 평가


각 class별 성능을 확인

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


