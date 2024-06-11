# 알약 분류 모델
알약 분류하는 모델 제작

약 봉지를 잃어버린 경우 약에 대한 정보를 알지 못하는 문제가 발생한다. 따라서 이를 해결하기 위해 약의 사진을 보고 분류하는 모델을 제작하여 이를 해결하고자 한다.


ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
#라이브러리
## torch 관련 라이브러리
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CyclicLR, CosineAnnealingLR, ExponentialLR
from torchsummary import summary
## 일반 라이브러리
import argparse
import numpy as np
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split





