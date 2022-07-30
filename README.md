# 기린그림_폰트팀_NerualFont
사용자 손글씨 폰트 생성 (Neural font 변형) - BOAZ 17기 분석 컨퍼런스 **기린그림**팀 폰트부분

## [Presentation Document](https://www.slideshare.net/BOAZbigdata/16-boaz-252346476)
## [Presentation Video](https://youtu.be/DxA-Cpu5qLQ)

## Our team
- 김송성
- 박종은
- 이보림 (Bolim Lee) / github: @bo-lim / (School of Computer Science & Engineering, Chung-Ang Univ.)

## Introduction
기존의 [neural font](https://github.com/periannath/neural-fonts)를 업그레이드 하였다.
'''
전처리 + 후처리를 통해 성능을 향상,

399자에서 288자로 줄여도 비슷한 결과물이 나올 수 있도록 수정,

Python3 환경에서 학습될 수 있도록 바꾸었다. (original neural font는 python2, tensorflow 1버전을 사용하여 Colab에서 학습시키기에 어려움이 있다.)
'''


## How to use
1. Download base [checkpoint](https://drive.google.com/file/d/1uLGAyY7zXUi2BHuc90-ILw-IgawVcsZ8/view)

2. Follow neuralfont.ipynb