# 맞춤형 음성 인식 시스템

이 프로젝트는 특정 문구를 인식하는 맞춤형 음성 인식 시스템입니다.

## 프로젝트 구조

```plaintext
progmatism-project/
│
├── data/                       # 녹음된 오디오 데이터를 저장하는 디렉토리
│   ├── 안녕하세요/               # '안녕하세요' 녹음을 위한 서브디렉토리
│   ├── 감사합니다/               # '감사합니다' 녹음을 위한 서브디렉토리
│   ├── 네/                      # '네' 녹음을 위한 서브디렉토리
│   └── 아니요/                  # '아니요' 녹음을 위한 서브디렉토리
│
├── models/                     # 학습된 모델을 저장하는 디렉토리
│   ├── personalized_speech_recognition_model.keras
│   └── label_encoder.pkl
│
├── .gitignore                  # Git에서 제외할 파일들을 지정하는 파일
├── app.py                      # 예측 요청을 처리하는 Flask 서버 파일
├── client.py                   # 오디오를 녹음하고 예측을 요청하는 클라이언트 파일
├── data_preprocessing.py       # 오디오 데이터를 전처리하는 스크립트
├── extract_features.py         # 오디오 파일에서 MFCC 특징을 추출하는 스크립트
├── record_audio_batch.py       # 오디오 파일을 일괄 녹음하는 스크립트
├── record_audio.py             # 단일 오디오 파일을 녹음하는 스크립트
├── retrain_model.py            # 새로운 데이터로 모델을 재학습하는 스크립트
├── speech_model.py             # 음성 인식 모델을 정의하는 스크립트
└── test_predict.py             # 예측을 테스트하는 스크립트
```

## 사용 방법

1. **오디오 데이터 녹음**:
   "안녕하세요", "감사합니다", "네", "아니요" 문구에 대한 오디오 데이터를 녹음합니다.
   ```bash
   python record_audio_batch.py

2. **모델 재학습**:
   "녹음한 새로운 데이터로 모델을 재학습합니다..
   ```bash
   python retrain_model.py

3. **Flask 서버 실행:**:
   "예측 요청을 처리하는 Flask 서버를 시작합니다...
   ```bash
   python app.py

4. **예측 요청:**:
   "오디오 샘플을 녹음하고 서버에 예측을 요청합니다....
   ```bash
   python client.py


## 요구 사항
- Python 3.x
- 필요한 Python 패키지는 requirements.txt에 나열되어 있습니다.

## 설치 방법
1. **리포지토리를 클론합니다**:
git clone <your-repository-url>
cd your_project

2. **필요한 패키지를 설치합니다**:
pip install -r requirements.txt