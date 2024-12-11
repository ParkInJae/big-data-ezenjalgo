import numpy as np
import pandas as pd
import re
import json
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras import layers

# 모델 클래스 정의
class CNNClassifier(tf.keras.Model):
    def __init__(self, **kargs):
        super(CNNClassifier, self).__init__(name=kargs['model_name'])
        self.embedding = layers.Embedding(input_dim=kargs['vocab_size'], output_dim=kargs['embbeding_size'])
        self.conv_list = [layers.Conv1D(filters=kargs['num_filters'], kernel_size=kernel_size, padding='valid', 
                                        activation=tf.keras.activations.relu,
                                        kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3)) 
                          for kernel_size in [3, 4, 5]]
        self.pooling = layers.GlobalMaxPooling1D()
        self.dropout = layers.Dropout(kargs['dropout_rate'])
        self.fc1 = layers.Dense(units=kargs['hidden_dimension'], activation=tf.keras.activations.relu,
                                kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.))
        self.fc2 = layers.Dense(units=kargs['output_dimension'], activation=tf.keras.activations.sigmoid,
                                kernel_constraint=tf.keras.constraints.MaxNorm(max_value=3.))

    def call(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = tf.concat([self.pooling(conv(x)) for conv in self.conv_list], axis=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Okt와 Tokenizer 초기화
okt = Okt()
tokenizer = Tokenizer()

# 데이터 설정 불러오기
DATA_CONFIGS = 'data_configs.json'
prepro_configs = json.load(open('/python/99.project/CLEAN_DATA/' + DATA_CONFIGS, 'r'))
word_vocab = prepro_configs['vocab']

# Tokenizer에 단어 사전 적용
tokenizer.fit_on_texts(list(word_vocab.keys()))

# 모델 인자 설정
kargs = {
    'model_name': 'cnn_classifier_kr',
    'vocab_size': len(word_vocab) + 1,  # vocab_size를 단어 사전 길이로 설정
    'embbeding_size': 128,
    'num_filters': 100,
    'dropout_rate': 0.5,
    'hidden_dimension': 250,
    'output_dimension': 1
}

# 모델 초기화 및 컴파일
model = CNNClassifier(**kargs)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])

# dummy input으로 모델 호출
dummy_input = np.zeros((1, 8), dtype=np.int32)
model(dummy_input)

# 모델 가중치 로드
model.load_weights('/python/99.project/DATA_OUT/cnn_classifier_kr/weights.h5')

# 문장 분석 함수
def analyze_sentence(sentence):
    MAX_LENGTH = 8  # 문장 최대 길이
    stopwords = ['은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한']  # 불용어

    # 문장 전처리
    sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]', '', sentence)
    sentence = okt.morphs(sentence, stem=True)
    sentence = [word for word in sentence if not word in stopwords]

    # 문장을 시퀀스로 변환 및 패딩
    vector = tokenizer.texts_to_sequences([sentence])
    pad_new = pad_sequences(vector, maxlen=MAX_LENGTH)

    # 예측 수행
    predictions = model.predict(pad_new)
    predictions = float(predictions.squeeze())

    # 결과 출력
    if predictions > 0.5:
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(predictions * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - predictions) * 100))

# 사용자 입력 받아서 감성 분석
sentence = input('감성분석할 문장을 입력해 주세요: ')
analyze_sentence(sentence)