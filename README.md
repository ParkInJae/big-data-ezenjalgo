*  해당 프로젝트에서 중요하다고 생각한 python 코드  <br/>
📗 test10.py <br/>
📗 test11.py <br/>
📗 test12.py <br/>
📗 test13.py <br/>
----------------------------------------

1) test10.py <br/>

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

DATA_PATH = '/python/99.project/DATA/' #데이터경로 설정
print('파일 크기: ')
for file in os.listdir(DATA_PATH):
  if 'txt' in file:
    print(file.ljust(30)+str(round(os.path.getsize(DATA_PATH+ file) / 100000,2))+'MB')
    
#트레인 파일 불러오기 <br/>
train_data = pd.read_csv(DATA_PATH + 'ratings_train.txt',header = 0, delimiter = '\t', quoting=3)
train_data.head()
print('학습데이터 전체 개수: {}'.format(len(train_data)))

train_length = train_data['document'].astype(str).apply(len)
train_length.head()

print('리뷰 길이 최댓값: {}'.format(np.max(train_length)))
print('리뷰 길이 최솟값: {}'.format(np.min(train_length)))
print('리뷰 길이 평균값: {:.2f}'.format(np.mean(train_length)))
print('리뷰 길이 표준편차: {:.2f}'.format(np.std(train_length)))
print('리뷰 길이 중간값: {}'.format(np.median(train_length)))
print('리뷰 길이 제1사분위: {}'.format(np.percentile(train_length,25)))
print('리뷰 길이 제3사분위: {}'.format(np.percentile(train_length,75)))

# 문자열 아닌 데이터 모두 제거 <br/>
train_review = [review for review in train_data['document'] if type(review) is str]
train_review

wordcloud = WordCloud('HANYGO230.ttf').generate(' '.join(train_review))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#긍정 1, 부정 0
print('긍정 리뷰 갯수: {}'.format(train_data['label'].value_counts()[1]))
print('부정 리뷰 갯수: {}'.format(train_data['label'].value_counts()[0]))

--------------------------------------------------------------------------------------------------------------------------------
2) test11.py <br/>

import numpy as np
import pandas as pd
import os
import re
import json
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# 데이터 경로 설정 <br/>
DATA_PATH = '/python/99.project/DATA/'

# 파일 경로가 올바른지 확인 <br/>
file_path = DATA_PATH + 'ratings_train.txt'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"지정된 경로에 파일이 존재하지 않습니다: {file_path}")

# 데이터 로딩 <br/>
train_data = pd.read_csv(file_path, header=0, delimiter='\t', quoting=3)

# 데이터의 첫 몇 행을 출력하여 확인 <br/>
print(train_data.head())

# 전처리 함수 정의 <br/>
def preprocessing(review, okt, remove_stopwords=False, stop_words=[]):
    review_text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]', '', review)
    word_review = okt.morphs(review_text, stem=True)
    if remove_stopwords:
        word_review = [token for token in word_review if not token in stop_words]
    return word_review

# 전처리 테스트 <br/>
okt = Okt()
stop_words = ['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한']

# 리뷰 중 첫 번째를 전처리해서 확인 <br/>
test_review = train_data['document'][0]
if type(test_review) == str:
    processed_review = preprocessing(test_review, okt, remove_stopwords=True, stop_words=stop_words)
    print(f"Original: {test_review}")
    print(f"Processed: {processed_review}")

# 전체 텍스트 전처리 <br/>
clean_train_review = []
for review in train_data['document']:
    if type(review) == str:
        clean_train_review.append(preprocessing(review, okt, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_train_review.append([])

# 전처리 결과의 첫 몇 개를 출력하여 확인 <br/>
print(clean_train_review[:4])

#테스트 리뷰도 동일하게 전처리 <br/>
test_data = pd.read_csv(DATA_PATH + 'ratings_test.txt', header = 0, delimiter='\t', quoting=3)

clean_test_review = []
for review in test_data['document']:
  if type(review) == str:
    clean_test_review.append(preprocessing(review, okt, remove_stopwords=True, stop_words=stop_words))
  else:
    clean_test_review.append([])
    
# 인덱스 벡터 변환 후 일정 길이 넘어가거나 모자라는 리뷰 패딩처리 <br/>
tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_train_review)
train_sequences = tokenizer.texts_to_sequences(clean_train_review)
test_sequences = tokenizer.texts_to_sequences(clean_test_review)

word_vocab = tokenizer.word_index #단어사전형태
MAX_SEQUENCE_LENGTH = 8 #문장 최대 길이

#학습 데이터 <br/>
train_inputs = pad_sequences(train_sequences,maxlen=MAX_SEQUENCE_LENGTH,padding='post')

#학습 데이터 라벨 벡터화 <br/>
train_labels = np.array(train_data['label'])

#평가 데이터 <br/>
test_inputs = pad_sequences(test_sequences,maxlen=MAX_SEQUENCE_LENGTH,padding='post')
#평가 데이터 라벨 벡터화 <br/>
test_labels = np.array(test_data['label'])

DEFAULT_PATH  = '/python/99.project/' # 경로지정
DATA_PATH = 'CLEAN_DATA/' #.npy파일 저장 경로지정
TRAIN_INPUT_DATA = 'nsmc_train_input.npy'
TRAIN_LABEL_DATA = 'nsmc_train_label.npy'
TEST_INPUT_DATA = 'nsmc_test_input.npy'
TEST_LABEL_DATA = 'nsmc_test_label.npy'
DATA_CONFIGS = 'data_configs.json'

data_configs={}
data_configs['vocab'] = word_vocab
data_configs['vocab_size'] = len(word_vocab) + 1

#전처리한 데이터들 파일로저장 <br/>
if not os.path.exists(DEFAULT_PATH + DATA_PATH):
  os.makedirs(DEFAULT_PATH+DATA_PATH)

#전처리 학습데이터 넘파이로 저장 <br/>
np.save(open(DEFAULT_PATH+DATA_PATH+TRAIN_INPUT_DATA,'wb'),train_inputs)
np.save(open(DEFAULT_PATH+DATA_PATH+TRAIN_LABEL_DATA,'wb'),train_labels) <br/>
#전처리 테스트데이터 넘파이로 저장 <br/>
np.save(open(DEFAULT_PATH+DATA_PATH+TEST_INPUT_DATA,'wb'),test_inputs)
np.save(open(DEFAULT_PATH+DATA_PATH+TEST_LABEL_DATA,'wb'),test_labels)
<br/>
#데이터 사전 json으로 저장 <br/>
json.dump(data_configs,open(DEFAULT_PATH + DATA_PATH + DATA_CONFIGS,'w'),ensure_ascii=False)

--------------------------------------------------------------------------------------------------------------------------------
3) test12.py <br/>

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm

# 전처리 데이터 불러오기 <br/>
DATA_PATH = '/python/99.project/CLEAN_DATA/'
DATA_OUT = '/python/99.project/DATA_OUT/'
INPUT_TRAIN_DATA = 'nsmc_train_input.npy'
LABEL_TRAIN_DATA = 'nsmc_train_label.npy'
DATA_CONFIGS = 'data_configs.json'

train_input = np.load(open(os.path.join(DATA_PATH, INPUT_TRAIN_DATA), 'rb'))
train_input = pad_sequences(train_input, maxlen=train_input.shape[1])
train_label = np.load(open(os.path.join(DATA_PATH, LABEL_TRAIN_DATA), 'rb'))
prepro_configs = json.load(open(os.path.join(DATA_PATH, DATA_CONFIGS), 'r'))

model_name = 'cnn_classifier_kr'
BATCH_SIZE = 512
NUM_EPOCHS = 10
VALID_SPLIT = 0.1
MAX_LEN = train_input.shape[1]

kargs = {
    'model_name': model_name,
    'vocab_size': prepro_configs['vocab_size'],
    'embbeding_size': 128,
    'num_filters': 100,
    'dropout_rate': 0.5,
    'hidden_dimension': 250,
    'output_dimension': 1
}

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

model = CNNClassifier(**kargs)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])
<br/>
# 검증 정확도를 통한 EarlyStopping 기능 및 모델 저장 방식 지정 <br/>
earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=2)
checkpoint_path = os.path.join(DATA_OUT, model_name, 'weights.h5')
checkpoint_dir = os.path.dirname(checkpoint_path)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"{checkpoint_dir} -- Folder created\n")
else:
    print(f"{checkpoint_dir} -- Folder already exists\n")

cp_callback = ModelCheckpoint(
    checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True,
    save_weights_only=True
)

history = model.fit(train_input, train_label, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
                    validation_split=VALID_SPLIT, callbacks=[earlystop_callback, cp_callback])
<br/>
# 모델 저장하기 <br/>
model_save_path = os.path.join('/python/99.project/MODEL', 'model')
if not os.path.exists(os.path.dirname(model_save_path)):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
<br/>
# 테스트 데이터 로드 및 평가 <br/>
INPUT_TEST_DATA = 'nsmc_test_input.npy'
LABEL_TEST_DATA = 'nsmc_test_label.npy'
SAVE_FILE_NM = 'weights.h5'

test_input = np.load(open(os.path.join(DATA_PATH, INPUT_TEST_DATA), 'rb'))
test_input = pad_sequences(test_input, maxlen=test_input.shape[1])
test_label_data = np.load(open(os.path.join(DATA_PATH, LABEL_TEST_DATA), 'rb'))

model.load_weights(checkpoint_path)
model.evaluate(test_input, test_label_data)


--------------------------------------------------------------------------------------------------------------------------------
4) test13.py <br/>

import numpy as np
import pandas as pd
import re
import json
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras import layers
<br/>
# 모델 클래스 정의 <br/>
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
<br/>
# Okt와 Tokenizer 초기화 <br/>
okt = Okt()
tokenizer = Tokenizer()
<br/>
# 데이터 설정 불러오기 <br/>
DATA_CONFIGS = 'data_configs.json'
prepro_configs = json.load(open('/python/99.project/CLEAN_DATA/' + DATA_CONFIGS, 'r'))
word_vocab = prepro_configs['vocab']
<br/>
# Tokenizer에 단어 사전 적용 <br/>
tokenizer.fit_on_texts(list(word_vocab.keys()))
<br/>
# 모델 인자 설정 <br/>
kargs = {
    'model_name': 'cnn_classifier_kr',
    'vocab_size': len(word_vocab) + 1,  # vocab_size를 단어 사전 길이로 설정
    'embbeding_size': 128,
    'num_filters': 100,
    'dropout_rate': 0.5,
    'hidden_dimension': 250,
    'output_dimension': 1
}
<br/>
# 모델 초기화 및 컴파일 <br/>
model = CNNClassifier(**kargs)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy')])
<br/>
# dummy input으로 모델 호출 <br/>
dummy_input = np.zeros((1, 8), dtype=np.int32)
model(dummy_input)
<br/>
# 모델 가중치 로드 <br/>
model.load_weights('/python/99.project/DATA_OUT/cnn_classifier_kr/weights.h5')
<br/>
# 문장 분석 함수 <br/>
def analyze_sentence(sentence):
    MAX_LENGTH = 8  # 문장 최대 길이
    stopwords = ['은', '는', '이', '가', '하', '아', '것', '들', '의', '있', '되', '수', '보', '주', '등', '한']  # 불용어
<br/>
    # 문장 전처리 <br/>
    sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣\s]', '', sentence)
    sentence = okt.morphs(sentence, stem=True)
    sentence = [word for word in sentence if not word in stopwords]
<br/>
    # 문장을 시퀀스로 변환 및 패딩 <br/>
    vector = tokenizer.texts_to_sequences([sentence])
    pad_new = pad_sequences(vector, maxlen=MAX_LENGTH)
<br/>
    # 예측 수행 <br/>
    predictions = model.predict(pad_new)
    predictions = float(predictions.squeeze())
<br/>
    # 결과 출력 <br/>
    if predictions > 0.5:
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(predictions * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - predictions) * 100))
<br/>
# 사용자 입력 받아서 감성 분석 <br/>
sentence = input('감성분석할 문장을 입력해 주세요: ')
analyze_sentence(sentence)
<br/>
