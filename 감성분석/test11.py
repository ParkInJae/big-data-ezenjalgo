import numpy as np
import pandas as pd
import os
import re
import json
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# 데이터 경로 설정
DATA_PATH = '/python/99.project/DATA/'

# 파일 경로가 올바른지 확인
file_path = DATA_PATH + 'ratings_train.txt'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"지정된 경로에 파일이 존재하지 않습니다: {file_path}")

# 데이터 로딩
train_data = pd.read_csv(file_path, header=0, delimiter='\t', quoting=3)

# 데이터의 첫 몇 행을 출력하여 확인
print(train_data.head())

# 전처리 함수 정의
def preprocessing(review, okt, remove_stopwords=False, stop_words=[]):
    review_text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]', '', review)
    word_review = okt.morphs(review_text, stem=True)
    if remove_stopwords:
        word_review = [token for token in word_review if not token in stop_words]
    return word_review

# 전처리 테스트
okt = Okt()
stop_words = ['은','는','이','가','하','아','것','들','의','있','되','수','보','주','등','한']

# 리뷰 중 첫 번째를 전처리해서 확인
test_review = train_data['document'][0]
if type(test_review) == str:
    processed_review = preprocessing(test_review, okt, remove_stopwords=True, stop_words=stop_words)
    print(f"Original: {test_review}")
    print(f"Processed: {processed_review}")

# 전체 텍스트 전처리
clean_train_review = []
for review in train_data['document']:
    if type(review) == str:
        clean_train_review.append(preprocessing(review, okt, remove_stopwords=True, stop_words=stop_words))
    else:
        clean_train_review.append([])

# 전처리 결과의 첫 몇 개를 출력하여 확인
print(clean_train_review[:4])

#테스트 리뷰도 동일하게 전처리
test_data = pd.read_csv(DATA_PATH + 'ratings_test.txt', header = 0, delimiter='\t', quoting=3)

clean_test_review = []
for review in test_data['document']:
  if type(review) == str:
    clean_test_review.append(preprocessing(review, okt, remove_stopwords=True, stop_words=stop_words))
  else:
    clean_test_review.append([])
    
# 인덱스 벡터 변환 후 일정 길이 넘어가거나 모자라는 리뷰 패딩처리
tokenizer = Tokenizer()
tokenizer.fit_on_texts(clean_train_review)
train_sequences = tokenizer.texts_to_sequences(clean_train_review)
test_sequences = tokenizer.texts_to_sequences(clean_test_review)

word_vocab = tokenizer.word_index #단어사전형태
MAX_SEQUENCE_LENGTH = 8 #문장 최대 길이

#학습 데이터
train_inputs = pad_sequences(train_sequences,maxlen=MAX_SEQUENCE_LENGTH,padding='post')

#학습 데이터 라벨 벡터화
train_labels = np.array(train_data['label'])

#평가 데이터 
test_inputs = pad_sequences(test_sequences,maxlen=MAX_SEQUENCE_LENGTH,padding='post')
#평가 데이터 라벨 벡터화
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

#전처리한 데이터들 파일로저장
if not os.path.exists(DEFAULT_PATH + DATA_PATH):
  os.makedirs(DEFAULT_PATH+DATA_PATH)

#전처리 학습데이터 넘파이로 저장
np.save(open(DEFAULT_PATH+DATA_PATH+TRAIN_INPUT_DATA,'wb'),train_inputs)
np.save(open(DEFAULT_PATH+DATA_PATH+TRAIN_LABEL_DATA,'wb'),train_labels)
#전처리 테스트데이터 넘파이로 저장
np.save(open(DEFAULT_PATH+DATA_PATH+TEST_INPUT_DATA,'wb'),test_inputs)
np.save(open(DEFAULT_PATH+DATA_PATH+TEST_LABEL_DATA,'wb'),test_labels)

#데이터 사전 json으로 저장
json.dump(data_configs,open(DEFAULT_PATH + DATA_PATH + DATA_CONFIGS,'w'),ensure_ascii=False)