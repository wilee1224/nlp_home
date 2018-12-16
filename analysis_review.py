import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from multiprocessing import Pool
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# %matplotlib inline 설정을 해주어야지만 노트북 안에 그래프가 디스플레이 된다.
# matplotlib inline


def load_data():
    train = pd.read_csv('data/labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
    test = pd.read_csv('data/testData.tsv', header=0, delimiter='\t', quoting=3)
    return train

def review_to_words(raw_review):
    # 1. HTML 제거
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. 영문자가 아닌 문자는 공백으로 변환
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. 소문자 변환
    words = letters_only.lower().split()
    # 4. 파이썬에서는 리스트보다 세트로 찾는게 훨씬 빠르다.
    # stopwords 를 세트로 변환한다.
    stops = set(stopwords.words('english'))
    # 5. Stopwords 불용어 제거
    meaningful_words = [w for w in words if not w in stops]
    # 6. 어간추출
    stemmer = nltk.stem.PorterStemmer()
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 7. 공백으로 구분된 문자열로 결합하여 결과를 반환
    return (' '.join(stemming_words))

def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)

def apply_by_multiprocessing(df, func, **kwargs):
    # 키워드 항목 중 workers 파라메터를 꺼냄
    workers = kwargs.pop('workers')
    # 위에서 가져온 workers 수로 프로세스 풀을 정의
    pool = Pool(processes=workers)
    # 실행할 함수와 데이터프레임을 워커의 수 만큼 나눠 작업
    result = pool.map(_apply_df, [(d, func, kwargs)
            for d in np.array_split(df, workers)])
    pool.close()
    # 작업 결과를 합쳐서 반환
    return pd.concat(list(result))

def displayWordCloud(data = None, backgroundcolor = 'white', width=800, height=600 ):
    wordcloud = WordCloud(stopwords = STOPWORDS,
                          background_color = backgroundcolor,
                         width = width, height = height).generate(data)
    plt.figure(figsize = (15 , 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()




def clean_review():
    train = load_data()

    # multiporcess 를 이용하여 review 분석
    clean_review_after_prepro = apply_by_multiprocessing(\
        train['review'], review_to_words, workers=4)
    print(clean_review_after_prepro)

    # wordcloud를 이용하여 시각화
    displayWordCloud(' '.join(clean_review_after_prepro))

    # train데이타에 num_words 칼럼을 만들고 단어 수를 기입한다
    train['num_words'] = clean_review_after_prepro.apply(lambda x : len(str(x).split()))

    # train 데이터에 num_uniq_words 칼럼을 만들고, 중복을 제거한 단어 수를 기입한다
    train['num_uniq_words'] = clean_review_after_prepro.apply(lambda x : len(set(str(x).split())))

if __name__=='__main__':
    clean_review()