from konlpy.utils import pprint
from konlpy.tag import Komoran
from konlpy.tag import Kkma


def analysisHan():
    komoran = Komoran()
    kkma = Kkma()

    pprint(komoran.morphs(u'로그인창에 ID를 입력하고 PW창에 PW를 입력하고 로그인 버튼을 클릭한다.'))
    pprint(kkma.pos(u'로그인창에 ID를 입력하고 PW창에 PW를 입력하고 로그인 버튼을 클릭한다.'))

if __name__=="__main__":
    analysisHan()