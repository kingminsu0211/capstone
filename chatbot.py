# -*- coding: utf-8 -*-
"""chatbot.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16qRNCcYvhHVFF2tr8oa2yAwG8XI60EAw

BERT
"""

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import urllib.request
from sentence_transformers import SentenceTransformer

#챗봇 QA 엑셀 파일 불러오기
train_data = pd.read_excel('/content/drive/MyDrive/대처방법_챗봇.xlsx', sheet_name = '챗봇 QA')
train_data.head()

#코사인 유사도
def cos_sim(A, B):
  return dot(A, B)/(norm(A)*norm(B))

help_text = "문의하신 사항을 제대로 이해하지 못했습니다. 다시 문의해주십시오."

"""한국어 버전의 버트 https://huggingface.co/jhgan/ko-sroberta-multitask"""

model_finetune_5 = SentenceTransformer('chatbot_5') # 비교해보니 이게 제일 성능이 괜찮음

train_data['embedding_f5'] = train_data.apply(lambda row: model_finetune_5.encode(row.text), axis = 1)

#원래 모델은 전체적으로 유사도가 높게 나오고 파인튜닝 모델은 전체적으로 유사도가 낮게 나와서 각각 0.7, 0.6으로 설정
#입력 질문과 가장 유사한 질문 데이터를 찾아서 해당 질문데이터의 대답데이터 반환

def return_answer_f5(question): #파인튜닝
    embedding = model_finetune_5.encode(question)
    train_data['score_f5'] = train_data.apply(lambda x: cos_sim(x['embedding_f5'], embedding), axis=1)
    print("유사도:", train_data.loc[train_data['score_f5'].idxmax()]['score_f5'])
    if train_data.loc[train_data['score_f5'].idxmax()]['score_f5'] < 0.65:
      return help_text
    print("매칭된 질문은: ", train_data.loc[train_data['score_f5'].idxmax()]['text'])
    return train_data.loc[train_data['score_f5'].idxmax()]['answer']


return_answer_f5('피해입은 금액을 보상받을 수 없나요...?')

#테스트용 문장 20개 - 엑셀파일에 데이터 추가를 위해 몇개 가져다 써서 다시 작성해야 함
test = [
    "사기 당했는데 보상 같은 거 받을 방법 없나요?",
    "돈을 돌려받을 수 있나요?",
    "피해구제신청하면 돈을 받을 수 있는 건가요?",
    "보이스피싱 당하면 어떤 거 해야 돼?",
    "보이스피싱대처방법알려줘",
    "이상한 문자 같은 것도 경찰에 신고해도 되나요?",
    "개인 전화로 검찰이라고 하면 다 보이스피싱인거죠?",
    "신분증 사진을 모르고 보냈는데 큰일난건가요?",
    "개인정보를 알려줘버렸는데 어떡하죠?",
    "목소리가 자식이었는데 보이스피싱일 수 있나요?",
    "보이스피싱은 누가하나요?",
    "사기 대처법 알려줘!",
    "나 사기 당했는데 어떻게 해?",
    "피해환급금이란 게 뭐야?",
    "피해구제신청하면 나라에서 도와주는건가요?",
    "계좌 지급정지가 무슨 뜻이죠?",
    "제 명의가 도용된 거 같은데 어떻게 확인해요?",
    "채권소멸절차란 게 뭐야?",
    "피해구제신청했는데 돈은 언제 주나요?",
    "보이스피싱인지 모르고 개인정보 보내버렸는데 어떡하죠?"
]

# 문장 20개 테스트 #미세조정 5 모델이 가장 정확함
for n, test_sen in enumerate(test):
    print(n, "번")
    print("테스트 문장:", test_sen)
    print("----")
    print("<<미세조정 5>>")
    print("답변: ", return_answer_f5(test_sen))
    print("**************")

#테스트용 관련없는 문장 20개
test_not = [
    "안녕하세요, 지금 뭐해요?",
    "돈 많이 벌고 싶어요",
    "이상한 전화는 바로 끊을게요",
    "보이스피싱은 나쁜 일입니다",
    "이 앱의 사용법을 알려주세요",
    "개인 정보에는 어떤 것이 있나요?",
    "나쁜 사람은 경찰에 신고하면 되나요?",
    "휴대폰에 게임을 다운로드 받았어요",
    "언니한테 돈 빌려달라는 문자를 받았어요",
    "엄마가 제가 보낸 문자를 보이스피싱인 줄 알았대요",
    "맛있는 가게 알려주세요",
    "맛있다고 해서 갔는데 맛없을 때 대처방법 알려주세요",
    "중고마켓 사기 당했어요ㅜㅜ 어떡하죠",
    "가해자는 피해자에게 적절한 보상을 해줘야 해요",
    "제 계좌에는 지금 1억이 없어요",
    "누가 누구인지 구별할 수 있나요?",
    "제 명의로 자동차를 샀습니다.",
    "오늘 신문에서 보이스피싱에 대한 기사를 봤어요",
    "아직까진 제 주변에 보이스피싱 피해를 입은 사람이 없네요",
    "방금 피해환급금이라는 단어를 찾아봤어요"
]

# 관련없는 문장 20개 테스트
for n, test_sen in enumerate(test_not):
    print(n, "번")
    print("테스트 문장:", test_sen)
    print("----")
    print("<<미세조정5>>")
    print("답변: ", return_answer_f5(test_sen))
    print("**************")