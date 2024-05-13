from rest_framework import generics
from .serializers import *
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import re
from konlpy.tag import Okt
from glove import Corpus, Glove
from numpy import dot
from numpy.linalg import norm
import pickle
import json
import requests
import numpy as np
import time


# 여기에는 JWT 토큰을 설정합니다.
YOUR_JWT_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYSI6dHJ1ZSwiZXhwIjoxNzE1NjIxNjg0LCJmdWUiOmZhbHNlLCJoYmkiOmZhbHNlLCJpYXQiOjE3MTU2MDAwODQsImp0aSI6InJMVVdOcDhjRWVlcWp4R1BvRGFWIiwicGxhbiI6ImJhc2ljIiwic2NvcGUiOiJzcGVlY2giLCJzdWIiOiJEd3hCNVk3azIyMHFUVjllUXBkbiIsInVjIjpmYWxzZSwidiI6MX0.LdCpl4BHMRQwFXpmuZ1u58QqRhJEXsFBLcSW3U3Z_kY'


glove = Glove.load('glove.model2')

okt = Okt()


def audio_to_text(audio_file, config={}):
    # WAV 파일을 API에 전송하여 전사 작업 시작
    resp = requests.post(
        'https://openapi.vito.ai/v1/transcribe',
        headers={'Authorization': 'bearer ' + YOUR_JWT_TOKEN},
        data={'config': json.dumps(config)},
        files={'file': audio_file}
    )
    resp.raise_for_status()
    response_data = resp.json()
    transcribe_id = response_data['id']
    print(f"전사 작업이 시작되었습니다. TRANSCRIBE_ID: {transcribe_id}")

    # 전사 작업이 완료될 때까지 대기
    while True:
        resp = requests.get(
            f"https://openapi.vito.ai/v1/transcribe/{transcribe_id}",
            headers={'Authorization': 'bearer ' + YOUR_JWT_TOKEN}
        )
        resp.raise_for_status()
        response_data = resp.json()
        status = response_data['status']

        if status == 'completed':
            break
        elif status == 'failed':
            print("전사 작업이 실패했습니다.")
            return None

        # 일정한 간격으로 폴링
        time.sleep(5)

    # 전사 결과 가져오기
    resp = requests.get(
        f"https://openapi.vito.ai/v1/transcribe/{transcribe_id}",
        headers={'Authorization': 'bearer ' + YOUR_JWT_TOKEN}
    )
    resp.raise_for_status()
    response_data = resp.json()
    results = response_data['results']

    # 전사 결과에서 텍스트 부분만 추출하여 리스트로 저장
    call_details = [utterance['msg'] for utterance in results['utterances']]
    print(call_details)
    # 리스트의 각 요소를 하나의 문자열로 결합하여 반환
    return ' '.join(call_details)


# 전처리 함수
def preprocessing(input_text):
    removed_data = []
    s = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", input_text)
    if s != '':
        removed_data.append(s)

    tokenized_data = []
    for sentence in removed_data:
        tokenized_sentence = okt.morphs(sentence, stem=True)
        tokenized_data.append(tokenized_sentence)
    return tokenized_data

# 토큰화된 단어 벡터화 함수
def to_vec(tokenized_data):
    vector_list = []
    for sentence_tokens in tokenized_data:
        for token in sentence_tokens:
            if token in glove.dictionary.keys():
                vector_list.append([token, glove.word_vectors[glove.dictionary[token]]])
    return vector_list

# 입력 문서 평균 벡터 구하기 함수
def get_document_vec(word_vec_list):
    input_vec = np.zeros_like(word_vec_list[0][1])
    for word_vec in word_vec_list:
        input_vec += word_vec[1]
    input_vec /= len(word_vec_list)
    return input_vec

# 코사인 유사도 함수
def cos_sim(A, B):
    similarity = dot(A, B) / (norm(A) * norm(B))
    return similarity

#코사인 유사도 구해서 높은 순서대로 단어와 유사도 출력 -> 단일 문서와 비교 버전
def s_cos_word_doc(word_vec_list, doc_vec):
  answer = []
  for word_vec in word_vec_list:
    answer.append([word_vec[0], cos_sim(doc_vec, word_vec[1])])
  answer = sorted(answer, key = lambda x: x[1], reverse = True)
  return answer
#10위까지 겹치는 단어 없이 (단어, 유사도) 출력

def get_sim_top10(answer, cnt, compare_doc_cnt):
  answerList = []
  num = 0
  i = -1
  while num < 10:
    i += 1
    if i >= cnt: break;
    answer[i][1] /= compare_doc_cnt
    if answer[i] in answerList:
      continue
    answerList.append(answer[i])
    num += 1;
  return answerList

#코사인 유사도 구해서 높은 순서대로 단어와 유사도 출력 -> 복수 문서와 비교 버전
def m_cos_word_doc(word_vec_list, doc_vec_list):
  answer = []
  is_first = 0
  for doc_vec in doc_vec_list:
    for num, word_vec in enumerate(word_vec_list):
      if is_first == 0:
        answer.append([word_vec[0], cos_sim(doc_vec, word_vec[1])])
      else:
        answer[num][1] += cos_sim(doc_vec, word_vec[1])
    is_first = 1
  answer = sorted(answer, key = lambda x: x[1], reverse = True)

  return answer

# 모델에 해당 기능을 추가하는 함수
def diagnose_phishing(call_details):
    tokenized_data = preprocessing(call_details)
    word_vec_list = to_vec(tokenized_data)
    input_vec = get_document_vec(word_vec_list)
    vec_cnt = len(word_vec_list)
    words_sim_top10 = []
    # 비교 문서 벡터 리스트 가져오기
    with open("phishing_vec_list2.pkl","rb") as f:
        compare_vec_list = pickle.load(f)

    # 코사인 유사도 평균 구하기
    mean_sim = 0
    if input_vec is not None:
        for doc_vec in compare_vec_list:
            doc_sim = cos_sim(input_vec, doc_vec)
            mean_sim += doc_sim

    mean_sim /= len(compare_vec_list)

    # 보이스피싱 의심 여부 초기화
    is_phishing = 0

    # 코사인 유사도 평균이 0.7 이상이면 보이스피싱 의심 여부를 1로 설정
    if mean_sim >= 0.7:
        is_phishing = 1
        words_sim = m_cos_word_doc(word_vec_list, compare_vec_list)
        words_sim_top10 = get_sim_top10(words_sim, vec_cnt, len(compare_vec_list))
    return is_phishing, mean_sim, words_sim_top10


# 통화 진단하기 API에 대한 Swagger 문서
@swagger_auto_schema(
    method='post',
    operation_description="통화 진단하기 API",
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        required=['diagnosis_type'],
        properties={
            'diagnosis_type': openapi.Schema(type=openapi.TYPE_STRING, description='진단 타입'),
            'call_details': openapi.Schema(type=openapi.TYPE_STRING, description='통화 내용'),
            'audio_file': openapi.Schema(type=openapi.TYPE_FILE, description='통화 녹음 파일'),
            'suspicion_percentage': openapi.Schema(type=openapi.TYPE_NUMBER, description='의심도(퍼센트)'),
        },
    ),
    responses={200: '성공적으로 진단됨', 400: '잘못된 요청'},
)
# 통화 진단하기
@api_view(['POST'])
def diagnose_voice(request):
    serializer = DiagnosisSerializer(data=request.data)
    if serializer.is_valid():
        diagnosis_type = serializer.validated_data.get('diagnosis_type')
        call_details = serializer.validated_data.get('call_details')
        audio_file = serializer.validated_data.get('audio_file')
        suspicion_percentage = serializer.validated_data.get('suspicion_percentage')

        if diagnosis_type == '직접 통화 내용 입력하기':
            # '직접 통화 내용 입력하기'일 경우 audio_file 필드는 무시
            serializer.validated_data.pop('call_details', None)
            if call_details:
                suspicion, mean_sim,words_sim_top10 = diagnose_phishing(call_details)
                if suspicion is not None:
                    suspicion_percentage = round(mean_sim * 100)
                    # 평균 유사도가 0.7 이상인 경우 의심 단어 반환
                    if suspicion >= 0.7:
                        response_data = {'suspicion_percentage': f'{suspicion_percentage}%','보이스피싱 의심 여부': '보이스피싱이 의심됨'}
                        response_data['보이스피싱 의심 상위 10단어'] = [word[0] for word in words_sim_top10]
                    return Response(response_data, status=status.HTTP_200_OK)
                else:
                    return Response({'message': '진단 결과를 얻을 수 없습니다.'}, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({'message': '통화 내용이 제공되지 않았습니다.'}, status=status.HTTP_400_BAD_REQUEST)

        # 통화 녹음본으로 입력하기
        elif diagnosis_type == '통화 녹음본으로 입력하기':
            # '통화 녹음본으로 입력하기'일 경우 call_details 필드는 무시
            serializer.validated_data.pop('audio_file', None)
            # 오디오 파일을 텍스트로 변환
            if audio_file:
                call_details = audio_to_text(audio_file)
            else:
                return Response({'message': '오디오 파일이 제공되지 않았습니다.'}, status=status.HTTP_400_BAD_REQUEST)

            if call_details:
                suspicion, mean_sim, words_sim_top10 = diagnose_phishing(call_details)
                if suspicion is not None:
                    suspicion_percentage = round(mean_sim * 100)
                    # 평균 유사도가 0.7 이상인 경우 의심 단어 반환
                    if suspicion >= 0.7:
                        response_data = {'suspicion_percentage': f'{suspicion_percentage}%',
                                         '보이스피싱 의심 여부': '보이스피싱이 의심됨'}
                        response_data['보이스피싱 의심 상위 10단어'] = [word[0] for word in words_sim_top10]
                    return Response(response_data, status=status.HTTP_200_OK)
                else:
                    return Response({'message': '진단 결과를 얻을 수 없습니다.'}, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({'message': '오디오 파일이나 URL로부터 텍스트를 가져올 수 없습니다.'}, status=status.HTTP_400_BAD_REQUEST)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

