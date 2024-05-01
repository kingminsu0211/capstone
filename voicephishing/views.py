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
import urllib.parse
import time


class DiagnosisListView(generics.ListCreateAPIView):
    queryset = Diagnosis.objects.all()
    serializer_class = DiagnosisSerializer

# # 여기에는 JWT 토큰을 설정합니다.
# YOUR_JWT_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJkYSI6dHJ1ZSwiZXhwIjoxNzEzMjY5NTAxLCJmdWUiOmZhbHNlLCJoYmkiOmZhbHNlLCJpYXQiOjE3MTMyNDc5MDEsImp0aSI6InlhaUt2MDJKVWxrX1RveWQ1V0REIiwicGxhbiI6ImJhc2ljIiwic2NvcGUiOiJzcGVlY2giLCJzdWIiOiJEd3hCNVk3azIyMHFUVjllUXBkbiIsInVjIjpmYWxzZSwidiI6MX0.47gPWgNqEIenwlgjDsTXKOZhXNCT2bI4SnfErklol90'
#
# # Whisper 모델 로드
# model_med = whisper.load_model("medium")
#
# glove = Glove.load('glove.model')
#
# okt = Okt()
#
#
# def audio_to_text(audio_file, config={}):
#     # WAV 파일을 API에 전송하여 전사 작업 시작
#     resp = requests.post(
#         'https://openapi.vito.ai/v1/transcribe',
#         headers={'Authorization': 'bearer ' + YOUR_JWT_TOKEN},
#         data={'config': json.dumps(config)},
#         files={'file': audio_file}
#     )
#     resp.raise_for_status()
#     response_data = resp.json()
#     transcribe_id = response_data['id']
#     print(f"전사 작업이 시작되었습니다. TRANSCRIBE_ID: {transcribe_id}")
#
#     # 전사 작업이 완료될 때까지 대기
#     while True:
#         resp = requests.get(
#             f"https://openapi.vito.ai/v1/transcribe/{transcribe_id}",
#             headers={'Authorization': 'bearer ' + YOUR_JWT_TOKEN}
#         )
#         resp.raise_for_status()
#         response_data = resp.json()
#         status = response_data['status']
#
#         if status == 'completed':
#             break
#         elif status == 'failed':
#             print("전사 작업이 실패했습니다.")
#             return None
#
#         # 일정한 간격으로 폴링
#         time.sleep(5)
#
#     # 전사 결과 가져오기
#     resp = requests.get(
#         f"https://openapi.vito.ai/v1/transcribe/{transcribe_id}",
#         headers={'Authorization': 'bearer ' + YOUR_JWT_TOKEN}
#     )
#     resp.raise_for_status()
#     response_data = resp.json()
#     results = response_data['results']
#
#     # 전사 결과에서 텍스트 부분만 추출하여 리스트로 저장
#     call_details = [utterance['msg'] for utterance in results['utterances']]
#     print(call_details)
#     # 리스트의 각 요소를 하나의 문자열로 결합하여 반환
#     return ' '.join(call_details)
#
#
# # 전처리 함수
# def preprocessing(input_text):
#     removed_data = []
#     s = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","", input_text)
#     if s != '':
#         removed_data.append(s)
#
#     tokenized_data = []
#     for sentence in removed_data:
#         tokenized_sentence = okt.morphs(sentence, stem=True)
#         tokenized_data.append(tokenized_sentence)
#     return tokenized_data
#
# # 토큰화된 단어 벡터화 함수
# def to_vec(tokenized_data):
#     vector_list = []
#     for sentence_tokens in tokenized_data:
#         for token in sentence_tokens:
#             if token in glove.dictionary.keys():
#                 vector_list.append([token, glove.word_vectors[glove.dictionary[token]]])
#     return vector_list
#
# # 입력 문서 평균 벡터 구하기 함수
# def get_document_vec(word_vec_list):
#     input_vec = None
#     if word_vec_list is not None:
#         for word_vec in word_vec_list:
#             if input_vec is None:
#                 input_vec = word_vec[1]
#             else:
#                 input_vec = input_vec + word_vec[1]
#
#     if input_vec is not None:
#         input_vec = input_vec / len(word_vec_list)
#
#     return input_vec
#
# # 코사인 유사도 함수
# def cos_sim(A, B):
#   return dot(A, B)/(norm(A)*norm(B))
#
# # 모델에 해당 기능을 추가하는 함수
# def diagnose_phishing(call_details):
#     tokenized_data = preprocessing(call_details)
#     word_vec_list = to_vec(tokenized_data)
#     input_vec = get_document_vec(word_vec_list)
#
#     # 비교 문서 벡터 리스트 가져오기
#     with open("phishing_vec_list.pkl","rb") as f:
#         compare_vec_list = pickle.load(f)
#
#     # 코사인 유사도 평균 구하기
#     mean_sim = 0
#     for doc_vec in compare_vec_list:
#         doc_sim = cos_sim(input_vec, doc_vec)
#         mean_sim += doc_sim
#     mean_sim /= len(compare_vec_list)
#
#     return mean_sim
#
# # 통화 진단하기 API에 대한 Swagger 문서
# @swagger_auto_schema(
#     method='post',
#     operation_description="통화 진단하기 API",
#     request_body=openapi.Schema(
#         type=openapi.TYPE_OBJECT,
#         required=['diagnosis_type'],
#         properties={
#             'diagnosis_type': openapi.Schema(type=openapi.TYPE_STRING, description='진단 타입'),
#             'call_details': openapi.Schema(type=openapi.TYPE_STRING, description='통화 내용'),
#             'audio_file': openapi.Schema(type=openapi.TYPE_FILE, description='통화 녹음 파일'),
#             'suspicion_percentage': openapi.Schema(type=openapi.TYPE_NUMBER, description='의심도(퍼센트)'),
#         },
#     ),
#     responses={200: '성공적으로 진단됨', 400: '잘못된 요청'},
# )
# # 통화 진단하기
# @api_view(['POST'])
# def diagnose_voice(request):
#     serializer = DiagnosisSerializer(data=request.data)
#     if serializer.is_valid():
#         diagnosis_type = serializer.validated_data.get('diagnosis_type')
#         call_details = serializer.validated_data.get('call_details')
#         audio_file = serializer.validated_data.get('audio_file')
#         suspicion_percentage = serializer.validated_data.get('suspicion_percentage')
#
#         if diagnosis_type == '직접 통화 내용 입력하기':
#             # '직접 통화 내용 입력하기'일 경우 audio_file 필드는 무시
#             serializer.validated_data.pop('call_details', None)
#             if call_details:
#                 suspicion = diagnose_phishing(call_details)
#                 if suspicion is not None:
#                     suspicion_percentage = round(suspicion * 100)
#                     return Response({'suspicion_percentage': f'{suspicion_percentage}%'}, status=status.HTTP_200_OK)
#                 else:
#                     return Response({'message': '진단 결과를 얻을 수 없습니다.'}, status=status.HTTP_400_BAD_REQUEST)
#             else:
#                 return Response({'message': '통화 내용이 제공되지 않았습니다.'}, status=status.HTTP_400_BAD_REQUEST)
#
#         # 통화 녹음본으로 입력하기
#         elif diagnosis_type == '통화 녹음본으로 입력하기':
#             # '통화 녹음본으로 입력하기'일 경우 call_details 필드는 무시
#             serializer.validated_data.pop('audio_file', None)
#             # 오디오 파일을 텍스트로 변환
#             if audio_file:
#                 call_details = audio_to_text(audio_file)
#             else:
#                 # audio_file이 제공되지 않은 경우, URL로부터 텍스트 가져오기
#                 url = 'http://127.0.0.1:8000/voice/'
#                 headers = {
#                     'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0'}
#                 try:
#                     response = requests.get(url, headers=headers)
#                     response.raise_for_status()
#                     call_details = response.text
#                 except Exception as e:
#                     print(f"Error fetching text from URL: {e}")
#                     return Response({'message': '텍스트를 가져올 수 없습니다.'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#
#             if call_details:
#                 suspicion = diagnose_phishing(call_details)
#                 if suspicion is not None:
#                     suspicion_percentage = round(suspicion * 100)
#                     return Response(
#                         {'suspicion_percentage': f'{suspicion_percentage}%', 'call_details': call_details},
#                         status=status.HTTP_200_OK)
#                 else:
#                     return Response({'message': '진단 결과를 얻을 수 없습니다.'}, status=status.HTTP_400_BAD_REQUEST)
#             else:
#                 return Response({'message': '오디오 파일이나 URL로부터 텍스트를 가져올 수 없습니다.'}, status=status.HTTP_400_BAD_REQUEST)
#         serializer.save()
#
#         if diagnosis_type == '직접 통화 내용 입력하기':
#             return Response({'message': '성공적으로 입력되었습니다.', 'data': serializer.data}, status=status.HTTP_201_CREATED)
#         elif diagnosis_type == '통화 녹음본으로 입력하기':
#             return Response({'message': '성공적으로 업로드되었습니다.', 'data': serializer.data},
#                             status=status.HTTP_201_CREATED)
#
#     return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)