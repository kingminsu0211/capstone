from rest_framework import generics
from .serializers import *
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
import re
# import pandas as pd
from konlpy.tag import Okt
from glove import Corpus, Glove
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pickle
import requests
from django.http import FileResponse


class DiagnosisListView(generics.ListCreateAPIView):
    queryset = Diagnosis.objects.all()
    serializer_class = DiagnosisSerializer

glove = Glove.load('glove.model')

okt = Okt()

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
    input_vec = None
    if word_vec_list is not None:
        for word_vec in word_vec_list:
            if input_vec is None:
                input_vec = word_vec[1]
            else:
                input_vec = input_vec + word_vec[1]

    if input_vec is not None:
        input_vec = input_vec / len(word_vec_list)

    return input_vec

# 코사인 유사도 함수
def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))


# 모델에 해당 기능을 추가하는 함수
def diagnose_phishing(call_details):
    tokenized_data = preprocessing(call_details)
    word_vec_list = to_vec(tokenized_data)
    input_vec = get_document_vec(word_vec_list)

    # 비교 문서 벡터 리스트 가져오기
    with open("phishing_vec_list.pkl","rb") as f:
        compare_vec_list = pickle.load(f)

    # 코사인 유사도 평균 구하기
    mean_sim = 0
    for doc_vec in compare_vec_list:
        doc_sim = cos_sim(input_vec, doc_vec)
        mean_sim += doc_sim
    mean_sim /= len(compare_vec_list)

    return mean_sim

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
                suspicion = diagnose_phishing(call_details)
                if suspicion is not None:
                    suspicion_percentage = round(suspicion * 100)
                    return Response({'suspicion_percentage': f'{suspicion_percentage}%'}, status=status.HTTP_200_OK)
                else:
                    return Response({'message': '진단 결과를 얻을 수 없습니다.'}, status=status.HTTP_400_BAD_REQUEST)
            else:
                return Response({'message': '통화 내용이 제공되지 않았습니다.'}, status=status.HTTP_400_BAD_REQUEST)

        elif diagnosis_type == '통화 녹음본으로 입력하기':
            # '통화 녹음본으로 입력하기'일 경우 call_details 필드는 무시
            serializer.validated_data.pop('audio_file', None)
            if audio_file is None:
                return Response({'message': '업로드 되지 않았습니다.', 'data': serializer.data}, status=status.HTTP_201_CREATED)

            elif suspicion_percentage is None:
                return Response({'message': '퍼센트가 입력되지 않았습니다.', 'data': serializer.data},
                                status=status.HTTP_201_CREATED)

        serializer.save()

        if diagnosis_type == '직접 통화 내용 입력하기':
            return Response({'message': '성공적으로 입력되었습니다.', 'data': serializer.data}, status=status.HTTP_201_CREATED)
        elif diagnosis_type == '통화 녹음본으로 입력하기':
            return Response({'message': '성공적으로 업로드되었습니다.', 'data': serializer.data},
                            status=status.HTTP_201_CREATED)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# 오디오 파일 업로드
def send_audio_file_to_colab(request):
    audio_file = request.FILES.get('audio_file')

    # Django에서 코랩으로 파일 보내기
    with open(audio_file.path, 'rb') as f:
        response = FileResponse(f)
        response['Content-Disposition'] = f'attachment; filename="{audio_file.name}"'
        return response