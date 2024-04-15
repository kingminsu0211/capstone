from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sentence_transformers import SentenceTransformer
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from rest_framework.decorators import api_view


# 엑셀 파일을 해당 디렉토리에 저장하여 불러오도록 합니다.
train_data = pd.read_excel('대처방법_챗봇.xlsx', sheet_name='챗봇 QA')

# 각 모델을 로드합니다.
model_finetune_5 = SentenceTransformer('chatbot_5')

def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

help_text = "문의하신 사항을 제대로 이해하지 못했습니다. 다시 문의해주십시오."

train_data['embedding_f5'] = train_data.apply(lambda row: model_finetune_5.encode(row.text), axis = 1)

@swagger_auto_schema(
    method='post',
    operation_description='FAQ 챗봇',
    request_body=openapi.Schema(
        type=openapi.TYPE_OBJECT,
        required=['question'],
        properties={
            'question': openapi.Schema(type=openapi.TYPE_STRING, description='질문 내용'),
        },
    ),
    responses={
            200: openapi.Response(description='성공', schema=openapi.Schema(type=openapi.TYPE_OBJECT, properties={
                'answer_f5': openapi.Schema(type=openapi.TYPE_STRING, description='모델 답변'),
            })),
            400: openapi.Response(description='잘못된 요청', schema=openapi.Schema(type=openapi.TYPE_OBJECT, properties={
                'error': openapi.Schema(type=openapi.TYPE_STRING, description='에러 메시지'),
            })),
            405: openapi.Response(description='허용되지 않는 메서드', schema=openapi.Schema(type=openapi.TYPE_OBJECT, properties={
                'error': openapi.Schema(type=openapi.TYPE_STRING, description='에러 메시지'),
            })),
        }
)
@api_view(['post'])
@csrf_exempt
def chatbot(request):
    if request.method == 'POST':
        data = request.POST
        question = data.get('question', None)

        if question:
            # 각 모델을 통해 답변을 가져옵니다.
            answer_f5 = return_answer_f5(question)

            # JSON 응답 생성
            response_data = {
                'answer_f5': answer_f5
            }
            return JsonResponse(response_data)
        else:
            return JsonResponse({'error': 'Question not provided'}, status=400)
    else:
        return JsonResponse({'error': 'Only POST method allowed'}, status=405)

def return_answer_f5(question):
    embedding = model_finetune_5.encode(question)
    train_data['score_f5'] = train_data.apply(lambda x: cos_sim(x['embedding_f5'], embedding), axis=1)
    if train_data.loc[train_data['score_f5'].idxmax()]['score_f5'] < 0.65:
        return help_text
    return train_data.loc[train_data['score_f5'].idxmax()]['answer']
