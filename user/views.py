from rest_framework import viewsets

from .serializers import *

from django.contrib.auth.models import User
from django.contrib.auth import login
from django.core.exceptions import ValidationError
from rest_framework.views import APIView
from django.http import HttpRequest
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework import status
from .models import CustomUser
from .serializers import SignupSerializer
from django.contrib.auth import authenticate, logout, login as auth_login
from django.contrib.auth.decorators import login_required
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from rest_framework.permissions import AllowAny
from rest_framework.permissions import IsAuthenticated
from rest_framework.authtoken.models import Token
from django.views.decorators.csrf import csrf_exempt

class CheckAccountViewset(viewsets.ModelViewSet):
    queryset = CustomUser.objects.all()
    serializer_class = AccountsSerializer

# 회원가입 API 문서화
@swagger_auto_schema(method='post', request_body=SignupSerializer)
# @api_view(['POST'])
@permission_classes([AllowAny])
# @permission_classes([IsAuthenticated])
#회원가입
@api_view(['POST'])
def signup(request: HttpRequest):
    serializer = SignupSerializer(data=request.data)
    print("Request Data:", request.data)
    if serializer.is_valid():
        username = serializer.validated_data['username']
        password = serializer.validated_data['password']
        last_name=serializer.validated_data['last_name']
        first_name=serializer.validated_data['first_name']
        nickname = serializer.validated_data['nickname']
        number = serializer.validated_data['number']

        try:
            # Create a new user
            user = CustomUser.objects.create_user(username=username, password=password,last_name=last_name,first_name=first_name,nickname=nickname,number=number)
        except ValidationError as e:
            return Response({'error': e.message}, status=status.HTTP_400_BAD_REQUEST)

        # Log in the user
        # login(request, user)

        return Response({'message': '회원가입이 완료되었습니다.'}, status=status.HTTP_201_CREATED)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# 로그인 API 문서화
@swagger_auto_schema(method='post', manual_parameters=[
    openapi.Parameter('username', openapi.IN_QUERY, description="사용자 ID", type=openapi.TYPE_STRING),
    openapi.Parameter('password', openapi.IN_QUERY, description="비밀번호", type=openapi.TYPE_STRING),
])
@api_view(['POST'])
@permission_classes([AllowAny])
# @csrf_exempt
def user_login(request):
    username = request.query_params.get('username')
    password = request.query_params.get('password')

    try:
        user = authenticate(request, username=username, password=password)
        if user is not None:
            auth_login(request, user)
            return Response({'message': '로그인 성공'}, status=status.HTTP_200_OK)
        else:
            return Response({'error': '아이디 또는 비밀번호가 잘못되었습니다.'}, status=status.HTTP_401_UNAUTHORIZED)

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@swagger_auto_schema(method='post')
@api_view(['POST'])
def user_logout(request):
    logout(request._request)  # request._request 사용
    return Response({'message': '로그아웃 성공'}, status=status.HTTP_200_OK)


# 현재 사용자 정보 API 문서화
@swagger_auto_schema(method='get')
@api_view(['GET'])
@login_required  # 로그인이 필요한 경우에만 접근 가능하도록 하는 데코레이터
def current_user(request):
    if request.user.is_authenticated:
        user = request.user
        return Response({'username': user.username, 'email': user.email}, status=status.HTTP_200_OK)
    else:
        return Response({'message': '로그인이 필요합니다.'}, status=status.HTTP_401_UNAUTHORIZED)