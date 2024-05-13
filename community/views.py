from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework import generics
from .serializers import *
from .models import *
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from datetime import timedelta
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.generics import ListAPIView
from rest_framework import filters
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import api_view, permission_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.decorators import authentication_classes

def is_admin(user):
    return user.is_staff and user.is_superuser

# 게시물 검색
@swagger_auto_schema(
    operation_description="게시글 검색"
)
class PostSearchAPIView(ListAPIView):
    serializer_class = PostSerializer
    def get_queryset(self):
        queryset = Post.objects.all()
        search_query = self.request.query_params.get('q', None)
        if search_query:
            queryset = queryset.filter(title__icontains=search_query)
        return queryset

#내가 쓴 게시글
class MyPostListView(generics.ListAPIView):
    serializer_class = PostSerializer

    @swagger_auto_schema(operation_description="내가 쓴 게시글 목록을 가져옵니다.")
    def get_queryset(self):
        return Post.objects.filter(writer=self.request.user)

# 전체 게시글
class AllPostListView(generics.ListAPIView):
    serializer_class = PostSerializer

    @swagger_auto_schema(operation_description="전체 게시글 목록을 가져옵니다.")
    def get_queryset(self):
        return Post.objects.all()

#게시글 자세히 보기
class PostDetailView(APIView):
    @swagger_auto_schema(
        operation_description="특정 게시글을 자세히 보여줍니다.",
        manual_parameters=[
            openapi.Parameter(
                'post_id',
                openapi.IN_PATH,
                description="게시글 ID",
                type=openapi.TYPE_INTEGER,
            )
        ],
    )
    def get(self, request, post_id):
        try:
            post = Post.objects.get(id=post_id)
        except Post.DoesNotExist:
            return Response({"message": "게시물을 찾을 수 없습니다."}, status=status.HTTP_404_NOT_FOUND)

        serializer = PostSerializer(post)
        return Response(serializer.data, status=status.HTTP_200_OK)

#게시글 쓰기
@swagger_auto_schema(
    method='post',
    request_body=PostSerializer,
    operation_description="게시글 쓰기"
)
@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
@csrf_exempt
def create_post(request):
    serializer = PostSerializer(data=request.data)

    if serializer.is_valid():
        # Assuming the serializer has 'title' and 'content' fields
        title = serializer.validated_data.get('title')
        content = serializer.validated_data.get('content')
        report_number = serializer.validated_data.get('report_number')

        user = request.user
        user_nickname = user.nickname

        # Create a new Post object
        post =Post.objects.create(title=title, content=content, writer=user, report_number=report_number)

        return Response({'message': '게시물이 성공적으로 작성되었습니다.','유저 닉네임': user_nickname}, status=status.HTTP_201_CREATED)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# 게시글 수정
@swagger_auto_schema(
    method='put',
    request_body=PostSerializer,
    operation_description="게시글 수정"
)
@api_view(['PUT'])
def update_post(request, post_id):
    post = get_object_or_404(Post, id=post_id)

    if request.user != post.writer:
        return Response({'error': '게시물을 수정할 권한이 없습니다.'}, status=status.HTTP_403_FORBIDDEN)

    serializer = PostSerializer(post, data=request.data)

    if serializer.is_valid():
        serializer.save()
        return Response({'message': '게시물이 성공적으로 수정되었습니다.'}, status=status.HTTP_200_OK)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#댓글 목록
class AllCommentListView(generics.ListAPIView):
    serializer_class = CommentSerializer

    @swagger_auto_schema(operation_description="전체 댓글 목록")
    def get_queryset(self):
        return Comment.objects.all()

class MyCommentListView(generics.ListAPIView):
    serializer_class = CommentSerializer

    @swagger_auto_schema(operation_description="내가 쓴 댓글 목록")
    def get_queryset(self):
        return Comment.objects.filter(user=self.request.user)

#댓글 쓰기
@swagger_auto_schema(
    method='post',
    request_body=CommentSerializer,
    operation_description="댓글 쓰기"
)
@csrf_exempt
@api_view(['POST'])
def comment_write(request, post_id):
    post = get_object_or_404(Post, id=post_id)
    serializer = CommentSerializer(data=request.data)

    if serializer.is_valid():
        content = serializer.validated_data.get('content')
        user_id = serializer.validated_data.get('user')
        # post_id = serializer.validated_data.get('post')
        # nickname = serializer.validated_data.get('nickname')

        # 사용자 및 게시물 인스턴스를 가져옵니다
        # user = get_object_or_404(CustomUser, id=user_id.id)
        user = request.user
        user_nickname = user.nickname
        # post = get_object_or_404(Ask, id=post_id)
        # 새로운 댓글 객체를 생성합니다
        comment = Comment.objects.create(
            content=content,
            user=user,
            post=post,
            # nickname=nickname,
        )
        user_nickname = comment.get_user_nickname()
        return Response({'message': '댓글이 성공적으로 작성되었습니다.', '유저 닉네임': user_nickname}, status=status.HTTP_201_CREATED)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#댓글 수정
@swagger_auto_schema(
    method='put',
    request_body=CommentSerializer,
    operation_description="댓글 수정"
)
@api_view(['PUT'])
def comment_update(request, comment_id):
    comment = get_object_or_404(Comment, id=comment_id)

    # 현재 사용자와 댓글의 작성자가 같은지 확인
    if request.user.id != comment.user.id:
        return Response({'message': '댓글을 수정할 수 있는 권한이 없습니다.'}, status=status.HTTP_403_FORBIDDEN)

    serializer = CommentSerializer(comment, data=request.data)

    if serializer.is_valid():
        serializer.save()
        return Response({'message': '댓글이 성공적으로 수정되었습니다.'}, status=status.HTTP_200_OK)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
# 번호 검색
@swagger_auto_schema(
    operation_description="번호 검색"
)
class NumberSearchAPIView(ListAPIView):
    serializer_class = ReportSerializer
    def get_queryset(self):
        queryset = Report.objects.all()
        search_query = self.request.query_params.get('q', None)
        if search_query:
            queryset = queryset.filter(report_number__icontains=search_query)
        return queryset

# 모든 신고 리스트
class AllReportListView(generics.ListAPIView):
    serializer_class = ReportSerializer

    @swagger_auto_schema(operation_description="모든 신고 리스트를 가져옵니다.")
    def get_queryset(self):
        return Report.objects.all()

#신고하기
# @login_required
@swagger_auto_schema(
    method='post',
    request_body=ReportSerializer,
    operation_description="신고하기"
)
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
@csrf_exempt
@api_view(['POST'])
def report(request):
    serializer = ReportSerializer(data=request.data)

    if serializer.is_valid():
        report_number = serializer.validated_data.get('report_number')
        report_type = serializer.validated_data.get('report_type')
        report_content = serializer.validated_data.get('report_content')
        reporter = serializer.validated_data.get('reporter')
        voice_phishing_record_id = serializer.validated_data.get('voice_phishing_record.id')

        user = request.user
        user_nickname = user.nickname if user.is_authenticated else "Anonymous"

        if isinstance(user, CustomUser):
            # 인증된 사용자인 경우에만 보고서를 작성합니다
            report = Report.objects.create(
                report_number=report_number,
                report_type=report_type,
                report_content=report_content,
                reporter=user,
                voice_phishing_record_id=voice_phishing_record_id
            )

            return Response({'message': '성공적으로 신고되었습니다.', '유저 닉네임': user_nickname}, status=status.HTTP_201_CREATED)
        else:
            # 인증되지 않은 사용자가 접근할 경우 오류를 반환합니다
            return Response({'detail': 'Authentication credentials were not provided.'},
                            status=status.HTTP_403_FORBIDDEN)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#신고 수정
@swagger_auto_schema(
    method='put',
    request_body=ReportSerializer,
    operation_description="신고 수정"
)
@api_view(['PUT'])
def report_update(request, report_id):
    report = get_object_or_404(Report, id=report_id)

    # 현재 사용자와 리포트의 작성자가 같은지 확인
    if request.user.id != report.reporter.id:
        return Response({'message': '게시글을 수정할 수 있는 권한이 없습니다.'}, status=status.HTTP_403_FORBIDDEN)

    serializer = ReportSerializer(report, data=request.data)

    if serializer.is_valid():
        serializer.save()
        return Response({'message': '신고가 성공적으로 수정되었습니다.'}, status=status.HTTP_200_OK)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
#내 신고내역
class MyReportListView(generics.ListAPIView):
    serializer_class = ReportSerializer

    @swagger_auto_schema(operation_description="내 신고내역을 가져옵니다.")
    def get_queryset(self):
        # 현재 날짜
        current_date = timezone.now()

        # 일주일 전의 날짜 계산
        one_week_ago = current_date - timedelta(days=7)

        # 신고한 날짜가 일주일 전 이후인 데이터 필터링
        return Report.objects.filter(reporter=self.request.user, report_date__gte=one_week_ago)

#신고내역 자세히 보기
class ReportDetailView(APIView):
    def get(self, request, report_id):
        try:
            report = Report.objects.get(id=report_id)
        except Report.DoesNotExist:
            return Response({"message": "신고 정보를 찾을 수 없습니다."}, status=status.HTTP_404_NOT_FOUND)

        serializer = ReportSerializer(report)
        return Response(serializer.data, status=status.HTTP_200_OK)

#문의하기 작성
@swagger_auto_schema(
    method='post',
    request_body=AskSerializer,
    operation_description="문의하기 작성"
)
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
@csrf_exempt
@api_view(['POST'])
def create_ask(request):
    serializer = AskSerializer(data=request.data)

    if serializer.is_valid():
        title = serializer.validated_data.get('title')
        content = serializer.validated_data.get('content')

        user = request.user
        user_nickname = user.nickname if user.is_authenticated else "Anonymous"

        if isinstance(user, CustomUser):
            ask = Ask.objects.create(title=title, content=content, writer=user)
            return Response({'message': '성공적으로 문의되었습니다.', '유저 닉네임': user_nickname}, status=status.HTTP_201_CREATED)
        else:
            return Response({'detail': 'Authentication credentials were not provided.'}, status=status.HTTP_403_FORBIDDEN)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# 문의사항 수정
@swagger_auto_schema(
    method='put',
    request_body=AskSerializer,
    operation_description="문의사항 수정"
)
@api_view(['PUT'])
def update_ask(request, post_id):
    ask = get_object_or_404(Ask, id=post_id)

    # 댓글이 달려있는지 확인
    has_comments = AskComment.objects.filter(post=ask).exists()

    if has_comments:
        return Response({'error': '답변이 달린 게시물은 수정할 수 없습니다.'}, status=status.HTTP_403_FORBIDDEN)

    if request.user != ask.writer:
        return Response({'error': '게시물을 수정할 권한이 없습니다.'}, status=status.HTTP_403_FORBIDDEN)

    serializer = AskSerializer(ask, data=request.data)

    if serializer.is_valid():
        serializer.save()
        user = request.user
        user_nickname = user.nickname
        return Response({'message': '게시물이 성공적으로 수정되었습니다.','유저 닉네임': user_nickname}, status=status.HTTP_200_OK)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@swagger_auto_schema(
    method='post',
    request_body=AskCommentSerializer,
    operation_description="문의사항 답변"
)
@api_view(['POST'])
def ask_comment_write(request, post_id):
    serializer = AskCommentSerializer(data=request.data)

    if serializer.is_valid():
        content = serializer.validated_data.get('content')
        user_id = serializer.validated_data.get('user')

        # 사용자 및 게시물 인스턴스를 가져옵니다
        # user = get_object_or_404(CustomUser, id=user_id.id)
        user = request.user
        post = get_object_or_404(Ask, id=post_id)

        # 관리자인 경우에만 댓글 객체를 생성합니다
        if user.is_staff:
            comment = AskComment.objects.create(
                content=content,
                user=user,
                post=post,
            )
            user_nickname = user.nickname

            return Response({'message': '답변이 성공적으로 작성되었습니다.', '관리자 닉네임': user_nickname}, status=status.HTTP_201_CREATED)
        else:
            return Response({'error': '관리자만 답변을 작성할 수 있습니다.'}, status=status.HTTP_403_FORBIDDEN)

    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# 내 문의 리스트
class MyAskListView(generics.ListAPIView):
    serializer_class = AskSerializer

    @swagger_auto_schema(operation_description="내가 쓴 문의하기 목록을 가져옵니다.")
    def get_queryset(self):
        return Ask.objects.filter(writer=self.request.user)

# 전체 문의 리스트
class AllAskListView(generics.ListAPIView):
    serializer_class = AskSerializer

    @swagger_auto_schema(operation_description="전체 문의하기 목록을 가져옵니다.")
    def get_queryset(self):
        return Ask.objects.all()