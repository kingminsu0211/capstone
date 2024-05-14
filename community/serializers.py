from rest_framework import serializers
from .models import Post, Comment, Report, Ask, AskComment
from user.models import CustomUser

class PostSerializer(serializers.ModelSerializer):
    # def get_user_nickname(self, obj):
    #     return obj.user.nickname
    user_nickname = serializers.CharField(source='writer.nickname', read_only=True)
    class Meta:
        model = Post
        fields = ['id', 'title', 'content', 'created_at', 'updated_at', 'report_number', 'user_nickname']

class CommentSerializer(serializers.ModelSerializer):
    # def get_user_nickname(self, obj):
    #     return obj.user.nickname
    user_nickname = serializers.CharField(source='user.nickname', read_only=True)
    class Meta:
        model = Comment
        fields = ['id', 'content', 'created_at', 'user', 'user_nickname','post']
        read_only_fields = ['post']

class ReportSerializer(serializers.ModelSerializer):
    user_nickname = serializers.CharField(source='reporter.nickname', read_only=True)
    # def get_user_nickname(self, obj):
    #     return obj.user.nickname
    class Meta:
        model = Report
        fields = '__all__'

class AskSerializer(serializers.ModelSerializer):
    user_nickname = serializers.CharField(source='writer.nickname', read_only=True)
    class Meta:
        model = Ask
        fields = "__all__"

class AskCommentSerializer(serializers.ModelSerializer):
    user_nickname = serializers.CharField(source='user.nickname', read_only=True)
    # def get_user_nickname(self, obj):
    #     return obj.user.nickname
    class Meta:
        model = AskComment
        fields = ['id', 'content', 'created_at', 'user', 'user_nickname', 'post']
        read_only_fields = ['post']
