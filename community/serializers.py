from rest_framework import serializers
from .models import Post, Comment, Report, Ask, AskComment
from user.models import CustomUser

class PostSerializer(serializers.ModelSerializer):
    def get_user_nickname(self, obj):
        return obj.user.nickname
    class Meta:
        model = Post
        fields = '__all__'

class CommentSerializer(serializers.ModelSerializer):
    def get_user_nickname(self, obj):
        return obj.user.nickname
    class Meta:
        model = Comment
        fields = '__all__'
        read_only_fields= ['post','user_nickname']

class ReportSerializer(serializers.ModelSerializer):
    def get_user_nickname(self, obj):
        return obj.user.nickname
    class Meta:
        model = Report
        fields = '__all__'

class AskSerializer(serializers.ModelSerializer):
    class Meta:
        model = Ask
        fields = "__all__"

class AskCommentSerializer(serializers.ModelSerializer):
    def get_user_nickname(self, obj):
        return obj.user.nickname
    class Meta:
        model = AskComment
        fields = '__all__'
        read_only_fields = ['post','user_nickname']
