from rest_framework import serializers
from django.contrib.auth.models import User
from .models import CustomUser

from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from rest_framework_simplejwt.views import TokenObtainPairView

from django.db import IntegrityError
from rest_framework import serializers
from .models import CustomUser
from .models import *

from django.contrib.auth.password_validation import validate_password


class AccountsSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ('username','password','last_name','first_name','nickname','number')

class LoginSerializer(serializers.ModelSerializer):
    class Meta:
        model = CustomUser
        fields = ('username','password')

class SignupSerializer(serializers.ModelSerializer):
    password = serializers.CharField(
        write_only=True,
        required=True,
        validators=[validate_password],  # 비밀번호에 대한 검증
    )
    class Meta:
        model = CustomUser
        fields = ('username','password','last_name','first_name','nickname','number')
