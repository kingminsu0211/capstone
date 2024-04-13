from django.db import models
from konlpy.tag import Okt
from glove import Glove
import numpy as np
from numpy.linalg import norm
import pickle
import re

class Diagnosis(models.Model):
    DIAGNOSIS_TYPE_CHOICES = [
        ('통화 녹음본으로 입력하기', '통화 녹음본으로 입력하기'),
        ('직접 통화 내용 입력하기', '직접 통화 내용 입력하기'),
    ]

    diagnosis_type = models.CharField(
        max_length=50,
        choices=DIAGNOSIS_TYPE_CHOICES,
    )

    audio_file = models.FileField(upload_to='voice_recordings/',null=True, blank=True)
    call_details = models.TextField(null=True, blank=True)
    diagnosis_date = models.DateTimeField(auto_now_add=True)
    suspicion_percentage = models.FloatField(null=True,blank=True)
