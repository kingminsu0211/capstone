from django.urls import path
from .views import *

urlpatterns = [
    path('chatbot/', chatbot, name='chatbot'),
    # 다른 URL 매핑들...
]
