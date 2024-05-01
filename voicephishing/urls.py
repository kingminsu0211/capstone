from django.urls import include, path
from .views import *
from rest_framework import routers

router = routers.DefaultRouter()


urlpatterns = [
    # path('user/', include('user.urls')),
    path('diagnosis/', DiagnosisListView.as_view(), name='diagnosis-list'),
    path('voice/', diagnose_voice),

    # 다른 URL 패턴들을 필요에 따라 추가
]
