from django.urls import include, path
from rest_framework import routers
from . import views
from .views import *
from rest_framework import urls

router = routers.DefaultRouter()

router.register(r'check', views.CheckAccountViewset)


urlpatterns = [
    path('',include(router.urls)),
    path('first/', include('first.urls')),
    path('signup/', signup, name='signup'),
    path('login/', user_login, name='login'),
    path('logout/', user_logout, name='logout'),
    path('myview/', current_user, name='current-user'),
    # path('api-auth/', include('rest_framework.urls')),
]