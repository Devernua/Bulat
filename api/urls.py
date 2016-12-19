# coding: utf-8
from . import views
from django.conf.urls import url

urlpatterns = [
    url(r'^$', views.get_json, name='get_json'),
    url(r'^train$', views.get_train, name='get_train'),
    url(r'^check$', views.get_check, name='get_check'),
]
