#coding: utf-8
from . import views
from django.conf.urls import url

urlpatterns = [
		url(r'^$', views.get_json, name='get_json'),
]
