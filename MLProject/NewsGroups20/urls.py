from django.conf.urls import url
from . import views
urlpatterns = [
    url(r'^$', views.index, name="index"),
    url(r'^classification/', views.classification, name="classification"),
    url(r'^clustering/', views.clustering, name="clustering"),
    url(r'^live_news/', views.live_news, name="live_news"),
    url(r'^classify_news', views.classify_news, name="classify_news"),
    url(r'^alchemy_api_test_results/', views.alchemy_api_test_results, name="alchemy_api_test_results"),
]
