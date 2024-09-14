from django.urls import path, include
from .views import load_frontend, PredictLoadAPIView 

urlpatterns = [
    path('', load_frontend, name='frontend'),  # Map root URL to load your frontend
    path('predict/', PredictLoadAPIView.as_view(), name='predict_load'),
    #path('scrape/', ScrapeAPIView.as_view(), name='scrape'),
]
