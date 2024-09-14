from django.contrib import admin
from django.urls import path, include
from main.views import load_frontend  # Import the view to load your frontend

urlpatterns = [
    path('', load_frontend, name='frontend'),  # Root URL points to your custom frontend
    path('api/', include('main.urls')),  # API routes
    path('admin/', admin.site.urls),  # Admin routes
]
