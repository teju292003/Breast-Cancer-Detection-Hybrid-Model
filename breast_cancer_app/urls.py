from django.urls import path
from . import views
app_name = 'breast_cancer_app'
urlpatterns = [
    path('', views.home, name='home'),
    path('predict/', views.predict, name='predict'),
    path('results/<int:pk>/', views.results, name='results'),
    path('compare/', views.compare_models, name='compare'),
    path('about/', views.about, name='about'),
    path('history/', views.prediction_history, name='history'),
]
