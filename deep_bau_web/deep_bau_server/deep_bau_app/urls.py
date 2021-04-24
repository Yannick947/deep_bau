from django.urls import path
from deep_bau_app import views

urlpatterns = [
    path('', views.index, name='index'),
    path("app/predict", views.predict, name="predict"),
]