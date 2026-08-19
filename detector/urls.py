from django.urls import path

from . import views

app_name = "detector"

urlpatterns = [
    path("", views.index, name="index"),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("dashboard/reset/", views.reset_history, name="reset_history"),
    path("flag/<int:pk>/", views.flag_reading, name="flag_reading"),
    path("api/predict/", views.PredictAPIView.as_view(), name="predict_api"),
]
