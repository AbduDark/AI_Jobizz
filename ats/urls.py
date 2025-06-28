from django.urls import path
from .views import ResumeAnalysisView

urlpatterns = [
    path('resume/analyze/', ResumeAnalysisView.as_view(), name='analyze-resume'),
]