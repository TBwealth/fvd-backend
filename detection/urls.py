from django.urls import path
from .views import FakeVoiceDetectionView, train_dataset

urlpatterns = [
    path('detect/', FakeVoiceDetectionView.as_view(), name='detect-fake-voice'),
    path('train/', train_dataset.as_view(), name='train'),
]