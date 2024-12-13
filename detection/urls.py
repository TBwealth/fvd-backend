from django.urls import path
from .views import FakeVoiceDetectionView, train_dataset, analyse_youtube_video, AudioLinkAnalysisView

urlpatterns = [
    path('detect/fileUpload', FakeVoiceDetectionView.as_view(), name='detect-fake-voice-upload'),
    path('detect/youtubeLink', analyse_youtube_video.as_view(), name='detect-fake-voice-youtube-link'),
    path('detect/link', AudioLinkAnalysisView.as_view(), name='detect-fake-voice-link'),
    path('train/', train_dataset.as_view(), name='train'),
]