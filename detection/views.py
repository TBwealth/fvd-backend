from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import FileUploadParser, MultiPartParser
from rest_framework.decorators import action
from urllib.error import HTTPError
from.traindataset import train_function
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from .analyzer import analyse_audo, get_audio_from_youtube_url, is_audio_file, is_video_file, extract_audio_from_video_url, download_audio
import os
import requests

class FakeVoiceDetectionView(APIView):
    parser_classes = [MultiPartParser]

    @swagger_auto_schema(
    operation_description="Detect if the uploaded audio is fake or real.",
    manual_parameters=[openapi.Parameter(name="file",in_=openapi.IN_FORM,type=openapi.TYPE_FILE,required=True,description="Document")],
    responses={200: openapi.Response('Detection Result', openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'is_fake': openapi.Schema(type=openapi.TYPE_BOOLEAN, description='Indicates if the audio is fake'),
            'confidence': openapi.Schema(type=openapi.TYPE_NUMBER, format=openapi.FORMAT_FLOAT, description='Confidence score of the prediction')
        }
    ))}
    )
    @action(detail=False, methods=['post'])
    def post(self, request):
        file_obj = request.FILES['file']
        temp_dir = 'temp_audio'
        os.makedirs(temp_dir, exist_ok=True)

        audio_path = os.path.join(temp_dir, f'{file_obj.name}')    
        # Save the uploaded file
        with open(audio_path, 'wb') as f:
            for chunk in file_obj.chunks():
                f.write(chunk)        
        
         # Determine if the file is audio or video
        if is_audio_file(file_obj.name):
            analysis_response = analyse_audo(audio_path)
        elif is_video_file(file_obj.name):
            audio_path = self.extract_audio_from_video(audio_path)
            analysis_response = analyse_audo(audio_path)
        else:
            return Response({"error": "Unsupported file type."}, status=400)

        # Clean up the temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return Response(analysis_response)
    
class train_dataset(APIView):

    def get(self, request):
        train_function()

class analyse_youtube_video(APIView):
    @swagger_auto_schema(
    operation_description="Analyze a YouTube video by URL.",
    request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'url': openapi.Schema(type=openapi.TYPE_STRING, description="YouTube video URL")
            },
            required=['url']
        ),
    responses={200: openapi.Response('Analysis Result', openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'result': openapi.Schema(type=openapi.TYPE_STRING, description='Analysis result of the video')
        }
    ))}
    )
    def post(self, request):
        url = request.data.get("url")  # Get the URL from the request body
        try:
            anylysis_response = get_audio_from_youtube_url(url)
            return Response(anylysis_response)
        except HTTPError as e:
            return Response({"error": str(e)}, status=e.code)  # Handle HTTPError
            
class AudioLinkAnalysisView(APIView):
    @swagger_auto_schema(
    operation_description="Analyze a video/audio link.",
    request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'url': openapi.Schema(type=openapi.TYPE_STRING, description="video/audio link")
            },
            required=['url']
        ),
    responses={200: openapi.Response('Analysis Result', openapi.Schema(
        type=openapi.TYPE_OBJECT,
        properties={
            'result': openapi.Schema(type=openapi.TYPE_STRING, description='Analysis result of the link')
        }
    ))}
    )

    def post(self, request):
        url = request.data.get("url")
        if not url:
            return Response({"error": "URL is required."}, status=400)

        # Determine if the URL is audio or video
        if is_audio_file(url):
            audio_path = download_audio(url)
            analysis_response = analyse_audo(audio_path)
        elif is_video_file(url):
            audio_path = extract_audio_from_video_url(url)
            analysis_response = analyse_audo(audio_path)
        else:
            return Response({"error": "Unsupported URL type."}, status=400)

        # Clean up the temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

        return Response(analysis_response)
    
