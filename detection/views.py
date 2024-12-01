from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import FileUploadParser
import librosa
import numpy as np
import tensorflow as tf
from.traindataset import train_function

class FakeVoiceDetectionView(APIView):
    parser_classes = [FileUploadParser]

    def post(self, request):
        file_obj = request.FILES['file']
        audio_path = f'/tmp/{file_obj.name}'
        
        # Save the uploaded file
        with open(audio_path, 'wb') as f:
            for chunk in file_obj.chunks():
                f.write(chunk)

        # Load audio and extract features
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)

        # Load pre-trained model and predict
        model = tf.keras.models.load_model('path/to/your/model.h5')
        prediction = model.predict(np.expand_dims(mfcc_scaled, axis=0))

        # Return response
        is_fake = prediction[0][0] > 0.5
        return Response({"is_fake": is_fake, "confidence": float(prediction[0][0])})
    
class train_dataset(APIView):

    def get(self, request):
        train_function()



