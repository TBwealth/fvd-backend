import librosa
import numpy as np
import tensorflow as tf
from pytube import YouTube
import ffmpeg
from urllib.error import HTTPError
import pytubefix
import shutil
import os
import requests

def analyse_audo(audio_path, duration=None):
    # Load audio and extract features
    y, sr = librosa.load(audio_path, sr=16000, duration=duration)
    # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # mfcc_scaled = np.mean(mfcc.T, axis=0)

        # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
        # Ensure consistent dimensions
    # Transpose and slice to match expected input shape
    mfcc_processed = mfcc.T  # Transpose to get (time_steps, features)

    # Slice or pad to ensure consistent shape
    if mfcc_processed.shape[0] > 128:
        mfcc_processed = mfcc_processed[:128, :]
    else:
        # Pad if shorter
        pad_width = ((0, max(0, 128 - mfcc_processed.shape[0])), (0, 0))
        mfcc_processed = np.pad(mfcc_processed, pad_width, mode='constant')
    
    # Reshape to match model input shape (128, 32, 1)
    mfcc_reshaped = mfcc_processed[:, :32].reshape(1, 128, 32, 1)


    # Load pre-trained model and predict
    model = tf.keras.models.load_model('final_hybrid_model.h5')
    # prediction = model.predict(np.expand_dims(mfcc_reshaped, axis=0))
    prediction = model.predict(mfcc_reshaped)

    # Return response
    is_fake = prediction[0][0] > 0.7  # Load audio at specified sampling rate
    # Delete the audio file after analysis
    if os.path.exists(audio_path):
        os.remove(audio_path)
    return {"is_fake": is_fake, "confidence": float(prediction[0][0])}

def get_audio_from_youtube_url(url):   
     # Check if ffmpeg is available
    if not shutil.which("ffmpeg"):
        return {"error": "FFmpeg is not installed or not found in PATH."}

     # Create temp directory if it doesn't exist
    temp_dir = 'temp_audio'
    os.makedirs(temp_dir, exist_ok=True)

    audio_path = os.path.join(temp_dir, 'youtube_audio.mp3')  # Specify the path for the audio file


    try:
        yt = pytubefix.YouTube(url)
        stream_url = yt.streams[0].url  # Get the URL of the video stream
        # print(stream_url)
        # Probe the audio streams (use it in case you need information like sample rate):
        #probe = ffmpeg.probe(stream_url)
        #audio_streams = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
        #sample_rate = audio_streams['sample_rate']
        # Read audio into memory buffer.
        # Get the audio using stdout pipe of ffmpeg sub-process.
        # The audio is transcoded to PCM codec in WAC container.
        audio, err = (
            ffmpeg
            .input(stream_url)
            .output("pipe:", format='mp3', acodec='mp3')  # Select WAV output format, and pcm_s16le auidio codec. My add ar=sample_rate
            .run(capture_stdout=True)
        )
        # Write the audio buffer to file for testing
        with open(audio_path, 'wb') as f:
            f.write(audio)

        # Analyze the audio
        analysis_response = analyse_audo(audio_path, duration=60)

    except HTTPError as e:
        return {"error1": str(e)} # Handle HTTPError
    except pytubefix.exceptions.RegexMatchError as e:
        return {"error2": "Invalid YouTube URL. Please check the URL format."}  # Handle RegexMatchError

    return analysis_response

def is_audio_file(filename):
    print(filename)
    audio_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg']
    return any(filename.lower().endswith(ext) for ext in audio_extensions)

def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def extract_audio_from_video(video_path):
    audio_path = video_path.rsplit('.', 1)[0] + '.flac'  # Change extension to .flac
    # Use ffmpeg to extract audio
    ffmpeg.input(video_path).output(audio_path, format='flac').run(overwrite_output=True)
    return audio_path

def extract_audio_from_video_url(video_url):
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, video_url.split("/")[-1])  # Save with the original filename
    audio_path = os.path.join(temp_dir, f"{os.path.splitext(video_path)[0]}.flac")  # Change extension to .flac

    # Download the video file
    response = requests.get(video_url)
    with open(video_path, 'wb') as f:
        f.write(response.content)

    # Use ffmpeg to extract audio
    ffmpeg.input(video_path).output(audio_path, format='flac').run(overwrite_output=True)

    # Clean up the video file after extraction
    if os.path.exists(video_path):
        os.remove(video_path)

    return audio_path

def download_audio(url):
    response = requests.get(url)
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)
    audio_path = os.path.join(temp_dir, url.split("/")[-1])  # Save with the original filename

    with open(audio_path, 'wb') as f:
        f.write(response.content)

    return audio_path