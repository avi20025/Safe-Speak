from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Union, Tuple
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import base64
import joblib
import numpy as np
import librosa
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

app = Flask(__name__)

# CORS for all routes, allowing requests from dedicated front end
CORS(app) 

# Configuration variables
SAMPLE_RATE = 22050
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
BEST_MODEL_DIR = "best_models"
BEST_TRANSFORMER_PATH = os.path.join(BEST_MODEL_DIR, "best_transformer.keras")  # Added .keras extension
BEST_ENSEMBLE_PATH = os.path.join(BEST_MODEL_DIR, "best_ensemble.keras")  # Added .keras extension

def extract_high_advanced_features(audio_path: str, sr: int = SAMPLE_RATE, n_mfcc: int = N_MFCC,
                           n_fft: int = N_FFT, hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    Extract a comprehensive set of audio features beyond basic MFCCs.

    Args:
        audio_path: Path to audio file
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length for feature extraction

    Returns:
        Feature vector combining spectral, temporal, and perceptual features
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr)

        # ===== Spectral Features =====

        # 1. MFCCs (mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_vars = np.var(mfccs, axis=1)

        # 2. Spectral centroid (center of mass of the spectrum)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        cent_mean = np.mean(cent)
        cent_var = np.var(cent)

        # 3. Spectral bandwidth (variance around the centroid)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        bandwidth_mean = np.mean(bandwidth)
        bandwidth_var = np.var(bandwidth)

        # 4. Spectral contrast (valley/peak contrast in each frequency subband)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        contrast_mean = np.mean(contrast, axis=1)

        # 5. Spectral rolloff (frequency below which a percentage of spectral energy lies)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        rolloff_mean = np.mean(rolloff)
        rolloff_var = np.var(rolloff)

        # 6. Chroma features (pitch class profiles)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        chroma_mean = np.mean(chroma, axis=1)

        # ===== Temporal Features =====

        # 7. Zero crossing rate (sign changes along a signal)
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
        zcr_mean = np.mean(zcr)
        zcr_var = np.var(zcr)

        # 8. Tempo estimation (beat tracking)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        # ===== Voice-specific Features =====

        # 9. Harmonic components
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # 10. RMS energy (root-mean-square energy)
        rms = librosa.feature.rms(y=y, hop_length=hop_length)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)

        # 11. Spectral flatness (how noise-like a sound is)
        flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
        flatness_mean = np.mean(flatness)
        flatness_var = np.var(flatness)

        # ===== Voice Formant Features =====

        # 12. Apply a simple approximation of formant frequencies
        # This uses the first few MFCCs which roughly correspond to vocal tract shape
        formant_approximation = mfcc_means[1:5]  # MFCCs 1-4 roughly correspond to formants

        # ===== Statistical Features =====

        # 13. Pitch statistics based on CREPE if needed (using simpler approach here)
        # Extract pitch using auto-correlation method
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        pitch_max = np.mean(np.max(pitches, axis=0))
        pitch_var = np.var(np.max(pitches, axis=0))

        # 14. Voice jitter approximation (variation in pitch periodicity)
        # This is a simplified approach - professional voice analysis uses dedicated algorithms
        if len(y) > sr // 1000:  # Only if we have enough samples
            y_trimmed = y[:sr//2]  # Use first half-second for efficiency
            zero_crossings = librosa.zero_crossings(y_trimmed)
            zero_crossing_indices = np.where(zero_crossings)[0]
            if len(zero_crossing_indices) > 2:
                # Calculate intervals between consecutive zero crossings
                intervals = np.diff(zero_crossing_indices)
                jitter_approx = np.var(intervals) / (np.mean(intervals) + 1e-5)
            else:
                jitter_approx = 0
        else:
            jitter_approx = 0

        # Fixed: Make sure all arrays have the same dimensions (1D)
        # Reshape any 2D arrays to 1D before concatenation
        feature_vector = np.concatenate([
          mfcc_means.reshape(-1, 1),             # Convert to 2D (N, 1)
          mfcc_vars.reshape(-1, 1),              # Convert to 2D
          np.array([cent_mean, cent_var]).reshape(-1, 1),
          np.array([bandwidth_mean, bandwidth_var]).reshape(-1, 1),
          contrast_mean.reshape(-1, 1),          # Convert to 2D
          np.array([rolloff_mean, rolloff_var]).reshape(-1, 1),
          chroma_mean.reshape(-1, 1),            # Convert to 2D
          np.array([zcr_mean, zcr_var]).reshape(-1, 1),
          np.array([tempo]).reshape(-1, 1),      # Convert to 2D
          np.array([rms_mean, rms_var]).reshape(-1, 1),
          np.array([flatness_mean, flatness_var]).reshape(-1, 1),
          formant_approximation.reshape(-1, 1),  # Convert to 2D
          np.array([pitch_max, pitch_var]).reshape(-1, 1),
          np.array([jitter_approx]).reshape(-1, 1)
        ], axis=0)  # Stack them vertically

        return feature_vector

    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

# Load pre-trained model
def load_best_models():
    """Load the best saved models."""
    try:
        best_base = tf.keras.models.load_model(BEST_TRANSFORMER_PATH)
        return best_base
    except Exception as e:
        print(f"Error loading best models: {str(e)}")
        return None, None
    
# Function to validate audio file    
def validate_audio_file(file_path: str) -> bool:
    """Check if file exists and has valid format."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    if not file_path.lower().endswith(".wav"):
        raise ValueError("Only WAV files are supported")
    return True

@app.route('/verify-audio', methods=['POST'])
def predict_audio():
    
    """
    Endpoint to handle audio file uploads from UI post request and predicting whether the audio is real or fake 
    """
    
    uploads_dir = "/Users/boo/PyCharmProjects/ML Model API/uploads"

    # Load trained model and scaler for feature normalization
    best_base_model = load_best_models()
    
    # check if the request contains an audio file
    if "audio_file" not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    # retrieve the uploaded file
    audio_file = request.files['audio_file']
    
    if audio_file is None:
        return jsonify({'error': 'No audio file'}), 400
    
    print("File receive:", audio_file.filename)
    
    try:
        # define the file path where the uploaded file will be temporarily saved
        file_path = os.path.join(uploads_dir, audio_file.filename)
        audio_file.save(file_path)

        # Validate the audio file
        validate_audio_file(file_path)
    
        # analyze the audio file using the trained model and feature scaler
        features = extract_high_advanced_features(file_path)
        if features is not None:
            features = features.reshape(1, -1, 1)
            prediction = best_base_model.predict(features)[0][0]
            label = "REAL" if prediction < 0.5 else "FAKE"
            print(f"{os.path.basename(file_path)} prediction: Prediction: {label}")
        
        # print("Backend prediction result:", result)  # Debugging print statement

        base_url = request.host_url.rstrip("/") + "/static/plots"
        
        # send back the classification result and plots/statistics in json format to front end
        return jsonify({
            'Prediction': label,
            'FeatureDistributionPlotURL': f"{base_url}/Model_Plots.png",
            'FeatureImportancePlotURL': f"{base_url}/Feature_Importance.png",
            'FeatureStatsPlotURL': f"{base_url}/Feature_Statistics.png",
        }), 200
    
    except Exception as e:
        # handle any errors that occur during processing
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run()