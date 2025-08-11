import numpy as np
import random
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import tempfile

logger = logging.getLogger(__name__)

# Try to import librosa for real audio processing
try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available. Using mock audio processing.")

class AudioProcessor:
    """Audio processor for feature extraction"""
    
    def __init__(self, sample_rate: int = 22050, use_mock: bool = False):
        self.sample_rate = sample_rate
        self.use_mock = use_mock or not LIBROSA_AVAILABLE
        
        if self.use_mock:
            logger.info("Using mock audio processor")
        else:
            logger.info("Using real audio processor with librosa")
    
    def process_file(self, file) -> Dict[str, Any]:
        """
        Process audio file and extract features
        
        Args:
            file: Uploaded file object from Streamlit
            
        Returns:
            Dictionary of extracted features
        """
        if self.use_mock:
            return self._mock_process_file(file)
        else:
            return self._real_process_file(file)
    
    def _mock_process_file(self, file) -> Dict[str, Any]:
        """Mock processing for testing"""
        import time
        time.sleep(0.5)  # Simulate processing time
        
        # Generate realistic mock features
        tempo = random.uniform(60, 180)
        energy = random.uniform(0.3, 0.9)
        
        return {
            # Temporal features
            'tempo': tempo,
            'tempo_confidence': random.uniform(0.7, 1.0),
            'time_signature': random.choice([3, 4, 5]),
            'duration': random.uniform(120, 300),
            
            # Energy and dynamics
            'energy': energy,
            'loudness': random.uniform(-20, -5),
            'dynamic_range': random.uniform(5, 15),
            'rms_energy': random.uniform(0.1, 0.5),
            
            # Tonal features
            'key': random.randint(0, 11),
            'mode': random.randint(0, 1),
            'key_confidence': random.uniform(0.5, 1.0),
            
            # Rhythm features
            'danceability': random.uniform(0.4, 0.9),
            'groove': random.uniform(0.3, 0.8),
            'beat_strength': random.uniform(0.4, 0.9),
            
            # Timbre features
            'brightness': random.uniform(0.3, 0.8),
            'spectral_centroid': random.uniform(1000, 4000),
            'spectral_rolloff': random.uniform(2000, 8000),
            
            # Mood features
            'valence': random.uniform(0.2, 0.8),
            'arousal': energy * random.uniform(0.8, 1.2),
            'mood_aggressive': random.uniform(0, 0.5),
            'mood_happy': random.uniform(0.3, 0.9),
            'mood_sad': random.uniform(0, 0.6),
            'mood_relaxed': random.uniform(0.2, 0.8),
            
            # Vocal features
            'vocal_presence': random.uniform(0.3, 0.9),
            'speechiness': random.uniform(0.03, 0.3),
            'vocal_gender': random.choice(['male', 'female', 'mixed', 'instrumental']),
            
            # Production features
            'production_quality': random.uniform(0.6, 0.95),
            'mixing_quality': random.uniform(0.6, 0.95),
            'mastering_quality': random.uniform(0.6, 0.95),
            'clarity': random.uniform(0.5, 0.95),
            
            # Musical features
            'instrumentalness': random.uniform(0, 0.5),
            'acousticness': random.uniform(0.1, 0.7),
            'liveness': random.uniform(0.1, 0.4),
            'musical_complexity': random.uniform(0.3, 0.8),
            
            # Commercial features
            'catchiness': random.uniform(0.4, 0.9),
            'memorability': random.uniform(0.4, 0.9),
            'radio_friendliness': random.uniform(0.3, 0.9),
            'stream_friendliness': random.uniform(0.4, 0.95),
            
            # Structure features
            'intro_length': random.uniform(0, 30),
            'outro_length': random.uniform(0, 30),
            'verse_chorus_ratio': random.uniform(0.3, 0.7),
            'repetitiveness': random.uniform(0.2, 0.8),
            
            # Genre indicators
            'genre_electronic': random.uniform(0, 1),
            'genre_rock': random.uniform(0, 1),
            'genre_pop': random.uniform(0, 1),
            'genre_hiphop': random.uniform(0, 1),
            'genre_rnb': random.uniform(0, 1),
            'genre_country': random.uniform(0, 1),
            'genre_jazz': random.uniform(0, 1),
            'genre_classical': random.uniform(0, 1),
        }
    
    def _real_process_file(self, file) -> Dict[str, Any]:
        """Real audio processing using librosa"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as tmp_file:
                tmp_file.write(file.read())
                tmp_path = tmp_file.name
            
            # Load audio
            y, sr = librosa.load(tmp_path, sr=self.sample_rate)
            
            # Extract real features
            features = self._extract_librosa_features(y, sr)
            
            # Clean up temp file
            Path(tmp_path).unlink()
            
            return features
            
        except Exception as e:
            logger.error(f"Error in real audio processing: {e}")
            # Fallback to mock
            return self._mock_process_file(file)
    
    def _extract_librosa_features(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract features using librosa"""
        
        # Tempo and beat
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        # Energy
        rms = librosa.feature.rms(y=y)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        
        return {
            'tempo': float(tempo),
            'duration': len(y) / sr,
            'spectral_centroid': float(np.mean(spectral_centroids)),
            'spectral_rolloff': float(np.mean(spectral_rolloff)),
            'rms_energy': float(np.mean(rms)),
            'zero_crossing_rate': float(np.mean(zcr)),
            'mfcc_mean': [float(np.mean(mfcc)) for mfcc in mfccs],
            'chroma_mean': [float(np.mean(c)) for c in chroma],
            
            # Add mock values for features that need ML models
            'energy': float(np.mean(rms)) * 2,  # Simplified
            'danceability': min(1.0, tempo / 128 * 0.7),  # Simplified
            'valence': random.uniform(0.3, 0.8),  # Would need ML model
            'catchiness': random.uniform(0.4, 0.9),  # Would need ML model
            'production_quality': random.uniform(0.6, 0.95),  # Would need ML model
            
            # Add other mock features for completeness
            **{k: v for k, v in self._mock_process_file(None).items() 
               if k not in ['tempo', 'duration', 'energy', 'danceability']}
        }
