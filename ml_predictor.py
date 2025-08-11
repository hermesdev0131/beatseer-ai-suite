import random
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

# Try to import sklearn
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available. Using mock predictions.")

class MLPredictor:
    """ML predictor for music analysis"""
    
    def __init__(self, model_path: Optional[Path] = None, use_mock: bool = False):
        self.model_path = model_path
        self.use_mock = use_mock or not SKLEARN_AVAILABLE
        self.models_loaded = False
        
        if not self.use_mock:
            self._load_models()
        else:
            logger.info("Using mock ML predictor")
    
    def _load_models(self):
        """Load or create ML models"""
        if self.model_path and self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.models = pickle.load(f)
                self.models_loaded = True
                logger.info("Loaded existing models")
            except Exception as e:
                logger.error(f"Error loading models: {e}")
                self._create_mock_models()
        else:
            self._create_mock_models()
    
    def _create_mock_models(self):
        """Create mock models for demonstration"""
        logger.info("Creating mock models")
        self.models = {
            'hit_predictor': None,  # Would be trained model
            'genre_classifier': None,  # Would be trained model
            'scaler': StandardScaler() if SKLEARN_AVAILABLE else None
        }
        self.models_loaded = True
    
    def predict(self, features: Dict[str, Any], role: str = 'artist') -> Dict[str, Any]:
        """
        Make predictions based on audio features
        
        Args:
            features: Dictionary of audio features
            role: User role (affects prediction weighting)
            
        Returns:
            Dictionary of predictions
        """
        if self.use_mock or not self.models_loaded:
            return self._mock_predict(features, role)
        else:
            return self._real_predict(features, role)
    
    def _mock_predict(self, features: Dict[str, Any], role: str) -> Dict[str, Any]:
        """Generate mock predictions"""
        
        # Base prediction on some "features"
        base_score = 50
        
        # Adjust based on features
        if features.get('energy', 0.5) > 0.7:
            base_score += 10
        if features.get('danceability', 0.5) > 0.7:
            base_score += 8
        if features.get('catchiness', 0.5) > 0.7:
            base_score += 12
        if features.get('production_quality', 0.7) > 0.8:
            base_score += 10
        if features.get('tempo', 120) > 110 and features.get('tempo', 120) < 130:
            base_score += 5
        
        # Add randomness for realism
        base_score += random.uniform(-10, 10)
        
        # Determine genre based on features
        genres = ['pop', 'rock', 'electronic', 'hip-hop', 'r&b', 'indie', 'alternative', 'dance']
        if features.get('genre_electronic', 0) > 0.6:
            predicted_genre = 'electronic'
        elif features.get('genre_hiphop', 0) > 0.6:
            predicted_genre = 'hip-hop'
        elif features.get('genre_rock', 0) > 0.6:
            predicted_genre = 'rock'
        else:
            predicted_genre = random.choice(genres)
        
        # Role-based adjustments
        if role == 'label':
            base_score *= 0.95  # Labels are more conservative
        elif role == 'producer':
            base_score *= 1.02  # Producers focus on technical quality
        
        # Ensure valid range
        hit_probability = min(95, max(25, base_score))
        
        return {
            'hit_probability': hit_probability,
            'predicted_genre': predicted_genre,
            'sub_genre': self._determine_subgenre(predicted_genre, features),
            'confidence': random.uniform(0.65, 0.95),
            'beatseer_score': min(100, max(0, hit_probability + random.uniform(-5, 5))),
            'commercial_potential': min(100, max(0, hit_probability + random.uniform(-8, 8))),
            
            # Market intelligence
            'market_intelligence': {
                'viral_score': self._calculate_viral_score(features),
                'playlist_potential': min(95, max(30, hit_probability + random.uniform(-10, 15))),
                'radio_friendliness': self._calculate_radio_friendliness(features),
                'streaming_projection': {
                    'month_1': int(hit_probability * random.uniform(100, 1500)),
                    'month_6': int(hit_probability * random.uniform(500, 7500)),
                    'year_1': int(hit_probability * random.uniform(1000, 25000))
                },
                'tiktok_potential': self._calculate_tiktok_potential(features),
                'sync_licensing_potential': self._calculate_sync_potential(features)
            },
            
            # Genre probabilities
            'genre_probabilities': self._calculate_genre_probabilities(features),
            
            # Quality metrics
            'quality_metrics': {
                'production': features.get('production_quality', 0.7) * 100,
                'mixing': features.get('mixing_quality', 0.7) * 100,
                'mastering': features.get('mastering_quality', 0.7) * 100,
                'overall': np.mean([
                    features.get('production_quality', 0.7),
                    features.get('mixing_quality', 0.7),
                    features.get('mastering_quality', 0.7)
                ]) * 100
            }
        }
    
    def _determine_subgenre(self, genre: str, features: Dict) -> str:
        """Determine sub-genre based on features"""
        subgenres = {
            'electronic': ['house', 'techno', 'dubstep', 'drum & bass', 'ambient'],
            'rock': ['alternative', 'indie rock', 'hard rock', 'punk', 'progressive'],
            'pop': ['dance pop', 'indie pop', 'electropop', 'teen pop', 'synth pop'],
            'hip-hop': ['trap', 'conscious', 'mumble rap', 'boom bap', 'cloud rap'],
            'r&b': ['contemporary r&b', 'alternative r&b', 'neo soul', 'funk', 'soul']
        }
        
        if genre in subgenres:
            return f"{genre} - {random.choice(subgenres[genre])}"
        return genre
    
    def _calculate_viral_score(self, features: Dict) -> float:
        """Calculate viral potential"""
        score = 50
        
        # Factors that increase viral potential
        if features.get('catchiness', 0.5) > 0.7:
            score += 20
        if features.get('danceability', 0.5) > 0.7:
            score += 15
        if features.get('energy', 0.5) > 0.7:
            score += 10
        if features.get('tempo', 120) > 120 and features.get('tempo', 120) < 140:
            score += 10
        
        # Short intros are better for viral
        if features.get('intro_length', 15) < 10:
            score += 5
        
        return min(95, max(20, score + random.uniform(-10, 10)))
    
    def _calculate_radio_friendliness(self, features: Dict) -> float:
        """Calculate radio friendliness"""
        score = 50
        
        # Radio-friendly characteristics
        if features.get('duration', 240) > 150 and features.get('duration', 240) < 240:
            score += 20  # Good duration
        if features.get('production_quality', 0.7) > 0.8:
            score += 15
        if features.get('vocal_presence', 0.5) > 0.6:
            score += 10
        if features.get('explicit_content', False) == False:
            score += 10
        
        return min(90, max(20, score + random.uniform(-5, 10)))
    
    def _calculate_tiktok_potential(self, features: Dict) -> float:
        """Calculate TikTok potential"""
        score = 40
        
        # TikTok-friendly characteristics
        if features.get('catchiness', 0.5) > 0.75:
            score += 25
        if features.get('danceability', 0.5) > 0.7:
            score += 20
        if features.get('tempo', 120) > 120 and features.get('tempo', 120) < 140:
            score += 15
        if features.get('energy', 0.5) > 0.6:
            score += 10
        
        return min(95, max(15, score + random.uniform(-5, 10)))
    
    def _calculate_sync_potential(self, features: Dict) -> Dict[str, float]:
        """Calculate sync licensing potential for different media"""
        return {
            'film_tv': self._calculate_film_tv_sync(features),
            'advertising': self._calculate_advertising_sync(features),
            'gaming': self._calculate_gaming_sync(features),
            'sports': self._calculate_sports_sync(features)
        }
    
    def _calculate_film_tv_sync(self, features: Dict) -> float:
        """Calculate film/TV sync potential"""
        score = 50
        if features.get('instrumentalness', 0.2) > 0.5:
            score += 20
        if features.get('mood_aggressive', 0.3) < 0.3:
            score += 10
        if features.get('production_quality', 0.7) > 0.8:
            score += 15
        return min(90, max(20, score + random.uniform(-5, 10)))
    
    def _calculate_advertising_sync(self, features: Dict) -> float:
        """Calculate advertising sync potential"""
        score = 50
        if features.get('valence', 0.5) > 0.6:
            score += 20  # Positive mood
        if features.get('energy', 0.5) > 0.6:
            score += 15
        if features.get('catchiness', 0.5) > 0.7:
            score += 15
        return min(90, max(20, score + random.uniform(-5, 10)))
    
    def _calculate_gaming_sync(self, features: Dict) -> float:
        """Calculate gaming sync potential"""
        score = 50
        if features.get('energy', 0.5) > 0.7:
            score += 20
        if features.get('tempo', 120) > 130:
            score += 15
        if features.get('genre_electronic', 0) > 0.5:
            score += 15
        return min(90, max(20, score + random.uniform(-5, 10)))
    
    def _calculate_sports_sync(self, features: Dict) -> float:
        """Calculate sports sync potential"""
        score = 50
        if features.get('energy', 0.5) > 0.75:
            score += 25
        if features.get('tempo', 120) > 120:
            score += 15
        if features.get('mood_aggressive', 0.3) > 0.4:
            score += 10
        return min(90, max(20, score + random.uniform(-5, 10)))
    
    def _calculate_genre_probabilities(self, features: Dict) -> Dict[str, float]:
        """Calculate genre probabilities"""
        # In a real implementation, this would use a trained classifier
        probs = {
            'pop': features.get('genre_pop', random.uniform(0, 1)),
            'rock': features.get('genre_rock', random.uniform(0, 1)),
            'electronic': features.get('genre_electronic', random.uniform(0, 1)),
            'hip-hop': features.get('genre_hiphop', random.uniform(0, 1)),
            'r&b': features.get('genre_rnb', random.uniform(0, 1)),
            'country': features.get('genre_country', random.uniform(0, 1)),
            'indie': random.uniform(0, 0.5),
            'alternative': random.uniform(0, 0.5)
        }
        
        # Normalize to sum to 1
        total = sum(probs.values())
        if total > 0:
            return {k: v/total for k, v in probs.items()}
        return probs
    
    def _real_predict(self, features: Dict[str, Any], role: str) -> Dict[str, Any]:
        """Real prediction using trained models (placeholder)"""
        # In production, this would use actual trained models
        # For now, return mock predictions
        return self._mock_predict(features, role)
