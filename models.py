import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os

class AnomalyDetectionEnsemble:
    """
    Ensemble anomaly detection system combining:
    - Isolation Forest
    - One-Class SVM
    - Local Outlier Factor
    - Deep Autoencoder
    """
    
    def __init__(self, contamination=0.1):
        """
        Initialize the ensemble with all models.
        
        Args:
            contamination: Expected proportion of anomalies in the dataset
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        
        # Initialize classical models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        self.one_class_svm = OneClassSVM(
            nu=contamination,
            kernel='rbf',
            gamma='auto'
        )
        
        self.lof = LocalOutlierFactor(
            contamination=contamination,
            novelty=True,
            n_neighbors=20
        )
        
        # Autoencoder will be built based on input dimension
        self.autoencoder = None
        self.encoder = None
        self.input_dim = None
        self.trained = False
        
    def build_autoencoder(self, input_dim):
        """Build deep autoencoder architecture."""
        self.input_dim = input_dim
        
        # Encoder
        encoder_input = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(64, activation='relu')(encoder_input)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(32, activation='relu')(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(16, activation='relu')(encoded)
        encoded = layers.Dense(8, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(16, activation='relu')(encoded)
        decoded = layers.Dense(32, activation='relu')(decoded)
        decoded = layers.Dense(64, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Full autoencoder
        self.autoencoder = keras.Model(encoder_input, decoded)
        self.encoder = keras.Model(encoder_input, encoded)
        
        self.autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
    def train(self, X, epochs=50, batch_size=32, verbose=0):
        """
        Train all models in the ensemble.
        
        Args:
            X: Training data (normal traffic only recommended)
            epochs: Number of epochs for autoencoder training
            batch_size: Batch size for autoencoder
            verbose: Verbosity level
        """
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Build autoencoder if not already built
        if self.autoencoder is None:
            self.build_autoencoder(X_scaled.shape[1])
        
        # Train classical models
        print("Training Isolation Forest...")
        self.isolation_forest.fit(X_scaled)
        
        print("Training One-Class SVM...")
        self.one_class_svm.fit(X_scaled)
        
        print("Training Local Outlier Factor...")
        self.lof.fit(X_scaled)
        
        # Train autoencoder
        print("Training Autoencoder...")
        history = self.autoencoder.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=verbose,
            shuffle=True
        )
        
        self.trained = True
        print("All models trained successfully!")
        
        return history
    
    def predict_individual(self, X):
        """
        Get predictions from each individual model.
        
        Returns:
            Dictionary with predictions from each model (-1 for anomaly, 1 for normal)
        """
        if not self.trained:
            raise ValueError("Models must be trained before prediction!")
        
        X_scaled = self.scaler.transform(X)
        
        predictions = {
            'isolation_forest': self.isolation_forest.predict(X_scaled),
            'one_class_svm': self.one_class_svm.predict(X_scaled),
            'lof': self.lof.predict(X_scaled)
        }
        
        # Autoencoder prediction based on reconstruction error
        reconstructed = self.autoencoder.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        threshold = np.percentile(mse, (1 - self.contamination) * 100)
        predictions['autoencoder'] = np.where(mse > threshold, -1, 1)
        
        return predictions
    
    def predict_ensemble(self, X):
        """
        Get ensemble prediction using majority voting.
        
        Returns:
            predictions: -1 for anomaly, 1 for normal
            confidence: Confidence score (0-1)
            severity: Severity level (low/medium/high)
        """
        individual_preds = self.predict_individual(X)
        
        # Stack all predictions
        all_preds = np.column_stack([
            individual_preds['isolation_forest'],
            individual_preds['one_class_svm'],
            individual_preds['lof'],
            individual_preds['autoencoder']
        ])
        
        # Majority voting
        anomaly_votes = np.sum(all_preds == -1, axis=1)
        ensemble_pred = np.where(anomaly_votes >= 2, -1, 1)
        
        # Confidence: proportion of models agreeing
        confidence = np.maximum(anomaly_votes, 4 - anomaly_votes) / 4.0
        
        # Severity: based on number of models detecting anomaly
        severity = np.array(['normal'] * len(X))
        severity[anomaly_votes == 2] = 'low'
        severity[anomaly_votes == 3] = 'medium'
        severity[anomaly_votes == 4] = 'high'
        
        return ensemble_pred, confidence, severity, individual_preds
    
    def get_anomaly_scores(self, X):
        """Get anomaly scores from each model."""
        X_scaled = self.scaler.transform(X)
        
        scores = {}
        
        # Isolation Forest scores
        scores['isolation_forest'] = -self.isolation_forest.score_samples(X_scaled)
        
        # One-Class SVM scores
        scores['one_class_svm'] = -self.one_class_svm.score_samples(X_scaled)
        
        # LOF scores
        scores['lof'] = -self.lof.score_samples(X_scaled)
        
        # Autoencoder reconstruction error
        reconstructed = self.autoencoder.predict(X_scaled, verbose=0)
        scores['autoencoder'] = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        
        return scores
    
    def save_models(self, directory='models'):
        """Save all models to disk."""
        os.makedirs(directory, exist_ok=True)
        
        # Save classical models
        joblib.dump(self.isolation_forest, f'{directory}/isolation_forest.pkl')
        joblib.dump(self.one_class_svm, f'{directory}/one_class_svm.pkl')
        joblib.dump(self.lof, f'{directory}/lof.pkl')
        joblib.dump(self.scaler, f'{directory}/scaler.pkl')
        
        # Save autoencoder
        self.autoencoder.save(f'{directory}/autoencoder.keras')
        
        print(f"Models saved to {directory}/")
    
    def load_models(self, directory='models'):
        """Load all models from disk."""
        self.isolation_forest = joblib.load(f'{directory}/isolation_forest.pkl')
        self.one_class_svm = joblib.load(f'{directory}/one_class_svm.pkl')
        self.lof = joblib.load(f'{directory}/lof.pkl')
        self.scaler = joblib.load(f'{directory}/scaler.pkl')
        self.autoencoder = keras.models.load_model(f'{directory}/autoencoder.keras')
        
        self.trained = True
        print(f"Models loaded from {directory}/")


def calculate_feature_importance(model_ensemble, X, feature_names):
    """
    Calculate feature importance using permutation-based method.
    This is a simplified version since SHAP has compatibility issues.
    """
    X_scaled = model_ensemble.scaler.transform(X)
    
    # Get baseline predictions
    baseline_preds, _, _, _ = model_ensemble.predict_ensemble(X)
    baseline_anomalies = np.sum(baseline_preds == -1)
    
    importance_scores = []
    
    for i, feature_name in enumerate(feature_names):
        # Permute feature
        X_permuted = X_scaled.copy()
        np.random.shuffle(X_permuted[:, i])
        
        # Inverse transform for prediction
        X_permuted_original = model_ensemble.scaler.inverse_transform(X_permuted)
        
        # Get predictions with permuted feature
        permuted_preds, _, _, _ = model_ensemble.predict_ensemble(X_permuted_original)
        permuted_anomalies = np.sum(permuted_preds == -1)
        
        # Importance = change in anomaly detection
        importance = abs(permuted_anomalies - baseline_anomalies) / len(X)
        importance_scores.append(importance)
    
    # Normalize scores
    importance_scores = np.array(importance_scores)
    if importance_scores.sum() > 0:
        importance_scores = importance_scores / importance_scores.sum()
    
    return dict(zip(feature_names, importance_scores))
