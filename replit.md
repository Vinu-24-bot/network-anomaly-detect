# AI-Powered Network Anomaly Detection System

## Overview

This project is an AI-powered network anomaly detection system that identifies malicious activities in network traffic using machine learning. The system employs an ensemble approach combining four different models: Isolation Forest, One-Class SVM, Local Outlier Factor, and a Deep Autoencoder. It provides real-time detection capabilities through an interactive Streamlit dashboard, allowing users to upload network traffic data, view detection results, and provide feedback for continuous improvement. The system includes explainability features, severity scoring for detected anomalies, and comprehensive performance metrics visualization.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture

**Technology**: Streamlit web framework

**Design Pattern**: Single-page application with interactive dashboard

**Rationale**: Streamlit was chosen for rapid prototyping and deployment of ML applications. It provides built-in components for data visualization, file uploads, and real-time updates without requiring frontend framework expertise.

**Key Components**:
- Interactive dashboard with metrics display
- File upload interface for CSV network traffic data
- Real-time visualization using Plotly and Matplotlib
- Feedback collection mechanism for model improvement

### Machine Learning Architecture

**Approach**: Ensemble learning with unsupervised anomaly detection

**Problem Addressed**: Detecting rare, high-impact malicious network activities in high-volume traffic without labeled training data

**Solution**: Combines four complementary anomaly detection algorithms:

1. **Isolation Forest** - Tree-based anomaly detection for global patterns
2. **One-Class SVM** - Support vector approach for boundary detection
3. **Local Outlier Factor (LOF)** - Density-based local anomaly detection
4. **Deep Autoencoder** - Neural network for learning complex feature representations

**Rationale**: 
- Each algorithm has different strengths and weaknesses
- Ensemble voting reduces false positives
- Multiple perspectives improve detection of diverse attack types
- Unsupervised approach generalizes to unknown threats

**Pros**:
- Robust to individual model failures
- No labeled data required
- Detects novel attack patterns
- Explainable through feature importance

**Cons**:
- Higher computational cost
- More complex to tune and maintain
- Potential for conflicting predictions

### Data Processing Pipeline

**Preprocessing**: StandardScaler normalization applied to all features before model training

**Feature Set**: Network traffic attributes including:
- Packet characteristics (size, rate)
- Connection metadata (duration, bytes transferred)
- Protocol information (type, flags, ports)
- Error metrics (error rate)

**Anomaly Scoring**: Majority voting ensemble with severity classification:
- **Ensemble Decision**: Anomaly detected when ≥2 out of 4 models agree
- **Severity Levels**:
  - Low: 2 models agree
  - Medium: 3 models agree  
  - High: 4 models agree (all models detected the anomaly)
- **Confidence Score**: Proportion of models in agreement (0.0 to 1.0)

**Important Note**: The final ensemble anomaly count is typically different from any individual model's count because it's based on majority voting. For example, if Model A detects 200 anomalies, Model B detects 180, Model C detects 150, and Model D detects 190, the ensemble might detect 174 anomalies where at least 2 models agreed.

### Model Persistence

**Storage Format**: Joblib serialization for scikit-learn models, native Keras format for deep learning

**Directory Structure**: Models stored in `models/` directory with separate files per component

**Rationale**: Allows pre-trained models to be loaded for inference without retraining, enabling faster deployment and consistent predictions

## External Dependencies

### Machine Learning Frameworks

- **scikit-learn**: Classical ML algorithms (Isolation Forest, One-Class SVM, LOF) and preprocessing utilities
- **TensorFlow/Keras**: Deep learning framework for Autoencoder neural network implementation
- **NumPy**: Numerical computing for array operations and data manipulation
- **pandas**: Data structure and analysis for handling network traffic datasets

### Visualization Libraries

- **Plotly**: Interactive plotting for real-time dashboard visualizations
- **Matplotlib**: Static plotting for detailed analysis charts
- **Seaborn**: Statistical visualization built on matplotlib

### Web Framework

- **Streamlit**: Web application framework for creating the interactive dashboard and user interface

### Utilities

- **joblib**: Model serialization and deserialization for persistence
- **datetime**: Timestamp handling for network traffic data

### Data Generation

**Synthetic Data**: `generate_sample_data.py` creates realistic network traffic with configurable anomaly ratios for testing and demonstration purposes

**Rationale**: Allows system testing without access to real network traffic data, which may be sensitive or unavailable