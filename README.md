# 🔒 AI-Powered Network Anomaly Detection System

## Ensemble Classical Models + Deep Autoencoder

CLICK HERE TO SEE THE PROJECT : https://network-anomaly-detect-4k6njuhwikfdv8yr2lgrcs.streamlit.app/

A comprehensive, end-to-end network anomaly detection system designed for detecting malicious activities in network traffic using an ensemble of machine learning models combined with deep learning.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Performance Metrics](#performance-metrics)
- [Deployment](#deployment)
- [Future Enhancements](#future-enhancements)

---

## 🎯 Overview

This project implements an **AI-powered network anomaly detection system** that combines multiple machine learning approaches to identify malicious network activities. The system uses an ensemble of four different models to provide robust, accurate, and explainable anomaly detection.

### Key Highlights

✅ **Ensemble Approach**: Combines 4 different models for robust detection  
✅ **Real-time Detection**: Process streaming network traffic data  
✅ **Explainable AI**: Understand why anomalies are detected  
✅ **Interactive Dashboard**: User-friendly Streamlit interface  
✅ **Feedback Loop**: Continuous improvement through user feedback  
✅ **Comprehensive Metrics**: Detailed performance evaluation  
✅ **Severity Scoring**: Classify anomalies by severity (low/medium/high)  

---

## 📝 Problem Statement

Modern networks generate high-volume, heterogeneous traffic where malicious activities are rare but high-impact. Manual monitoring is infeasible, and traditional rule-based systems are fragile to unknown attack patterns.

**Challenge**: Design an unsupervised, real-time anomaly detection system that:
- Generalizes to unseen threats
- Supports explainability
- Adapts using operator feedback
- Reduces false positives over time

**Solution**: A deployable dashboard that ingests CSV/stream data, detects anomalies with an ensemble of Isolation Forest, One-Class SVM, Local Outlier Factor, and an Autoencoder, and retrains from feedback to improve performance iteratively.

---

## 🏗️ System Architecture

### Ensemble Components

The system combines four complementary anomaly detection approaches:

#### 1. **Isolation Forest**
- **Type**: Tree-based ensemble method
- **How it works**: Isolates anomalies by randomly selecting features and split values
- **Strengths**: Fast, efficient, handles high-dimensional data
- **Use case**: Detecting global outliers

#### 2. **One-Class SVM**
- **Type**: Support Vector Machine
- **How it works**: Learns the boundary of normal behavior in feature space
- **Strengths**: Good generalization, handles non-linear patterns
- **Use case**: Learning complex decision boundaries

#### 3. **Local Outlier Factor (LOF)**
- **Type**: Density-based method
- **How it works**: Identifies local density deviations
- **Strengths**: Detects anomalies in varying density regions
- **Use case**: Finding local outliers missed by global methods

#### 4. **Deep Autoencoder**
- **Type**: Neural network (unsupervised deep learning)
- **Architecture**: 
  - Encoder: Input → 64 → 32 → 16 → 8 (bottleneck)
  - Decoder: 8 → 16 → 32 → 64 → Output
- **How it works**: Learns to reconstruct normal traffic; high reconstruction error indicates anomaly
- **Strengths**: Captures complex non-linear patterns
- **Use case**: Deep feature learning and pattern recognition

### Ensemble Voting System

- **Decision Rule**: Majority voting (≥2 models agree → anomaly)
- **Confidence Score**: Proportion of models agreeing (0.0 to 1.0)
- **Severity Classification**:
  - **Low**: 2 models detect anomaly
  - **Medium**: 3 models detect anomaly
  - **High**: All 4 models detect anomaly

---

## ✨ Features

### 1. Data Management
- 📁 Upload custom network traffic CSV files
- 📊 Load sample dataset for testing
- 🔍 Dataset overview and statistics

### 2. Model Training
- 🤖 Train ensemble of 4 models simultaneously
- ⚙️ Configurable contamination ratio
- 💾 Model persistence (save/load trained models)
- 📈 Training progress tracking

### 3. Anomaly Detection
- 🔍 Real-time anomaly detection
- 🎯 Confidence scoring (0-1 scale)
- 🚨 Severity classification (low/medium/high)
- 📊 Individual model predictions
- 📉 Comprehensive visualizations

### 4. Visualizations
- 📈 **Timeline View**: Anomalies plotted over time
- 🔢 **Model Comparison**: Compare individual model performance
- 🥧 **Severity Distribution**: Breakdown of anomaly severity
- 📊 **Feature Distributions**: Normal vs anomaly feature patterns
- 🔥 **Confusion Matrix**: Performance evaluation (when labels available)

### 5. Explainability
- 🔬 Feature importance analysis
- 💡 Permutation-based importance scoring
- 🔎 Individual sample analysis
- 🗳️ Model voting breakdown

### 6. Feedback Loop
- 🏷️ Label anomalies as true/false positives
- 🔄 Retrain models with corrected labels
- 📈 Iterative performance improvement
- 📊 Feedback statistics

### 7. Export & Reporting
- 📥 Download anomaly reports (CSV)
- 📄 Timestamped reports
- 📊 Detailed anomaly information

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- pip or uv package manager

### Clone Repository
```bash
git clone <your-repo-url>
cd network-anomaly-detection
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

Or using the packager:
```bash
uv sync
```

### Dependencies
- `streamlit` - Web dashboard
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - ML models (IF, SVM, LOF)
- `tensorflow` - Deep learning (Autoencoder)
- `plotly` - Interactive visualizations
- `matplotlib` - Static plots
- `seaborn` - Statistical visualizations
- `joblib` - Model persistence

---

## 📖 Usage Guide

### Step 1: Generate Sample Data (Optional)

If you want to test with sample data:

```bash
python generate_sample_data.py
```

This creates `sample_network_traffic.csv` with 1000 samples (15% anomalies).

### Step 2: Run the Application

```bash
streamlit run app.py --server.port 5000
```

The app will open at `http://localhost:5000`

### Step 3: Upload Data

1. Navigate to **🏠 Home & Upload** page
2. Either:
   - Upload your own CSV file with network traffic data
   - Click "Load Sample Network Traffic Data" to use the sample dataset

### Step 4: Train Models

1. Go to **🤖 Model Training** page
2. Configure settings:
   - **Expected Anomaly Ratio**: Proportion of anomalies (default: 0.15)
   - **Train on full dataset**: Use all data or only normal traffic
3. Click **"🚀 Train Ensemble Models"**
4. Wait for training to complete (~1-2 minutes)

### Step 5: Detect Anomalies

1. Navigate to **📊 Detection & Analysis** page
2. Click **"🔍 Detect Anomalies"**
3. View results:
   - Detection summary (total anomalies, rate, confidence)
   - Timeline visualization
   - Model comparison
   - Severity distribution
   - Feature distributions
   - Detected anomalies table
4. Download anomaly report if needed

### Step 6: Understand Results (Explainability)

1. Go to **🔍 Explainability** page
2. Click **"🔬 Calculate Feature Importance"**
3. View:
   - Feature importance ranking
   - Which features contribute most to detection
4. Select individual anomalies to analyze:
   - See which models detected it
   - View feature values
   - Understand voting breakdown

### Step 7: Provide Feedback (Optional)

1. Navigate to **📝 Feedback & Retraining** page
2. Review detected anomalies
3. Label each as:
   - ✅ **True Positive**: Correctly identified anomaly
   - ❌ **False Positive**: Incorrectly flagged as anomaly
4. Click **"🔄 Retrain Model with Feedback"**
5. Model improves based on your corrections

---

## 🧠 Model Details

### Network Traffic Features

The system analyzes 10 key features:

| Feature | Description | Normal Range | Anomaly Indicators |
|---------|-------------|--------------|-------------------|
| `packet_size` | Network packet size (bytes) | 400-650 | Very small (<200) or large (>1200) |
| `duration` | Connection duration (sec) | 0.5-5 | Very short (<0.1) or long (>10) |
| `src_bytes` | Bytes from source | 3500-6500 | Very small (<500) or large (>30000) |
| `dst_bytes` | Bytes to destination | 3500-6500 | Very small (<500) or large (>30000) |
| `protocol_type` | Protocol (0=TCP, 1=UDP, 2=ICMP) | Mostly TCP/UDP | Excessive ICMP |
| `flag` | Connection flag | 0-2 common | Unusual flags (3-5) |
| `src_port` | Source port | >1024 | Privileged ports (<1024) |
| `dst_port` | Destination port | 80, 443, 22, etc. | Random high ports |
| `packet_rate` | Packets per second | 35-65 | Very low (<5) or high (>300) |
| `error_rate` | Error percentage | 0.2-0.8% | High error rate (>10%) |

### Training Process

1. **Data Preprocessing**
   - Feature scaling using StandardScaler
   - Z-score normalization

2. **Model Training**
   - Isolation Forest: 100 estimators
   - One-Class SVM: RBF kernel
   - LOF: 20 neighbors
   - Autoencoder: 30 epochs, batch size 32

3. **Model Persistence**
   - Saved to `models/` directory
   - Can be loaded for future use

### Prediction Process

1. **Individual Predictions**
   - Each model predicts independently
   - Returns -1 (anomaly) or 1 (normal)

2. **Ensemble Aggregation**
   - Count votes from all 4 models
   - ≥2 votes → classify as anomaly

3. **Confidence & Severity**
   - Confidence = agreement ratio
   - Severity based on vote count

---

## 📊 Dataset

### Sample Dataset

The included sample dataset (`sample_network_traffic.csv`) contains:
- **Total samples**: 1000
- **Normal traffic**: ~850 samples (85%)
- **Anomalous traffic**: ~150 samples (15%)

### Required CSV Format

Your custom dataset should have these columns:

```csv
packet_size,duration,src_bytes,dst_bytes,protocol_type,flag,src_port,dst_port,packet_rate,error_rate,label
512.5,2.3,5000,4800,0,1,45678,80,50.2,0.5,0
1400.2,15.7,45000,42000,2,4,500,8080,450.8,18.5,1
...
```

- All features are **required**
- `label` column is **optional** (0=normal, 1=anomaly)
- Used only for performance evaluation

### Anomaly Types in Sample Data

1. **Large Packet Attacks**: Packet size >1200 bytes
2. **Tiny Packet Floods**: Packet size <200 bytes
3. **Long Connections**: Duration >10 seconds
4. **Flash Connections**: Duration <0.1 seconds
5. **Data Exfiltration**: Very high src_bytes/dst_bytes
6. **Port Scanning**: Unusual port combinations
7. **Protocol Anomalies**: Excessive ICMP traffic
8. **High Error Rates**: Error rate >10%

---

## 📁 Project Structure

```
network-anomaly-detection/
│
├── app.py                          # Main Streamlit application
├── models.py                       # Anomaly detection models & ensemble
├── generate_sample_data.py         # Sample dataset generator
├── README.md                       # This file
├── requirements.txt                # Python dependencies
│
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
│
├── models/                         # Saved trained models (auto-generated)
│   ├── isolation_forest.pkl
│   ├── one_class_svm.pkl
│   ├── lof.pkl
│   ├── scaler.pkl
│   └── autoencoder.keras
│
└── sample_network_traffic.csv      # Sample dataset (generated)
```

---

## 📈 Performance Metrics

When ground truth labels are available, the system calculates:

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of Precision and Recall

### Confusion Matrix
- True Positives (TP): Correctly identified anomalies
- True Negatives (TN): Correctly identified normal traffic
- False Positives (FP): Normal traffic flagged as anomaly
- False Negatives (FN): Missed anomalies

### Expected Performance
On the sample dataset:
- **Accuracy**: ~92-95%
- **Precision**: ~85-90%
- **Recall**: ~90-95%
- **F1-Score**: ~87-92%

---

## 🚀 Deployment

### Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Select your repository
5. Set main file: `app.py`
6. Deploy!

Your app will be live at: `https://your-app-name.streamlit.app`

### Replit

Already configured! Just click "Run" or use:
```bash
streamlit run app.py --server.port 5000
```

Then publish using the Replit deployment feature.

### Local Deployment

```bash
streamlit run app.py --server.port 5000
```

Access at: `http://localhost:5000`

---

## 🔮 Future Enhancements

### Planned Features
- [ ] Real-time streaming from PCAP files
- [ ] Advanced feature engineering (flow-based features)
- [ ] Database persistence for anomaly history
- [ ] Email/SMS alerts for high-severity anomalies
- [ ] Multi-user authentication
- [ ] Model versioning and A/B testing
- [ ] Integration with SIEM systems
- [ ] Advanced SHAP-based explainability
- [ ] Automated report generation
- [ ] Time-series forecasting

### Research Directions
- [ ] Deep learning ensemble (CNN + LSTM)
- [ ] Graph neural networks for network topology
- [ ] Federated learning for distributed networks
- [ ] Adversarial robustness testing

---

## 🎓 For College Project

This project demonstrates proficiency in:

### Machine Learning
- Ensemble methods
- Supervised & unsupervised learning
- Model evaluation and metrics
- Feature engineering

### Deep Learning
- Autoencoder architecture
- Neural network training
- Loss functions and optimization

### Data Science
- Data preprocessing and normalization
- Statistical analysis
- Visualization and interpretation
- Explainable AI

### Software Engineering
- Modular code design
- Clean architecture
- Version control (Git)
- Documentation

### Full-Stack Development
- Interactive web applications
- UI/UX design
- Real-time data processing
- Deployment

---

## 📚 References

1. **Isolation Forest**: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In ICDM.
2. **One-Class SVM**: Schölkopf, B., et al. (2001). Estimating the support of a high-dimensional distribution.
3. **LOF**: Breunig, M. M., et al. (2000). LOF: identifying density-based local outliers.
4. **Autoencoders**: Sakurada, M., & Yairi, T. (2014). Anomaly detection using autoencoders with nonlinear dimensionality reduction.

---

## 👨‍💻 Author

**College Minor Project**  
**Topic**: Network Security & Anomaly Detection  
**Year**: 2025  

---

## 📄 License

This project is created for educational purposes as part of a college minor project.

---

## 🙏 Acknowledgments

- Scikit-learn for classical ML algorithms
- TensorFlow/Keras for deep learning
- Streamlit for the amazing dashboard framework
- The open-source community

---

## 📞 Support

For questions or issues:
1. Check the **📖 Documentation** page in the app
2. Review this README
3. Check the code comments for implementation details

---

**Happy Anomaly Hunting! 🔍🛡️**
